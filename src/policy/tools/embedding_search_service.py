"""Embedding search service (tool-side) with FAISS + nginx load balancing.

This service is intended to run on the retrieval/tool server where FAISS is installed.
The training/agent side only sends HTTP requests to this service.

Key design: DynamicBatchingEmbedder collects concurrent embed() calls within a short
time window and flushes them as a single large batch to the vLLM backend, maximising
GPU utilisation and reducing round-trip overhead.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import multiprocessing
import os
import socket
import sys
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.policy.tools.embedding_client import (
    _extract_embeddings,
    build_embeddings_url,
    load_corpus,
)

logger = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    query_list: list[str] | None = None
    k: int | list[int] | None = None
    # Backward compatibility for old callers.
    queries: list[str] | None = None
    topk: int | None = None
    return_scores: bool = True


class SearchResponse(BaseModel):
    result: list[list[dict[str, Any]]]


@dataclass
class EmbeddingSearchConfig:
    index_path: str
    corpus_path: str
    backend_url: str
    model: str
    timeout: int = 30
    max_retries: int = 2
    embed_batch_size: int = 256
    max_text_len: int = 4096
    faiss_gpu_id: int = 7
    faiss_use_gpu: bool = True
    faiss_use_float16: bool = False
    faiss_use_all_gpus: bool = False
    faiss_all_gpus_shard: bool = True
    # Dynamic batching window: wait up to this many ms to collect queries before flushing.
    batch_max_wait_ms: float = 20.0
    # Hard cap on total queries in one flush (prevents single oversized vLLM call).
    batch_max_queries: int = 4096
    # FAISS dynamic batching: short wait to merge concurrent search vectors into one call.
    faiss_batch_max_wait_ms: float = 5.0
    # Hard cap on total vectors in one FAISS flush.
    faiss_batch_max_vectors: int = 4096
    # Max concurrent sub-batches in flight to vLLM (prevents overwhelming backends).
    vllm_max_concurrent_batches: int = 4
    # Embedding dimension passed to vLLM as "dimensions" field (None = model default).
    dimensions: int | None = 4096


class DynamicBatchingEmbedder:
    """Collects concurrent embed() calls and flushes them as one batch to vLLM.

    Instead of 512 independent HTTP requests each carrying 1-3 queries, all
    concurrent requests are merged into a single large embedding call.  This
    keeps the vLLM GPU at high utilisation and avoids connection-exhaustion on
    the backend.
    """

    def __init__(self, config: EmbeddingSearchConfig) -> None:
        self.config = config
        self._url = build_embeddings_url(config.backend_url)
        self._queue: asyncio.Queue[tuple[list[str], asyncio.Future]] = asyncio.Queue()
        self._session: aiohttp.ClientSession | None = None
        self._task: asyncio.Task | None = None
        self._vllm_semaphore = asyncio.Semaphore(config.vllm_max_concurrent_batches)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def embed(self, queries: list[str]) -> np.ndarray:
        """Submit *queries*; await and return float32 embedding matrix [N, D]."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[np.ndarray] = loop.create_future()
        await self._queue.put((queries, future))
        return await future

    # ------------------------------------------------------------------
    # Internal flush loop
    # ------------------------------------------------------------------

    async def _flush_loop(self) -> None:
        max_wait = self.config.batch_max_wait_ms / 1000.0
        max_queries = self.config.batch_max_queries

        while True:
            # Block until the first item arrives (long timeout avoids busy-loop).
            try:
                first = await asyncio.wait_for(self._queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                return

            items: list[tuple[list[str], asyncio.Future]] = [first]
            total_queries = len(first[0])

            # Drain more items that arrive within the batching window.
            deadline = asyncio.get_event_loop().time() + max_wait
            while total_queries < max_queries:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    items.append(item)
                    total_queries += len(item[0])
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    break

            # Merge all queries into one flat list and track per-request offsets.
            all_queries: list[str] = []
            offsets: list[int] = []
            for queries, _ in items:
                offsets.append(len(all_queries))
                all_queries.extend(queries)

            logger.debug(
                "Flushing batch: %d requests → %d queries total",
                len(items),
                len(all_queries),
            )

            # Single vLLM call for the whole batch.
            try:
                all_embeddings = await self._call_vllm(all_queries)
                for i, (queries, fut) in enumerate(items):
                    if not fut.done():
                        start = offsets[i]
                        arr = np.array(
                            all_embeddings[start : start + len(queries)],
                            dtype=np.float32,
                        )
                        fut.set_result(arr)
            except Exception as exc:
                for _, fut in items:
                    if not fut.done():
                        fut.set_exception(exc)

    # ------------------------------------------------------------------
    # vLLM async HTTP
    # ------------------------------------------------------------------

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=0),
            )
        return self._session

    async def _call_vllm(self, queries: list[str]) -> list[list[float]]:
        """Embed *queries* via vLLM, sending sub-batches with bounded concurrency.

        A semaphore limits how many sub-batches are in flight at once, preventing
        any single vLLM backend from receiving more sequences than it can handle.
        """
        session = self._get_session()
        batches = [
            queries[i : i + self.config.embed_batch_size]
            for i in range(0, len(queries), self.config.embed_batch_size)
        ]
        payloads = [{"model": self.config.model, "input": b} for b in batches]
        if self.config.dimensions is not None:
            for p in payloads:
                p["dimensions"] = self.config.dimensions

        async def _guarded_post(payload: dict, expected: int) -> list[list[float]]:
            async with self._vllm_semaphore:
                return await self._post_with_retry(session, payload, expected)

        results = await asyncio.gather(
            *[_guarded_post(p, len(b)) for p, b in zip(payloads, batches)]
        )
        return [emb for batch_result in results for emb in batch_result]

    async def _post_with_retry(
        self,
        session: aiohttp.ClientSession,
        payload: dict,
        expected: int,
    ) -> list[list[float]]:
        retryable_status = {429, 500, 502, 503, 504}
        for attempt in range(self.config.max_retries + 1):
            # Increase timeout on each retry to tolerate temporarily slow servers.
            timeout = aiohttp.ClientTimeout(
                total=self.config.timeout * (attempt + 1)
            )
            try:
                async with session.post(
                    self._url, json=payload, timeout=timeout
                ) as resp:
                    if resp.ok:
                        data = await resp.json()
                        return _extract_embeddings(data, expected)
                    if resp.status in retryable_status and attempt < self.config.max_retries:
                        await asyncio.sleep(0.3 * (2**attempt))
                        continue
                    text = await resp.text()
                    raise RuntimeError(
                        f"vLLM returned HTTP {resp.status}: {text[:300]}"
                    )
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt < self.config.max_retries:
                    logger.warning(
                        "Embedding attempt %d/%d failed (%s), retrying...",
                        attempt + 1,
                        self.config.max_retries + 1,
                        exc,
                    )
                    await asyncio.sleep(0.3 * (2**attempt))
                    continue
                raise
        raise RuntimeError("Embedding request failed after all retries.")


class DynamicBatchingFAISSSearcher:
    """Collects concurrent FAISS search calls and flushes them as one batch.

    Without batching, N concurrent requests each call faiss.search() with a few
    vectors, resulting in N small GPU kernel launches.  This batcher merges them
    into a single faiss.search() call with all vectors concatenated, maximising
    GPU matrix-multiply throughput.
    """

    def __init__(self, search_fn: Any, max_wait_ms: float = 5.0, max_vectors: int = 4096) -> None:
        self._search_fn = search_fn  # (np.ndarray, int) -> (np.ndarray, np.ndarray)
        self._max_wait = max_wait_ms / 1000.0
        self._max_vectors = max_vectors
        self._queue: asyncio.Queue[tuple[np.ndarray, int, asyncio.Future]] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def search(self, vectors: np.ndarray, topk: int) -> tuple[np.ndarray, np.ndarray]:
        """Submit vectors for batched FAISS search; returns (distances, indices)."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[np.ndarray, np.ndarray]] = loop.create_future()
        await self._queue.put((vectors, topk, future))
        return await future

    async def _flush_loop(self) -> None:
        while True:
            try:
                first = await asyncio.wait_for(self._queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                return

            items: list[tuple[np.ndarray, int, asyncio.Future]] = [first]
            total_vectors = first[0].shape[0]

            deadline = asyncio.get_event_loop().time() + self._max_wait
            while total_vectors < self._max_vectors:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    items.append(item)
                    total_vectors += item[0].shape[0]
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    break

            all_vectors = np.concatenate([v for v, _, _ in items], axis=0)
            merged_topk = max(k for _, k, _ in items)

            logger.debug(
                "FAISS flush: %d requests → %d vectors, topk=%d",
                len(items),
                all_vectors.shape[0],
                merged_topk,
            )

            try:
                loop = asyncio.get_running_loop()
                distances, indices = await loop.run_in_executor(
                    None, self._search_fn, all_vectors, merged_topk
                )
                offset = 0
                for vectors, topk, fut in items:
                    n = vectors.shape[0]
                    if not fut.done():
                        d = distances[offset : offset + n]
                        idx = indices[offset : offset + n]
                        if topk < merged_topk:
                            d = d[:, :topk]
                            idx = idx[:, :topk]
                        fut.set_result((d, idx))
                    offset += n
            except Exception as exc:
                for _, _, fut in items:
                    if not fut.done():
                        fut.set_exception(exc)


class EmbeddingSearchService:
    _profile_lock: threading.Lock = threading.Lock()
    _profile_count: int = 0
    _profile_sums: dict[str, float] = {
        "embed": 0.0,
        "faiss": 0.0,
        "total": 0.0,
    }
    _profile_target: int = 10000
    _profile_output_path: Path = Path("embedding_search_profile.json")

    def __init__(self, config: EmbeddingSearchConfig):
        self.config = config
        self.batcher = DynamicBatchingEmbedder(config)
        self._load_corpus(config.corpus_path)
        self._load_index(config.index_path)
        self.faiss_batcher = DynamicBatchingFAISSSearcher(
            search_fn=self._search_all_indices,
            max_wait_ms=config.faiss_batch_max_wait_ms,
            max_vectors=config.faiss_batch_max_vectors,
        )

    def _load_corpus(self, corpus_path: str) -> None:
        if not os.path.isfile(corpus_path):
            raise FileNotFoundError(f"corpus_path not found: {corpus_path}")
        self.docs = load_corpus(corpus_path)

    def _resolve_index_paths(self, index_path: str) -> list[str]:
        path = Path(index_path)
        if path.is_file() and path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            shards = payload.get("shards") if isinstance(payload, dict) else None
            if not isinstance(shards, list) or not shards:
                raise ValueError(f"Invalid shard manifest format: {index_path}")
            resolved_paths: list[str] = []
            for shard in shards:
                if not isinstance(shard, dict) or "path" not in shard:
                    raise ValueError(f"Invalid shard entry in manifest: {shard}")
                shard_path = Path(str(shard["path"]))
                if not shard_path.is_absolute():
                    shard_path = (path.parent / shard_path).resolve()
                resolved_paths.append(str(shard_path))
            return resolved_paths

        if path.is_file():
            return [str(path)]

        manifest_path = Path(f"{index_path}.manifest.json")
        if manifest_path.is_file():
            return self._resolve_index_paths(str(manifest_path))
        raise FileNotFoundError(f"index_path not found: {index_path}")

    def _to_runtime_index(self, cpu_index, faiss, effective_gpu_id: int):
        gpu_resources = faiss.StandardGpuResources()
        clone_opts = faiss.GpuClonerOptions()
        clone_opts.useFloat16 = self.config.faiss_use_float16
        try:
            gpu_index = faiss.index_cpu_to_gpu(
                gpu_resources,
                effective_gpu_id,
                cpu_index,
                clone_opts,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() or "cudaMalloc" in str(exc):
                logger.warning(
                    "GPU %d OOM when loading index shard (ntotal=%d, dim=%d, float16=%s): %s. "
                    "Falling back to CPU for this shard.",
                    effective_gpu_id,
                    int(cpu_index.ntotal),
                    int(cpu_index.d),
                    self.config.faiss_use_float16,
                    exc,
                )
                return cpu_index, None
            raise
        return gpu_index, gpu_resources

    def _create_index_shards(self, faiss, indices: list[Any]):
        if not indices:
            raise ValueError("Cannot create IndexShards from empty indices.")
        try:
            # successive_ids=False keeps global ids from IndexIDMap2 shards unchanged.
            shard_index = faiss.IndexShards(self.index_dim, True, False)
        except TypeError:
            shard_index = faiss.IndexShards(self.index_dim)
            if hasattr(shard_index, "successive_ids"):
                shard_index.successive_ids = False
        for index in indices:
            shard_index.add_shard(index)
        return shard_index

    def _load_index(self, index_path: str) -> None:
        try:
            import faiss  # type: ignore
        except Exception as exc:
            raise ImportError("faiss is required for embedding search service.") from exc
        resolved_paths = self._resolve_index_paths(index_path)
        cpu_indices = []
        self._gpu_resources: list[Any] = []
        self.index_dim = -1
        for shard_path in resolved_paths:
            if not os.path.isfile(shard_path):
                raise FileNotFoundError(f"index shard not found: {shard_path}")
            cpu_index = faiss.read_index(shard_path)
            shard_dim = int(cpu_index.d)
            if self.index_dim < 0:
                self.index_dim = shard_dim
            elif shard_dim != self.index_dim:
                raise ValueError(
                    f"Index dim mismatch across shards: got={shard_dim}, expected={self.index_dim}, shard={shard_path}"
                )
            cpu_indices.append(cpu_index)

        if not cpu_indices:
            raise ValueError(f"No index loaded from: {index_path}")

        total_ntotal = sum(int(index.ntotal) for index in cpu_indices)
        if not self.config.faiss_use_gpu:
            self.search_index = (
                cpu_indices[0] if len(cpu_indices) == 1 else self._create_index_shards(faiss, cpu_indices)
            )
            logger.info(
                "Loaded %d FAISS shards on CPU (total_ntotal=%d, dim=%d)",
                len(cpu_indices),
                total_ntotal,
                self.index_dim,
            )
            return

        gpu_count = int(faiss.get_num_gpus())
        if gpu_count <= 0:
            raise RuntimeError("FAISS-GPU is enabled but no GPU is visible to faiss.")

        if self.config.faiss_use_all_gpus and len(cpu_indices) == 1:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = self.config.faiss_use_float16
            co.shard = self.config.faiss_all_gpus_shard
            self.search_index = faiss.index_cpu_to_all_gpus(cpu_indices[0], co=co)
            logger.info(
                "Loaded FAISS index on all %d GPUs (shard=%s, total_ntotal=%d, dim=%d, float16=%s)",
                gpu_count,
                self.config.faiss_all_gpus_shard,
                total_ntotal,
                self.index_dim,
                self.config.faiss_use_float16,
            )
            return

        if self.config.faiss_use_all_gpus and len(cpu_indices) > 1:
            logger.info(
                "Using per-shard GPU placement across %d GPUs for %d shards.",
                gpu_count,
                len(cpu_indices),
            )
            runtime_indices = []
            gpu_loaded = 0
            cpu_fallback = 0
            for shard_idx, cpu_index in enumerate(cpu_indices):
                local_gpu_id = shard_idx % gpu_count
                runtime_index, gpu_res = self._to_runtime_index(cpu_index, faiss, local_gpu_id)
                runtime_indices.append(runtime_index)
                if gpu_res is not None:
                    self._gpu_resources.append(gpu_res)
                    gpu_loaded += 1
                else:
                    cpu_fallback += 1
            self.search_index = self._create_index_shards(faiss, runtime_indices)
            logger.info(
                "Loaded %d FAISS shards on multi-GPU IndexShards: %d on GPU, %d on CPU "
                "(total_ntotal=%d, dim=%d, float16=%s)",
                len(cpu_indices),
                gpu_loaded,
                cpu_fallback,
                total_ntotal,
                self.index_dim,
                self.config.faiss_use_float16,
            )
            return

        effective_gpu_id = self.config.faiss_gpu_id
        if effective_gpu_id < 0:
            raise ValueError(
                f"faiss_gpu_id={self.config.faiss_gpu_id} out of range, visible_gpus={gpu_count}"
            )
        if effective_gpu_id >= gpu_count:
            # If process is pinned via CUDA_VISIBLE_DEVICES=7, faiss sees a single local GPU id 0.
            if gpu_count == 1 and self.config.faiss_gpu_id == 7 and os.environ.get("CUDA_VISIBLE_DEVICES"):
                logger.warning(
                    "faiss_gpu_id=7 but only one visible GPU detected; fallback to local GPU id 0."
                )
                effective_gpu_id = 0
            else:
                raise ValueError(
                    f"faiss_gpu_id={self.config.faiss_gpu_id} out of range, visible_gpus={gpu_count}"
                )
        bytes_per_vector = 2 if self.config.faiss_use_float16 else 4
        est_bytes = sum(int(idx.ntotal) for idx in cpu_indices) * self.index_dim * bytes_per_vector
        est_gib = est_bytes / (1024 ** 3)
        logger.info(
            "Estimated GPU memory for all %d shards: %.1f GiB (float16=%s). "
            "Will attempt to load onto GPU %d; shards that don't fit will fall back to CPU.",
            len(cpu_indices),
            est_gib,
            self.config.faiss_use_float16,
            effective_gpu_id,
        )
        runtime_indices = []
        gpu_loaded = 0
        cpu_fallback = 0
        for cpu_index in cpu_indices:
            runtime_index, gpu_res = self._to_runtime_index(cpu_index, faiss, effective_gpu_id)
            runtime_indices.append(runtime_index)
            if gpu_res is not None:
                self._gpu_resources.append(gpu_res)
                gpu_loaded += 1
            else:
                cpu_fallback += 1
        self.search_index = (
            runtime_indices[0]
            if len(runtime_indices) == 1
            else self._create_index_shards(faiss, runtime_indices)
        )
        logger.info(
            "Loaded %d FAISS shards: %d on GPU %d, %d on CPU (total_ntotal=%d, dim=%d, float16=%s)",
            len(cpu_indices),
            gpu_loaded,
            effective_gpu_id,
            cpu_fallback,
            total_ntotal,
            self.index_dim,
            self.config.faiss_use_float16,
        )

    def _search_all_indices(self, vectors: np.ndarray, topk: int) -> tuple[np.ndarray, np.ndarray]:
        distances, indices = self.search_index.search(vectors, topk)
        return np.asarray(distances, dtype=np.float32), np.asarray(indices, dtype=np.int64)

    def _format_doc(self, doc: dict[str, Any], score: float | None) -> dict[str, Any]:
        title = doc.get("title") or ""
        text = doc.get("text") or ""
        contents = f"{title}\n{text}" if title else text
        result = {
            "document": {
                "id": doc.get("doc_id"),
                "title": title,
                "contents": contents[: self.config.max_text_len],
            }
        }
        if score is not None:
            result["score"] = score
        return result

    async def search(
        self,
        queries: list[str],
        k_list: list[int],
        return_scores: bool,
    ) -> list[list[dict[str, Any]]]:
        if not queries:
            return []
        total_start = time.perf_counter()

        # Embed via the dynamic batcher (non-blocking, merged with other concurrent requests).
        embed_start = total_start
        vectors = await self.batcher.embed(queries)
        embed_end = time.perf_counter()

        # FAISS search via the dynamic batcher (merged with other concurrent requests).
        faiss_start = embed_end
        topk = max(k_list)
        distances, indices = await self.faiss_batcher.search(vectors, topk)
        faiss_end = time.perf_counter()

        results: list[list[dict[str, Any]]] = []
        for row_idx, doc_indices in enumerate(indices):
            row_results: list[dict[str, Any]] = []
            per_query_k = k_list[row_idx]
            for rank, doc_idx in enumerate(doc_indices.tolist()[:per_query_k]):
                if doc_idx < 0 or doc_idx >= len(self.docs):
                    continue
                score = float(distances[row_idx][rank]) if return_scores else None
                row_results.append(self._format_doc(self.docs[doc_idx], score))
            results.append(row_results)

        total_end = time.perf_counter()
        self._record_profile(
            embed_end - embed_start,
            faiss_end - faiss_start,
            total_end - total_start,
        )
        return results

    @staticmethod
    def normalize_query_list(req: SearchRequest) -> list[str]:
        raw_queries = req.query_list if req.query_list is not None else req.queries
        if not isinstance(raw_queries, list):
            return []
        return [q for q in raw_queries if isinstance(q, str) and q.strip()]

    @staticmethod
    def normalize_k_list(req: SearchRequest, query_count: int) -> list[int]:
        if query_count <= 0:
            return []
        k_value: Any = req.k if req.k is not None else req.topk
        if k_value is None:
            return [3] * query_count
        if isinstance(k_value, list):
            if not k_value:
                return [3] * query_count
            if len(k_value) == 1:
                try:
                    k0 = int(k_value[0])
                except Exception as exc:
                    raise ValueError("k contains non-integer value") from exc
                if k0 <= 0:
                    raise ValueError("k must be positive")
                return [k0] * query_count
            if len(k_value) != query_count:
                raise ValueError("k length mismatch with query_list")
            try:
                k_list = [int(v) for v in k_value]
            except Exception as exc:
                raise ValueError("k contains non-integer value") from exc
            if any(v <= 0 for v in k_list):
                raise ValueError("k must be positive")
            return k_list
        try:
            k_int = int(k_value)
        except Exception as exc:
            raise ValueError("k must be an integer or list[int]") from exc
        if k_int <= 0:
            raise ValueError("k must be positive")
        return [k_int] * query_count

    @classmethod
    def _record_profile(cls, embed_s: float, faiss_s: float, total_s: float) -> None:
        with cls._profile_lock:
            cls._profile_count += 1
            cls._profile_sums["embed"] += embed_s
            cls._profile_sums["faiss"] += faiss_s
            cls._profile_sums["total"] += total_s
            if cls._profile_count >= cls._profile_target:
                averages = {
                    key: (value / cls._profile_count if cls._profile_count else 0.0)
                    for key, value in cls._profile_sums.items()
                }
                payload = {
                    "count": cls._profile_count,
                    "average_seconds": averages,
                }
                cls._profile_output_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )


def create_app(service: EmbeddingSearchService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service.batcher.start()
        service.faiss_batcher.start()
        logger.info("DynamicBatchingEmbedder and DynamicBatchingFAISSSearcher started.")
        yield
        await service.faiss_batcher.stop()
        await service.batcher.stop()
        logger.info("DynamicBatchingEmbedder and DynamicBatchingFAISSSearcher stopped.")

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/search", response_model=SearchResponse)
    async def search(req: SearchRequest) -> SearchResponse:
        queries = service.normalize_query_list(req)
        if not queries:
            raise HTTPException(status_code=400, detail="query_list is empty")
        try:
            k_list = service.normalize_k_list(req, len(queries))
            results = await service.search(queries, k_list, req.return_scores)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Search request failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"search failed: {exc}") from exc
        return SearchResponse(result=results)

    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embedding search service.")
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--backend-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--embed-batch-size", type=int, default=256)
    parser.add_argument("--faiss-gpu-id", type=int, default=7)
    parser.add_argument(
        "--faiss-use-cpu",
        action="store_true",
        help="Load and query FAISS index on CPU instead of GPU.",
    )
    parser.add_argument(
        "--faiss-use-float16",
        action="store_true",
        help="Use float16 FAISS GPU index clone options.",
    )
    parser.add_argument(
        "--faiss-use-all-gpus",
        action="store_true",
        help="Use all visible GPUs via faiss.index_cpu_to_all_gpus when possible.",
    )
    parser.add_argument(
        "--faiss-no-all-gpus-shard",
        action="store_true",
        help="Disable sharding when --faiss-use-all-gpus is enabled (replicate index on each GPU).",
    )
    parser.add_argument(
        "--batch-max-wait-ms",
        type=float,
        default=20.0,
        help="Max milliseconds to wait collecting queries before flushing to vLLM (default: 20).",
    )
    parser.add_argument(
        "--batch-max-queries",
        type=int,
        default=4096,
        help="Hard cap on total queries per flush batch (default: 4096).",
    )
    parser.add_argument(
        "--faiss-batch-max-wait-ms",
        type=float,
        default=5.0,
        help="Max milliseconds to wait collecting FAISS search vectors before flushing (default: 5).",
    )
    parser.add_argument(
        "--faiss-batch-max-vectors",
        type=int,
        default=4096,
        help="Hard cap on total vectors per FAISS flush batch (default: 4096).",
    )
    parser.add_argument(
        "--vllm-max-concurrent-batches",
        type=int,
        default=4,
        help="Max sub-batches in flight to vLLM simultaneously (default: 4, match number of backends).",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=4096,
        help="Embedding dimension passed to vLLM (default: 4096). Set to 0 to use model default.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of worker processes sharing the same port via SO_REUSEPORT (default: 1).",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def _run_worker(config: EmbeddingSearchConfig, host: str, port: int, worker_id: int) -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    service = EmbeddingSearchService(config)
    app = create_app(service)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((host, port))
    sock.set_inheritable(True)

    logger.info("Worker %d starting on %s:%d", worker_id, host, port)
    uv_config = uvicorn.Config(app, log_level="info")
    server = uvicorn.Server(uv_config)
    asyncio.run(server.serve(sockets=[sock]))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    config = EmbeddingSearchConfig(
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        backend_url=args.backend_url,
        model=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
        embed_batch_size=args.embed_batch_size,
        faiss_gpu_id=args.faiss_gpu_id,
        faiss_use_gpu=not args.faiss_use_cpu,
        faiss_use_float16=args.faiss_use_float16,
        faiss_use_all_gpus=args.faiss_use_all_gpus,
        faiss_all_gpus_shard=not args.faiss_no_all_gpus_shard,
        batch_max_wait_ms=args.batch_max_wait_ms,
        batch_max_queries=args.batch_max_queries,
        faiss_batch_max_wait_ms=args.faiss_batch_max_wait_ms,
        faiss_batch_max_vectors=args.faiss_batch_max_vectors,
        vllm_max_concurrent_batches=args.vllm_max_concurrent_batches,
        dimensions=args.dimensions if args.dimensions > 0 else None,
    )

    if args.parallel > 1:
        # Divide vllm_max_concurrent_batches across workers so total vLLM concurrency stays the same.
        per_worker_batches = max(1, args.vllm_max_concurrent_batches // args.parallel)
        config.vllm_max_concurrent_batches = per_worker_batches
        logger.info(
            "Starting %d workers on %s:%d with SO_REUSEPORT "
            "(vllm_max_concurrent_batches per worker: %d)",
            args.parallel, args.host, args.port, per_worker_batches,
        )
        processes = []
        for i in range(args.parallel - 1):
            p = multiprocessing.Process(
                target=_run_worker,
                args=(config, args.host, args.port, i),
                daemon=True,
            )
            p.start()
            processes.append(p)
        _run_worker(config, args.host, args.port, args.parallel - 1)
        for p in processes:
            p.join()
    else:
        _run_worker(config, args.host, args.port, 0)


if __name__ == "__main__":
    main()
