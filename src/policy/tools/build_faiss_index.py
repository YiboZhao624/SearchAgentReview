"""Build and persist a FAISS index on the tool server.

Usage:
    python -m src.tools.build_faiss_index \
        --corpus-path /path/to/corpus.jsonl \
        --backend-url http://xx.xx.xx.xx:8000 \
        --model BAAI/bge-m3 \
        --index-out /path/to/index.faiss
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from tqdm import tqdm
import numpy as np

from src.policy.tools.embedding_client import call_openai_embeddings, load_corpus

logger = logging.getLogger(__name__)


def _doc_to_text(doc: dict[str, Any]) -> str:
    title = doc.get("title") or ""
    text = doc.get("text") or ""
    return f"{title}\n{text}".strip() if title else text


def _iter_batches(items: list[str], batch_size: int) -> list[list[str]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    checkpoint_path: str,
    partial_index_path: str,
    faiss_module: Any,
    index: Any,
    next_add_idx: int,
    total_added: int,
    current_shard_start: int,
    shard_paths: list[str],
    shard_ranges: list[dict],
    batch_size: int = 0,
    logger: Any = None,
) -> None:
    """Atomically persist progress so the run can be resumed after a crash.

    Write order (JSON-first) is intentional:
      1. Write partial FAISS index to ``partial.faiss.tmp``
      2. Write metadata JSON (includes ``partial_index_ntotal``) to ``checkpoint.json.tmp``
      3. ``rename(checkpoint.json.tmp → checkpoint.json)``  ← JSON committed
      4. ``rename(partial.faiss.tmp → partial.faiss)``      ← partial committed

    If the process crashes between steps 3 and 4 the metadata will record a
    ``partial_index_ntotal`` that does not match what is actually on disk.
    ``_load_checkpoint`` detects this and rolls back to the last shard
    boundary (``current_shard_start``) rather than silently producing
    duplicate vector IDs.
    """
    partial_ntotal: int = int(index.ntotal) if (index is not None and index.ntotal > 0) else 0
    partial_tmp = partial_index_path + ".tmp"
    json_tmp = checkpoint_path + ".tmp"

    # --- Step 1: write partial FAISS index to a temp file ---
    try:
        if partial_ntotal > 0:
            faiss_module.write_index(index, partial_tmp)
        else:
            # Clean shard boundary: remove stale partial files so a future
            # resume does not load outdated data.
            Path(partial_tmp).unlink(missing_ok=True)
            Path(partial_index_path).unlink(missing_ok=True)
    except Exception as exc:
        if logger is not None:
            logger.warning("Checkpoint: failed to write partial FAISS index: %s", exc)
        # Without a consistent partial index we cannot write a useful checkpoint.
        return

    # --- Step 2: write metadata JSON to a temp file ---
    meta: dict[str, Any] = {
        "next_add_idx": next_add_idx,
        "total_added": total_added,
        "current_shard_start": current_shard_start,
        # Stored so _load_checkpoint can detect a crash between steps 3 and 4.
        "partial_index_ntotal": partial_ntotal,
        "shard_paths": list(shard_paths),
        "shard_ranges": list(shard_ranges),
        # Stored so resume can detect a batch_size mismatch and so that a
        # shard-boundary rollback can recompute the correct next_add_idx.
        "batch_size": batch_size,
    }
    try:
        Path(json_tmp).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception as exc:
        if logger is not None:
            logger.warning("Checkpoint: failed to write metadata JSON: %s", exc)
        Path(partial_tmp).unlink(missing_ok=True)
        return

    # --- Step 3: commit JSON first (atomic rename) ---
    try:
        Path(json_tmp).replace(checkpoint_path)
    except Exception as exc:
        if logger is not None:
            logger.warning("Checkpoint: failed to commit metadata JSON: %s", exc)
        Path(json_tmp).unlink(missing_ok=True)
        Path(partial_tmp).unlink(missing_ok=True)
        return

    # --- Step 4: commit partial FAISS index (atomic rename) ---
    # A crash here leaves JSON committed but partial.faiss stale.
    # _load_checkpoint will detect the ntotal mismatch and roll back safely.
    if partial_ntotal > 0:
        try:
            Path(partial_tmp).replace(partial_index_path)
        except Exception as exc:
            if logger is not None:
                logger.warning(
                    "Checkpoint: partial FAISS rename failed (%s). "
                    "On resume _load_checkpoint will detect the mismatch and "
                    "roll back to shard boundary (current_shard_start=%d).",
                    exc,
                    current_shard_start,
                )
            # Leave partial_tmp; it will be overwritten on the next checkpoint.

    if logger is not None:
        logger.info(
            "Checkpoint saved at batch %d (total_added=%d, partial_ntotal=%d)",
            next_add_idx,
            total_added,
            partial_ntotal,
        )


def _load_checkpoint(
    checkpoint_path: str,
    partial_index_path: str,
    logger: Any = None,
) -> tuple[dict[str, Any], Any] | None:
    """Load a previously saved checkpoint.

    After loading, validates that the partial FAISS index on disk matches
    the ``partial_index_ntotal`` recorded in the metadata.  A mismatch means
    the process crashed between the JSON rename (step 3) and the partial-index
    rename (step 4) in ``_save_checkpoint``.  In that case the stale partial
    index is discarded and the metadata is rolled back to the last shard
    boundary (``current_shard_start``) so that re-processing starts from a
    safe point without producing duplicate vector IDs.

    Returns ``(meta_dict, index_or_None)`` on success, ``None`` if no
    checkpoint exists or loading fails (triggers a full restart).
    """
    if not Path(checkpoint_path).exists():
        return None
    try:
        import faiss  # type: ignore

        meta: dict[str, Any] = json.loads(
            Path(checkpoint_path).read_text(encoding="utf-8")
        )
        index = None
        if Path(partial_index_path).exists():
            index = faiss.read_index(partial_index_path)

        # --- Consistency check ---
        # Only performed when the checkpoint was written by the new code that
        # records partial_index_ntotal; older checkpoints lack this key and
        # are loaded as-is (best-effort).
        expected_ntotal: int | None = meta.get("partial_index_ntotal")
        if expected_ntotal is not None:
            actual_ntotal = int(index.ntotal) if index is not None else 0
            if actual_ntotal != expected_ntotal:
                if logger is not None:
                    logger.warning(
                        "Checkpoint inconsistency detected: partial index has "
                        "ntotal=%d but metadata expects %d.  The process likely "
                        "crashed between the JSON rename and the FAISS rename in "
                        "_save_checkpoint.  Rolling back to last shard boundary "
                        "(current_shard_start=%d) to avoid duplicate vector IDs.",
                        actual_ntotal,
                        expected_ntotal,
                        meta.get("current_shard_start", 0),
                    )
                shard_start: int = int(meta.get("current_shard_start", 0))
                saved_batch_size: int = int(meta.get("batch_size") or 0)
                if saved_batch_size <= 0:
                    # batch_size unknown: cannot safely recompute next_add_idx;
                    # perform a full restart rather than risk corruption.
                    if logger is not None:
                        logger.warning(
                            "batch_size missing from checkpoint; full restart."
                        )
                    return None
                # Because flushes are always at batch boundaries (see
                # build_faiss_index), shard_start is guaranteed to be an exact
                # multiple of batch_size, so this division is exact.
                shard_start_batch = shard_start // saved_batch_size
                meta = {
                    **meta,
                    "next_add_idx": shard_start_batch,
                    "total_added": shard_start,
                    "partial_index_ntotal": 0,
                }
                index = None
                # Remove the stale partial index so it is not re-loaded on the
                # next resume attempt.
                try:
                    Path(partial_index_path).unlink(missing_ok=True)
                    Path(partial_index_path + ".tmp").unlink(missing_ok=True)
                except Exception:
                    pass

        if logger is not None:
            ntotal = int(index.ntotal) if index is not None else 0
            logger.info(
                "Loaded checkpoint: next_add_idx=%s, total_added=%s, "
                "partial_index.ntotal=%d",
                meta.get("next_add_idx", "?"),
                meta.get("total_added", "?"),
                ntotal,
            )
        return meta, index
    except Exception as exc:
        if logger is not None:
            logger.warning(
                "Could not load checkpoint (%s) — restarting from scratch.", exc
            )
        return None


def build_faiss_index(
    corpus_path: str,
    backend_url: str,
    model: str,
    index_out: str,
    batch_size: int,
    timeout: int,
    max_workers: int,
    max_retries: int,
    max_batch_retries: int = 5,
    normalize: bool = False,
    vectors_per_shard: int = 0,
    checkpoint_every: int = 500,
) -> None:
    try:
        import faiss  # type: ignore
    except Exception as exc:
        raise ImportError("faiss is required to build the index.") from exc

    docs = load_corpus(corpus_path)
    if not docs:
        raise ValueError(f"No documents loaded from corpus: {corpus_path}")
    texts = [_doc_to_text(doc) for doc in docs]

    batches = _iter_batches(texts, batch_size)

    # ------------------------------------------------------------------
    # Checkpoint / resume paths
    # ------------------------------------------------------------------
    checkpoint_path = f"{index_out}.checkpoint.json"
    partial_index_path = f"{index_out}.partial.faiss"

    # Try to resume from a previous run.
    resume = _load_checkpoint(checkpoint_path, partial_index_path, logger=logger)
    if resume is not None:
        ckpt_meta, saved_index = resume
        # Guard: changing batch_size between runs shifts batch boundaries and
        # breaks the text ↔ embedding ID mapping.  Refuse to resume instead of
        # silently producing a corrupted index.
        ckpt_batch_size = ckpt_meta.get("batch_size", 0)
        if ckpt_batch_size and ckpt_batch_size != batch_size:
            raise ValueError(
                f"Cannot resume: checkpoint was built with batch_size={ckpt_batch_size} "
                f"but current batch_size={batch_size}.  "
                f"Delete {checkpoint_path} (and {partial_index_path}) to start over, "
                f"or rerun with --batch-size {ckpt_batch_size}."
            )
        start_batch: int = int(ckpt_meta["next_add_idx"])
        total_added: int = int(ckpt_meta["total_added"])
        current_shard_start: int = int(ckpt_meta["current_shard_start"])
        shard_paths: list[str] = list(ckpt_meta.get("shard_paths", []))
        shard_ranges: list[dict[str, int | str]] = list(ckpt_meta.get("shard_ranges", []))
        index = saved_index
        expected_dim: int | None = int(index.d) if index is not None else None
        logger.info(
            "Resuming from batch %d / %d  (total_added=%d, shards_done=%d)",
            start_batch,
            len(batches),
            total_added,
            len(shard_paths),
        )
    else:
        start_batch = 0
        expected_dim = None
        index = None
        total_added = 0
        shard_paths = []
        shard_ranges = []
        current_shard_start = 0

    # Flag set by _flush_current_index so we immediately checkpoint after a
    # shard is written (prevents re-adding vectors on the next resume).
    _flush_happened = [False]

    def _create_index(dim: int):
        base_index = faiss.IndexFlatIP(dim)
        return faiss.IndexIDMap2(base_index)

    def _flush_current_index() -> None:
        nonlocal index, current_shard_start
        if index is None or index.ntotal == 0:
            return

        if vectors_per_shard > 0:
            shard_id = len(shard_paths)
            shard_path = f"{index_out}.part{shard_id:04d}.faiss"
        else:
            shard_path = index_out

        faiss.write_index(index, shard_path)
        shard_paths.append(shard_path)
        shard_ranges.append(
            {
                "path": shard_path,
                "start_id": current_shard_start,
                "end_id": total_added - 1,
                "ntotal": int(index.ntotal),
            }
        )
        logger.info("Saved index shard to %s (ntotal=%d)", shard_path, index.ntotal)
        current_shard_start = total_added
        index = None
        _flush_happened[0] = True

    def _fetch_batch(batch_items: list[str]) -> np.ndarray:
        emb = call_openai_embeddings(
            base_url=backend_url,
            model=model,
            inputs=batch_items,
            timeout=timeout,
            max_retries=max_retries,
            batch_size=batch_size,
            logger=logger,
        )
        emb_arr = np.asarray(emb, dtype=np.float32)
        if emb_arr.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape={emb_arr.shape}")
        return emb_arr

    _ckpt_every = max(1, checkpoint_every) if checkpoint_every > 0 else 0

    def _maybe_save_checkpoint(next_add_idx: int) -> None:
        """Save checkpoint at regular intervals or right after a shard flush."""
        if _ckpt_every == 0:
            return
        if _flush_happened[0] or next_add_idx % _ckpt_every == 0:
            _save_checkpoint(
                checkpoint_path=checkpoint_path,
                partial_index_path=partial_index_path,
                faiss_module=faiss,
                index=index,
                next_add_idx=next_add_idx,
                total_added=total_added,
                current_shard_start=current_shard_start,
                shard_paths=shard_paths,
                shard_ranges=shard_ranges,
                batch_size=batch_size,
                logger=logger,
            )
            _flush_happened[0] = False

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        in_flight = {}
        pending_results: dict[int, np.ndarray] = {}
        # Track per-batch retry counts to avoid infinite retry loops.
        batch_retry_counts: dict[int, int] = {}
        next_batch_idx = start_batch
        next_add_idx = start_batch

        while next_batch_idx < len(batches) and len(in_flight) < max_workers:
            batch_idx = next_batch_idx
            future = executor.submit(_fetch_batch, batches[batch_idx])
            in_flight[future] = (batch_idx, time.monotonic())
            next_batch_idx += 1

        with tqdm(total=len(batches), initial=start_batch) as pbar:
            while in_flight:
                done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    idx, start_ts = in_flight.pop(future)
                    try:
                        emb_arr = future.result()
                    except Exception as exc:
                        retries_so_far = batch_retry_counts.get(idx, 0)
                        if retries_so_far < max_batch_retries:
                            batch_retry_counts[idx] = retries_so_far + 1
                            wait_secs = 2.0 * (2**retries_so_far)
                            logger.warning(
                                "Batch %d failed (attempt %d/%d): %s — retrying in %.0fs",
                                idx,
                                retries_so_far + 1,
                                max_batch_retries,
                                exc,
                                wait_secs,
                            )
                            time.sleep(wait_secs)
                            retry_future = executor.submit(_fetch_batch, batches[idx])
                            in_flight[retry_future] = (idx, time.monotonic())
                        else:
                            logger.error(
                                "Batch %d permanently failed after %d retries: %s",
                                idx,
                                max_batch_retries,
                                exc,
                            )
                            raise RuntimeError(
                                f"Batch {idx} failed after {max_batch_retries} retries."
                            ) from exc
                        continue
                    elapsed = time.monotonic() - start_ts
                    logger.info("Batch %d completed in %.2fs", idx, elapsed)
                    pending_results[idx] = emb_arr

                    # Keep corpus order: add contiguous completed batches only.
                    while next_add_idx in pending_results:
                        ordered_emb_arr = pending_results.pop(next_add_idx)
                        if expected_dim is None:
                            expected_dim = int(ordered_emb_arr.shape[1])
                            index = _create_index(expected_dim)
                        elif int(ordered_emb_arr.shape[1]) != expected_dim:
                            raise ValueError(
                                f"Embedding dimension mismatch in batch {next_add_idx}: got={ordered_emb_arr.shape[1]}, expected={expected_dim}"
                            )

                        if normalize:
                            norms = np.linalg.norm(ordered_emb_arr, axis=1, keepdims=True)
                            norms = np.clip(norms, a_min=1e-12, a_max=None)
                            ordered_emb_arr = ordered_emb_arr / norms

                        # Add the entire batch at once — never split a batch
                        # across shard boundaries.  This guarantees that after
                        # any flush, current_shard_start is an exact multiple of
                        # batch_size, which is required for _load_checkpoint's
                        # safe rollback arithmetic to hold.
                        batch_total = int(ordered_emb_arr.shape[0])
                        if index is None:
                            if expected_dim is None:
                                raise ValueError("Expected embedding dim is not initialized.")
                            index = _create_index(expected_dim)

                        ids = np.arange(total_added, total_added + batch_total, dtype=np.int64)
                        index.add_with_ids(ordered_emb_arr, ids)
                        total_added += batch_total

                        # Flush at the batch boundary once the shard is full.
                        if vectors_per_shard > 0 and int(index.ntotal) >= vectors_per_shard:
                            _flush_current_index()

                        pbar.update(1)
                        next_add_idx += 1
                        _maybe_save_checkpoint(next_add_idx)

                    if next_batch_idx < len(batches):
                        submit_idx = next_batch_idx
                        submit_future = executor.submit(_fetch_batch, batches[submit_idx])
                        in_flight[submit_future] = (submit_idx, time.monotonic())
                        next_batch_idx += 1

    if total_added == 0 or expected_dim is None:
        raise ValueError("No embeddings were generated from the corpus.")

    _flush_current_index()
    if vectors_per_shard <= 0:
        logger.info("Saved index to %s (ntotal=%d, dim=%d)", index_out, total_added, expected_dim)
    else:
        manifest = {
            "dim": expected_dim,
            "total_vectors": total_added,
            "vectors_per_shard": vectors_per_shard,
            "shards": shard_ranges,
        }
        manifest_path = f"{index_out}.manifest.json"
        Path(manifest_path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved %d shards. Manifest: %s", len(shard_paths), manifest_path)

    # ------------------------------------------------------------------
    # Cleanup checkpoint files on successful completion.
    # ------------------------------------------------------------------
    for _f in (checkpoint_path, partial_index_path):
        try:
            Path(_f).unlink(missing_ok=True)
        except Exception:
            pass
    logger.info("Checkpoint files cleaned up.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index for embedding search.")
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--backend-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--index-out", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument(
        "--max-batch-retries",
        type=int,
        default=5,
        help="Max times to resubmit a single batch on transient errors (e.g. timeout).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize embeddings before building IndexFlatIP (cosine via inner product).",
    )
    parser.add_argument(
        "--vectors-per-shard",
        type=int,
        default=0,
        help="If > 0, split index into multiple shard files with at most this many vectors per shard.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help=(
            "Save a resume checkpoint every N batch additions (0 = disabled). "
            "A checkpoint is also saved immediately after each shard flush. "
            "Checkpoint files are removed on successful completion."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_faiss_index(
        corpus_path=args.corpus_path,
        backend_url=args.backend_url,
        model=args.model,
        index_out=args.index_out,
        batch_size=args.batch_size,
        timeout=args.timeout,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        max_batch_retries=args.max_batch_retries,
        normalize=args.normalize,
        vectors_per_shard=args.vectors_per_shard,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()

