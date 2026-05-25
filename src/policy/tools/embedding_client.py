"""Shared helpers for embedding HTTP backends and corpus loading."""

from __future__ import annotations

import gzip
import io
import json
import tarfile
import threading
import time
from pathlib import Path
from typing import Any

import requests

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_SHARED_SESSION: requests.Session | None = None
_SESSION_LOCK = threading.Lock()


def get_shared_session() -> requests.Session:
    global _SHARED_SESSION
    if _SHARED_SESSION is None:
        with _SESSION_LOCK:
            if _SHARED_SESSION is None:
                _SHARED_SESSION = requests.Session()
    return _SHARED_SESSION


def build_embeddings_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1/embeddings") or base_url.endswith("/embeddings"):
        return base_url
    return f"{base_url}/v1/embeddings"


def _iter_batches(inputs: list[str], batch_size: int) -> list[list[str]]:
    return [inputs[i : i + batch_size] for i in range(0, len(inputs), batch_size)]


def _extract_embeddings(data: Any, expected_size: int) -> list[list[float]]:
    if not isinstance(data, dict):
        raise ValueError("Embedding backend response is not a JSON object.")
    rows = data.get("data")
    if not isinstance(rows, list):
        raise ValueError("Embedding backend response missing 'data' list.")
    ordered_rows: list[tuple[int, dict[str, Any]]] = []
    for pos, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("Embedding backend 'data' item is not an object.")
        idx = row.get("index")
        rank = idx if isinstance(idx, int) and idx >= 0 else pos
        ordered_rows.append((rank, row))
    ordered_rows.sort(key=lambda x: x[0])

    embeddings: list[list[float]] = []
    for _, row in ordered_rows:
        embedding = row.get("embedding")
        if not isinstance(embedding, list):
            raise ValueError("Embedding backend returned invalid embedding payload.")
        embeddings.append(embedding)

    if len(embeddings) != expected_size:
        raise ValueError(
            f"Embedding backend returned unexpected number of embeddings: expected={expected_size}, got={len(embeddings)}"
        )
    return embeddings


def _post_embeddings(
    *,
    session: requests.Session,
    url: str,
    payload: dict[str, Any],
    timeout: int,
    max_retries: int,
    retry_backoff: float,
    logger: Any = None,
) -> Any:
    retries = max(0, int(max_retries))
    for attempt in range(retries + 1):
        # Increase timeout on each retry to handle temporarily slow servers.
        effective_timeout = timeout * (attempt + 1)
        try:
            resp = session.post(url, json=payload, timeout=effective_timeout)
            if resp.ok:
                return resp.json()

            if logger is not None:
                logger.error(
                    "Embedding backend error %s for %s (attempt %d/%d, timeout=%ds): %s",
                    resp.status_code,
                    url,
                    attempt + 1,
                    retries + 1,
                    effective_timeout,
                    resp.text,
                )
            retryable = resp.status_code in _RETRYABLE_STATUS_CODES
            if retryable and attempt < retries:
                sleep_secs = retry_backoff * (2**attempt)
                if logger is not None:
                    logger.warning("Retrying in %.1fs...", sleep_secs)
                time.sleep(sleep_secs)
                continue
            resp.raise_for_status()
        except requests.RequestException as exc:
            if logger is not None:
                logger.warning(
                    "Embedding request exception on attempt %d/%d (timeout=%ds): %s",
                    attempt + 1,
                    retries + 1,
                    effective_timeout,
                    exc,
                )
            if attempt < retries:
                sleep_secs = retry_backoff * (2**attempt)
                if logger is not None:
                    logger.warning("Retrying in %.1fs...", sleep_secs)
                time.sleep(sleep_secs)
                continue
            raise
    raise RuntimeError("Embedding request failed after retries.")


def call_openai_embeddings(
    *,
    base_url: str,
    model: str,
    inputs: list[str],
    timeout: int,
    session: requests.Session | None = None,
    max_retries: int = 0,
    retry_backoff: float = 0.3,
    batch_size: int | None = None,
    logger=None,
) -> list[list[float]]:
    if not inputs:
        return []

    url = build_embeddings_url(base_url)
    request_session = session or get_shared_session()
    effective_batch_size = len(inputs) if not batch_size or batch_size <= 0 else int(batch_size)

    embeddings: list[list[float]] = []
    for batch_inputs in _iter_batches(inputs, effective_batch_size):
        payload = {"model": model, "input": batch_inputs}
        data = _post_embeddings(
            session=request_session,
            url=url,
            payload=payload,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            logger=logger,
        )
        batch_embeddings = _extract_embeddings(data, len(batch_inputs))
        embeddings.extend(batch_embeddings)

    if len(embeddings) != len(inputs):
        raise ValueError("Embedding backend returned unexpected number of embeddings.")
    return embeddings


def _iter_corpus_lines(corpus_path: str):
    path = Path(corpus_path)
    if tarfile.is_tarfile(corpus_path):
        with tarfile.open(corpus_path, "r:*") as tf:
            jsonl_members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".jsonl")]
            if not jsonl_members:
                raise ValueError(f"No .jsonl file found in tar corpus: {corpus_path}")
            member = jsonl_members[0]
            extracted = tf.extractfile(member)
            if extracted is None:
                raise ValueError(f"Failed to extract corpus member from tar: {member.name}")
            with io.TextIOWrapper(extracted, encoding="utf-8") as f:
                for line in f:
                    yield line
        return

    if path.suffix == ".gz":
        with gzip.open(corpus_path, "rt", encoding="utf-8") as f:
            for line in f:
                yield line
        return

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            yield line


def _normalize_corpus_row(row: dict[str, Any]) -> dict[str, Any]:
    doc_id = row.get("doc_id")
    if doc_id is None:
        doc_id = row.get("id")

    title = row.get("title") or ""
    text = row.get("text")
    if text is None:
        contents = row.get("contents")
        if isinstance(contents, str):
            if "\n" in contents:
                parsed_title, parsed_text = contents.split("\n", 1)
            else:
                parsed_title, parsed_text = "", contents
            # Keep explicit title from source rows when provided.
            title = title or parsed_title
            text = parsed_text
        else:
            text = ""

    return {
        "doc_id": doc_id,
        "title": title,
        "text": text,
    }


def load_corpus(corpus_path: str) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for line in _iter_corpus_lines(corpus_path):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Invalid corpus row (expect object): {type(row).__name__}")
        docs.append(_normalize_corpus_row(row))
    return docs


__all__ = ["build_embeddings_url", "call_openai_embeddings", "get_shared_session", "load_corpus"]

