"""Unify QA datasets into Hotpot-style JSONL and build mixed train/val/test splits.

Output row format is compatible with ``src/data/hotpot_local_dataset.py``:
{
  "id": str,
  "question": str,
  "answer": str,
  "positive_doc_ids": []
}

No corpus/context fields are kept.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def read_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level list in {path}, got {type(data)}")
    return data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use an 8 MiB write buffer to reduce system-call overhead when writing
    # large corpus files.  writelines avoids per-row string concatenation.
    with path.open("w", encoding="utf-8", buffering=8 * 1024 * 1024) as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def clear_jsonl_files(dir_path: Path) -> None:
    if not dir_path.exists():
        return
    for fp in dir_path.glob("*.jsonl"):
        fp.unlink()


def normalize_question(q: str) -> str:
    return " ".join(q.strip().lower().split())


_DATASET_NAME_MAP = {
    "2wiki": "2wikimultihopqa",
}

# Maps any alias → the short "output" name used in mixed train/val files.
# Internal normalised splits use "2wikimultihopqa"; mixed files use "2wiki".
# canonical_dataset_name() is used to normalise all *output* rows so that
# train/val JSONL files produced by this script are consistent with the
# existing train_mixed_9000.jsonl / val_mixed_900.jsonl format.
_OUTPUT_DATASET_MAP: dict[str, str] = {
    "2wikimultihopqa": "2wiki",
}


def canonical_dataset_name(name: str) -> str:
    """Normalise a dataset name to the short form used in output files."""
    return _OUTPUT_DATASET_MAP.get(name, name)


def to_hotpot_row(
    *,
    dataset: str,
    split: str,
    row_id: str,
    question: str,
    answer: str,
) -> dict[str, Any]:
    return {
        "id": row_id,
        "question": question.strip(),
        "answer": answer.strip(),
        "positive_doc_ids": [],
        "dataset": _DATASET_NAME_MAP.get(dataset, dataset),
        "split": split,
    }


def convert_hotpot_like_json(path: Path, dataset: str, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    data = read_json(path)
    for i, ex in enumerate(data):
        question = str(ex.get("question", "")).strip()
        if not question:
            continue
        answer = str(ex.get("answer", "")).strip()
        ex_id = ex.get("_id", ex.get("id", f"{dataset}-{split}-{i}"))
        rows.append(
            to_hotpot_row(
                dataset=dataset,
                split=split,
                row_id=str(ex_id),
                question=question,
                answer=answer,
            )
        )
    return rows


def convert_musique_jsonl(path: Path, dataset: str, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    data = read_jsonl(path)
    for i, ex in enumerate(data):
        question = str(ex.get("question", "")).strip()
        if not question:
            continue
        answer = str(ex.get("answer", "")).strip()
        ex_id = ex.get("id", f"{dataset}-{split}-{i}")
        rows.append(
            to_hotpot_row(
                dataset=dataset,
                split=split,
                row_id=str(ex_id),
                question=question,
                answer=answer,
            )
        )
    return rows


def parse_possible_answers(raw: str) -> list[str]:
    text = raw.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except json.JSONDecodeError:
        pass
    return [text]


def convert_popqa_tsv(path: Path, dataset: str, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, ex in enumerate(reader):
            question = str(ex.get("question", "")).strip()
            if not question:
                continue
            answers = parse_possible_answers(str(ex.get("possible_answers", "")).strip())
            answer = answers[0] if answers else ""
            ex_id = ex.get("id", f"{dataset}-{split}-{i}")
            rows.append(
                to_hotpot_row(
                    dataset=dataset,
                    split=split,
                    row_id=str(ex_id),
                    question=question,
                    answer=answer,
                )
            )
    return rows


def convert_bamboogle_parquet(path: Path, dataset: str, split: str) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "Reading bamboogle parquet requires pandas + pyarrow. "
            "Please run this script in an environment that has them installed."
        ) from e

    df = pd.read_parquet(path)
    rows: list[dict[str, Any]] = []
    for i, ex in enumerate(df.to_dict(orient="records")):
        question = str(ex.get("Question", "")).strip()
        if not question:
            continue
        answer = str(ex.get("Answer", "")).strip()
        ex_id = f"{dataset}-{split}-{i}"
        rows.append(
            to_hotpot_row(
                dataset=dataset,
                split=split,
                row_id=ex_id,
                question=question,
                answer=answer,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Wiki corpus helpers
# ---------------------------------------------------------------------------

def _corpus_key(title: str, text: str) -> str:
    """Canonical dedup key: MD5 hash of the normalised (title, text) pair.

    Using a fixed-size hash instead of the raw concatenated strings keeps
    the ``seen_keys`` set memory-efficient even for million-entry corpora.
    """
    raw = "\x00".join([title.strip().lower(), text.strip().lower()])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def load_and_dedup_corpus(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """Load wiki-18.corpus.jsonl and remove duplicate entries by (title, text).

    Returns:
        (deduped_docs, seen_keys) where seen_keys contains the canonical key for
        every retained document so that callers can cheaply check membership.
    """
    seen_keys: set[str] = set()
    deduped: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            title = str(row.get("title") or "").strip()
            text = str(row.get("text") or "").strip()
            key = _corpus_key(title, text)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(row)

    return deduped, seen_keys


def extract_supporting_docs_hotpot_like(path: Path) -> list[dict[str, Any]]:
    """Extract supporting paragraphs from a HotpotQA-style or 2WikiMultiHop JSON file.

    Only paragraphs whose title appears in ``supporting_facts`` are returned.
    Each returned doc has keys ``title`` and ``text``.
    """
    data = read_json(path)
    docs: list[dict[str, Any]] = []
    for ex in data:
        support_titles: set[str] = {
            str(title) for title, _ in ex.get("supporting_facts", [])
        }
        for ctx_title, sentences in ex.get("context", []):
            if str(ctx_title) not in support_titles:
                continue
            text = " ".join(str(s) for s in sentences)
            docs.append({"title": str(ctx_title), "text": text})
    return docs


def extract_supporting_docs_musique(path: Path) -> list[dict[str, Any]]:
    """Extract supporting paragraphs from a Musique JSONL file.

    Only paragraphs with ``is_supporting == True`` are returned.
    Each returned doc has keys ``title`` and ``text``.
    """
    data = read_jsonl(path)
    docs: list[dict[str, Any]] = []
    for ex in data:
        for para in ex.get("paragraphs", []):
            if not para.get("is_supporting", False):
                continue
            title = str(para.get("title") or "")
            text = str(para.get("paragraph_text") or "")
            docs.append({"title": title, "text": text})
    return docs


def supplement_corpus(
    base_docs: list[dict[str, Any]],
    seen_keys: set[str],
    new_docs: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Merge *new_docs* into *base_docs*, skipping entries already in *seen_keys*.

    New documents receive integer doc_ids continuing from the maximum existing id.
    *new_docs* may be any iterable (including generators) to avoid holding a
    second copy of all supporting paragraphs in memory simultaneously.

    Returns (merged_docs, n_added).  *base_docs* is mutated in-place and also
    returned as the merged result, so the caller should not rely on *base_docs*
    being unchanged after this call.
    """
    # Determine the starting doc_id for new entries.
    # Documents loaded by load_and_dedup_corpus are appended in file order so
    # the last entry almost certainly has the highest id; scan from the tail to
    # avoid iterating the entire list in the common case.
    max_id = -1
    for doc in reversed(base_docs):
        raw_id = doc.get("doc_id")
        try:
            max_id = int(raw_id)
            break  # first valid id from the tail is the maximum
        except (TypeError, ValueError):
            continue
    next_id = max_id + 1

    added = 0
    # Append directly to base_docs to avoid an O(N) full-list copy.
    for doc in new_docs:
        title = str(doc.get("title") or "").strip()
        text = str(doc.get("text") or "").strip()
        key = _corpus_key(title, text)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        base_docs.append({"doc_id": str(next_id), "title": title, "text": text})
        next_id += 1
        added += 1

    return base_docs, added


# ---------------------------------------------------------------------------

def sample_rows(rows: list[dict[str, Any]], k: int, rng: random.Random) -> list[dict[str, Any]]:
    if k >= len(rows):
        return list(rows)
    return rng.sample(rows, k)


def dedupe_by_question(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], set[str]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = normalize_question(str(row.get("question", "")))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out, seen


def round_robin_backfill(
    pools: dict[str, list[dict[str, Any]]],
    seen_questions: set[str],
    target_size: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    ds_names = list(pools.keys())
    rng.shuffle(ds_names)

    shuffled_pools: dict[str, list[dict[str, Any]]] = {}
    for ds in ds_names:
        data = list(pools[ds])
        rng.shuffle(data)
        shuffled_pools[ds] = data

    pointers = {ds: 0 for ds in ds_names}
    additions: list[dict[str, Any]] = []

    while len(additions) < target_size:
        progressed = False
        for ds in ds_names:
            data = shuffled_pools[ds]
            idx = pointers[ds]
            while idx < len(data):
                row = data[idx]
                idx += 1
                key = normalize_question(str(row.get("question", "")))
                if not key or key in seen_questions:
                    continue
                seen_questions.add(key)
                additions.append(row)
                progressed = True
                break
            pointers[ds] = idx
            if len(additions) >= target_size:
                break
        if not progressed:
            break
    return additions


def build_hierarchical_train_splits(
    train_pools: dict[str, list[dict[str, Any]]],
    val_questions: set[str],
    tier_per_dataset: list[int],
    seed: int,
) -> list[tuple[int, list[dict[str, Any]]]]:
    """Sample train splits in strictly nested tiers so smaller tiers are strict subsets.

    For each dataset:
      1. Filter rows whose question (normalised) matches any validation question.
      2. Deduplicate by normalised question within the dataset.
      3. Shuffle deterministically with a dataset-specific seed.
      4. Slice the first ``tier_size`` rows for each tier.

    Because each smaller tier slice is a prefix of the larger tier slice for the
    same dataset, the resulting combined splits satisfy:
        train_tier_0  ⊆  train_tier_1  ⊆  ...  (by row id).

    Cross-dataset deduplication is intentionally skipped to preserve the
    containment guarantee.  The three source datasets (2wiki, hotpotqa, musique)
    have negligible question overlap in practice.

    Args:
        train_pools: Mapping from dataset name to list of normalised rows.
        val_questions: Set of normalised validation question strings to exclude.
        tier_per_dataset: Per-dataset sample counts for each tier (e.g. [3000, 5000, 15000]).
            Will be sorted ascending internally.
        seed: Base random seed; each dataset and tier uses a derived sub-seed
            so results are independent of call order.

    Returns:
        List of ``(total_size, rows)`` tuples in ascending tier order.
    """
    sorted_tiers = sorted(tier_per_dataset)
    max_per_dataset = sorted_tiers[-1]
    ds_order = sorted(train_pools.keys())  # deterministic ordering

    per_ds_samples: dict[str, list[dict[str, Any]]] = {}
    for ds_name in ds_order:
        pool = train_pools[ds_name]

        # 1. Filter val questions to prevent leakage.
        filtered = [
            r for r in pool
            if normalize_question(str(r.get("question", ""))) not in val_questions
        ]

        # 2. Deduplicate by normalised question within this dataset.
        filtered, _ = dedupe_by_question(filtered)

        # 3. Deterministic per-dataset shuffle, stable across global rng state.
        ds_seed = (seed + abs(hash(ds_name))) % (2 ** 31)
        ds_rng = random.Random(ds_seed)
        ds_rng.shuffle(filtered)

        available = len(filtered)
        if available < max_per_dataset:
            print(
                f"  [WARNING] {ds_name}: only {available} rows available after "
                f"val-filtering and dedup (largest tier requests {max_per_dataset}). "
                f"Largest tier will use all {available} rows from this dataset."
            )

        # Normalise dataset field in pool rows to match the output format.
        out_name = canonical_dataset_name(ds_name)
        per_ds_samples[ds_name] = [
            {**r, "dataset": out_name} for r in filtered
        ]

    results: list[tuple[int, list[dict[str, Any]]]] = []
    for tier_size in sorted_tiers:
        tier_rows: list[dict[str, Any]] = []
        for ds_name in ds_order:
            tier_rows.extend(per_ds_samples[ds_name][:tier_size])

        # Shuffle combined rows with a deterministic tier-specific seed.
        tier_seed = (seed + tier_size * len(ds_order)) % (2 ** 31)
        tier_rng = random.Random(tier_seed)
        tier_rng.shuffle(tier_rows)

        results.append((len(tier_rows), tier_rows))

    return results


def build_hierarchical_train_splits_from_base(
    base_rows: list[dict[str, Any]],
    train_pools: dict[str, list[dict[str, Any]]],
    val_questions: set[str],
    tier_per_dataset: list[int],
    seed: int,
) -> list[tuple[int, list[dict[str, Any]]]]:
    """Expand an existing base train set into larger nested tiers.

    The base rows are preserved unchanged and included in every output tier.
    Additional rows are sampled from the training pools to reach each tier's
    per-dataset target, filtering out rows already in the base and any val
    questions.

    Args:
        base_rows: Existing training rows (the "9k" file).  All of these
            appear in every output tier.
        train_pools: Full normalised training pools keyed by dataset name.
        val_questions: Normalised validation question strings to exclude.
        tier_per_dataset: Per-dataset *total* target counts for each tier
            (e.g. [5000, 15000]).  Must be >= the per-dataset count already
            present in base_rows, or a warning is printed.
        seed: Deterministic seed for sampling additional rows.

    Returns:
        List of ``(total_size, rows)`` tuples in ascending tier order.
        Each tier is a strict superset of base_rows (by row id).
    """
    sorted_tiers = sorted(tier_per_dataset)
    max_per_dataset = sorted_tiers[-1]
    ds_order = sorted(train_pools.keys())

    # Split base rows by dataset for per-dataset accounting.
    # Base rows keep their original dataset field (e.g. "2wiki").
    base_by_ds: dict[str, list[dict[str, Any]]] = {}
    for r in base_rows:
        ds = canonical_dataset_name(str(r.get("dataset", "")))
        base_by_ds.setdefault(ds, []).append(r)

    base_ids: set[str] = {str(r["id"]) for r in base_rows}
    base_questions: set[str] = {
        normalize_question(str(r.get("question", ""))) for r in base_rows
    }

    # Per-dataset pools of *additional* rows (not in base, not in val).
    per_ds_additions: dict[str, list[dict[str, Any]]] = {}
    for ds_name in ds_order:
        pool = train_pools[ds_name]
        filtered = [
            r for r in pool
            if str(r["id"]) not in base_ids
            and normalize_question(str(r.get("question", ""))) not in val_questions
            and normalize_question(str(r.get("question", ""))) not in base_questions
        ]
        filtered, _ = dedupe_by_question(filtered)

        ds_seed = (seed + abs(hash(ds_name))) % (2 ** 31)
        ds_rng = random.Random(ds_seed)
        ds_rng.shuffle(filtered)

        out_name = canonical_dataset_name(ds_name)
        base_count = len(base_by_ds.get(out_name, []))
        needed_max = max_per_dataset - base_count
        if needed_max < 0:
            print(
                f"  [WARNING] {ds_name}: base already has {base_count} rows, "
                f"which exceeds the largest tier target {max_per_dataset}. "
                f"No additional rows will be sampled for this dataset."
            )
        if len(filtered) < max(0, needed_max):
            print(
                f"  [WARNING] {ds_name}: only {len(filtered)} additional rows available "
                f"after filtering (need up to {needed_max} more for largest tier)."
            )
        # Normalise dataset field in pool rows to match the output format.
        per_ds_additions[out_name] = [
            {**r, "dataset": out_name} for r in filtered
        ]

    results: list[tuple[int, list[dict[str, Any]]]] = []
    for tier_size in sorted_tiers:
        tier_rows: list[dict[str, Any]] = list(base_rows)  # always include base
        for out_name in sorted(per_ds_additions.keys()):
            base_count = len(base_by_ds.get(out_name, []))
            needed = max(0, tier_size - base_count)
            tier_rows.extend(per_ds_additions[out_name][:needed])

        tier_seed = (seed + tier_size * len(ds_order)) % (2 ** 31)
        tier_rng = random.Random(tier_seed)
        tier_rng.shuffle(tier_rows)

        results.append((len(tier_rows), tier_rows))

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("./data"),
        help="Root dir that contains 2wiki/hotpotqa/musique/popqa/bamboogle",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./data/unified_hotpot_format"),
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--train_target_size", type=int, default=9000)
    parser.add_argument("--val_per_dataset", type=int, default=300)
    parser.add_argument(
        "--train_initial_per_dataset",
        type=int,
        default=3500,
        help="Initial sample size from each train split before dedup",
    )
    parser.add_argument(
        "--hierarchical_tiers",
        type=str,
        default=None,
        help=(
            "Comma-separated per-dataset sample counts for generating nested train splits. "
            "E.g. '3000,5000,15000' produces train_mixed_9000.jsonl, "
            "train_mixed_15000.jsonl, and train_mixed_45000.jsonl where each "
            "larger file is a strict superset of all smaller files. "
            "Val questions are excluded from all tiers. "
            "When set, the legacy single-file train generation is skipped."
        ),
    )
    parser.add_argument(
        "--base_train_jsonl",
        type=Path,
        default=None,
        help=(
            "Path to an existing train JSONL file (e.g. train_mixed_9000.jsonl) to use "
            "as the fixed base when expanding with --hierarchical_tiers. "
            "Every row in the base is guaranteed to appear in all generated tiers. "
            "The base file itself is never modified. "
            "Only valid together with --hierarchical_tiers; ignored otherwise."
        ),
    )
    parser.add_argument(
        "--musique_variant",
        choices=["ans", "full"],
        default="full",
        help="Use musique_ans_* or musique_full_* files",
    )
    parser.add_argument(
        "--max_test_size",
        type=int,
        default=1000,
        help="Cap each test split to at most this many rows; set <=0 to disable capping",
    )
    parser.add_argument(
        "--wiki_corpus",
        type=Path,
        default=None,
        help=(
            "Path to wiki-18.corpus.jsonl. When provided, duplicate entries are removed "
            "and supporting paragraphs from all datasets are merged in. "
            "The result is written to --wiki_out."
        ),
    )
    parser.add_argument(
        "--wiki_out",
        type=Path,
        default=None,
        help=(
            "Output path for the supplemented corpus JSONL. "
            "Defaults to <wiki_corpus_dir>/wiki-supplemented.jsonl."
        ),
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data_root = args.data_root
    output_dir = args.output_dir
    normalized_dir = output_dir / "normalized"

    # Paths
    p_2wiki_train = data_root / "2wiki" / "train.json"
    p_2wiki_dev = data_root / "2wiki" / "dev.json"
    # p_2wiki_test is intentionally unused: 2wiki test.json has no answers.
    # Test set is built from dev minus validation rows instead.

    p_hotpot_train = data_root / "hotpotqa" / "train.json"
    p_hotpot_dev = data_root / "hotpotqa" / "hotpot_dev_distractor_v1.json"

    p_musique_train = data_root / "musique" / f"musique_{args.musique_variant}_v1.0_train.jsonl"
    p_musique_dev = data_root / "musique" / f"musique_{args.musique_variant}_v1.0_dev.jsonl"

    p_bamboogle_test = data_root / "bamboogle" / "data" / "test-00000-of-00001-fd9def31e0acf72c.parquet"
    p_popqa_test = data_root / "popqa" / "test.tsv"

    # Convert all needed splits to normalized Hotpot-style rows.
    # Note: 2wiki test.json has no answers, so we use "2wikimultihopqa" as the canonical
    # dataset name throughout (matching the searchR1_2wikimultihopqa reward function key).
    normalized: dict[str, dict[str, list[dict[str, Any]]]] = {
        "2wikimultihopqa": {
            "train": convert_hotpot_like_json(p_2wiki_train, "2wiki", "train"),
            "dev": convert_hotpot_like_json(p_2wiki_dev, "2wiki", "dev"),
        },
        "hotpotqa": {
            "train": convert_hotpot_like_json(p_hotpot_train, "hotpotqa", "train"),
            "dev": convert_hotpot_like_json(p_hotpot_dev, "hotpotqa", "dev"),
        },
        "musique": {
            "train": convert_musique_jsonl(p_musique_train, "musique", "train"),
            "dev": convert_musique_jsonl(p_musique_dev, "musique", "dev"),
        },
        "bamboogle": {
            "test": convert_bamboogle_parquet(p_bamboogle_test, "bamboogle", "test"),
        },
        "popqa": {
            "test": convert_popqa_tsv(p_popqa_test, "popqa", "test"),
        },
    }

    # Write normalized per-dataset splits first.
    for ds_name, splits in normalized.items():
        # Use "2wiki" as directory name for backward compatibility
        dir_name = "2wiki" if ds_name == "2wikimultihopqa" else ds_name
        for split_name, rows in splits.items():
            write_jsonl(normalized_dir / dir_name / f"{split_name}.jsonl", rows)

    train_sources = ["2wikimultihopqa", "hotpotqa", "musique"]

    if args.hierarchical_tiers:
        # ---------------------------------------------------------------
        # Hierarchical nested train split generation
        # ---------------------------------------------------------------
        # Val is loaded from an existing file so that experiments sharing the
        # same val_mixed file remain directly comparable.  If no such file
        # exists yet, val is generated fresh (same as legacy mode, using rng).
        tier_per_dataset = sorted(
            int(x.strip()) for x in args.hierarchical_tiers.split(",")
        )
        expected_val_path = output_dir / "mixed" / f"val_mixed_{args.val_per_dataset * len(train_sources)}.jsonl"
        if expected_val_path.exists():
            existing_val = read_jsonl(expected_val_path)
            val_questions: set[str] = {
                normalize_question(str(r.get("question", ""))) for r in existing_val
            }
            val_ids_by_dataset: dict[str, set[str]] = {}
            for r in existing_val:
                ds = canonical_dataset_name(str(r.get("dataset", "")))
                val_ids_by_dataset.setdefault(ds, set()).add(str(r["id"]))
            val_mixed = existing_val
            print(
                f"[hierarchical] Reusing existing val from {expected_val_path}  "
                f"({len(val_mixed)} rows, {len(val_questions)} unique questions)"
            )
        else:
            # Generate val fresh (no existing file).
            val_sources = {
                "2wikimultihopqa": normalized["2wikimultihopqa"]["dev"],
                "hotpotqa": normalized["hotpotqa"]["dev"],
                "musique": normalized["musique"]["dev"],
            }
            val_mixed = []
            val_ids_by_dataset = {}
            for ds_name, rows in val_sources.items():
                sampled = sample_rows(rows, args.val_per_dataset, rng)
                val_mixed.extend(sampled)
                val_ids_by_dataset[ds_name] = {str(x["id"]) for x in sampled}
            rng.shuffle(val_mixed)
            val_questions = {
                normalize_question(str(r.get("question", ""))) for r in val_mixed
            }
            print(
                f"[hierarchical] Val file not found; generated fresh val  "
                f"({len(val_mixed)} rows)"
            )

        # Build test sets (needed for stats even if files already exist).
        tests: dict[str, list[dict[str, Any]]] = {
            "2wikimultihopqa": [
                row for row in normalized["2wikimultihopqa"]["dev"]
                if str(row["id"]) not in val_ids_by_dataset.get(
                    canonical_dataset_name("2wikimultihopqa"), set()
                )
            ],
            "bamboogle": normalized["bamboogle"]["test"],
            "popqa": normalized["popqa"]["test"],
            "hotpotqa": [
                row for row in normalized["hotpotqa"]["dev"]
                if str(row["id"]) not in val_ids_by_dataset.get("hotpotqa", set())
            ],
            "musique": [
                row for row in normalized["musique"]["dev"]
                if str(row["id"]) not in val_ids_by_dataset.get("musique", set())
            ],
        }
        test_counts_before_cap = {k: len(v) for k, v in tests.items()}
        if args.max_test_size > 0:
            tests = {
                ds_name: sample_rows(rows, args.max_test_size, rng)
                for ds_name, rows in tests.items()
            }

        # Build hierarchical train splits.
        train_pools = {ds: normalized[ds]["train"] for ds in train_sources}

        if args.base_train_jsonl is not None:
            # Expand on top of an existing base file (e.g. train_mixed_9000.jsonl).
            if not args.base_train_jsonl.exists():
                raise FileNotFoundError(f"--base_train_jsonl not found: {args.base_train_jsonl}")
            base_rows = read_jsonl(args.base_train_jsonl)
            base_ds_counts = dict(Counter(r.get("dataset", "?") for r in base_rows))
            print(
                f"[hierarchical] Base file: {args.base_train_jsonl}  "
                f"({len(base_rows)} rows | {base_ds_counts})"
            )
            print(
                f"[hierarchical] Expanding to tiers per dataset: {tier_per_dataset}  "
                f"| Total per tier: {[t * len(train_sources) for t in tier_per_dataset]}"
            )
            print(f"[hierarchical] Val questions to exclude: {len(val_questions)}")
            tier_results = build_hierarchical_train_splits_from_base(
                base_rows, train_pools, val_questions, tier_per_dataset, args.seed
            )
        else:
            print(
                f"[hierarchical] Tiers per dataset: {tier_per_dataset}  "
                f"| Total per tier: {[t * len(train_sources) for t in tier_per_dataset]}"
            )
            print(f"[hierarchical] Val questions to exclude: {len(val_questions)}")
            tier_results = build_hierarchical_train_splits(
                train_pools, val_questions, tier_per_dataset, args.seed
            )

        # Write outputs; keep the mixed dir intact (preserve existing val).
        (output_dir / "mixed").mkdir(parents=True, exist_ok=True)
        if not expected_val_path.exists():
            write_jsonl(expected_val_path, val_mixed)
        clear_jsonl_files(output_dir / "test")
        for ds_name, rows in tests.items():
            write_jsonl(output_dir / "test" / f"{ds_name}_test_{len(rows)}.jsonl", rows)
        test_all: list[dict[str, Any]] = []
        for ds_name in sorted(tests.keys()):
            test_all.extend(tests[ds_name])
        write_jsonl(output_dir / "test" / "test_all.jsonl", test_all)

        hierarchical_train_stats: dict[str, Any] = {}
        for total_size, rows in tier_results:
            out_path = output_dir / "mixed" / f"train_mixed_{total_size}.jsonl"
            write_jsonl(out_path, rows)
            ds_counts = dict(Counter(r.get("dataset", "?") for r in rows))
            print(f"[hierarchical] Written {out_path.name}  ({total_size} rows | {ds_counts})")
            hierarchical_train_stats[f"train_mixed_{total_size}"] = {
                "total": total_size,
                "per_dataset": ds_counts,
            }

        # Containment verification.
        id_sets = [frozenset(r["id"] for r in rows) for _, rows in tier_results]
        for i in range(len(id_sets) - 1):
            missing = id_sets[i] - id_sets[i + 1]
            t_small, t_large = tier_results[i][0], tier_results[i + 1][0]
            if missing:
                print(
                    f"  [WARNING] Containment FAIL: {len(missing)} IDs in "
                    f"train_{t_small} are absent from train_{t_large}"
                )
            else:
                print(f"  [OK] Containment verified: train_{t_small} ⊆ train_{t_large}")

        stats = {
            "seed": args.seed,
            "mode": "hierarchical",
            "hierarchical_tiers_per_dataset": tier_per_dataset,
            "val_per_dataset": args.val_per_dataset,
            "musique_variant": args.musique_variant,
            "normalized_counts": {
                ds_name: {split: len(rows) for split, rows in splits.items()}
                for ds_name, splits in normalized.items()
            },
            "mixed_val_count": len(val_mixed),
            "hierarchical_train": hierarchical_train_stats,
            "max_test_size": args.max_test_size,
            "test_counts_before_cap": test_counts_before_cap,
            "test_counts": {k: len(v) for k, v in tests.items()},
            "test_all_count": len(test_all),
            "output_dir": str(output_dir.resolve()),
        }
        (output_dir / "stats_hierarchical.json").parent.mkdir(parents=True, exist_ok=True)
        with (output_dir / "stats_hierarchical.json").open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return

    # -----------------------------------------------------------------------
    # Legacy single train file generation (original behaviour, unchanged)
    # -----------------------------------------------------------------------

    # Build mixed train from 2wikimultihopqa/hotpotqa/musique train.
    initial_sampled: list[dict[str, Any]] = []
    for ds_name in train_sources:
        rows = normalized[ds_name]["train"]
        sampled = sample_rows(rows, args.train_initial_per_dataset, rng)
        initial_sampled.extend(sampled)
    rng.shuffle(initial_sampled)

    train_mixed, seen_questions = dedupe_by_question(initial_sampled)

    if len(train_mixed) < args.train_target_size:
        missing = args.train_target_size - len(train_mixed)
        pools = {ds: normalized[ds]["train"] for ds in train_sources}
        additions = round_robin_backfill(pools, seen_questions, missing, rng)
        train_mixed.extend(additions)

    if len(train_mixed) > args.train_target_size:
        rng.shuffle(train_mixed)
        train_mixed = train_mixed[: args.train_target_size]

    rng.shuffle(train_mixed)

    # Build mixed validation (300 from each dev split if possible).
    val_sources = {
        "2wikimultihopqa": normalized["2wikimultihopqa"]["dev"],
        "hotpotqa": normalized["hotpotqa"]["dev"],
        "musique": normalized["musique"]["dev"],
    }
    val_mixed: list[dict[str, Any]] = []
    val_ids_by_dataset: dict[str, set[str]] = {}
    for ds_name, rows in val_sources.items():
        sampled = sample_rows(rows, args.val_per_dataset, rng)
        val_mixed.extend(sampled)
        val_ids_by_dataset[ds_name] = {str(x["id"]) for x in sampled}
    rng.shuffle(val_mixed)

    # Build test sets for all 5 datasets.
    # 2wikimultihopqa: official test.json has no answers, so use dev minus validation rows.
    # hotpotqa/musique: same approach (no public test answers available).
    tests: dict[str, list[dict[str, Any]]] = {
        "2wikimultihopqa": [
            row for row in normalized["2wikimultihopqa"]["dev"] if str(row["id"]) not in val_ids_by_dataset["2wikimultihopqa"]
        ],
        "bamboogle": normalized["bamboogle"]["test"],
        "popqa": normalized["popqa"]["test"],
        "hotpotqa": [
            row for row in normalized["hotpotqa"]["dev"] if str(row["id"]) not in val_ids_by_dataset["hotpotqa"]
        ],
        "musique": [
            row for row in normalized["musique"]["dev"] if str(row["id"]) not in val_ids_by_dataset["musique"]
        ],
    }

    test_counts_before_cap = {k: len(v) for k, v in tests.items()}
    if args.max_test_size > 0:
        tests = {
            ds_name: sample_rows(rows, args.max_test_size, rng)
            for ds_name, rows in tests.items()
        }

    # Write mixed splits and test sets.
    clear_jsonl_files(output_dir / "mixed")
    clear_jsonl_files(output_dir / "test")
    write_jsonl(output_dir / "mixed" / f"train_mixed_{len(train_mixed)}.jsonl", train_mixed)
    write_jsonl(output_dir / "mixed" / f"val_mixed_{len(val_mixed)}.jsonl", val_mixed)
    for ds_name, rows in tests.items():
        write_jsonl(output_dir / "test" / f"{ds_name}_test_{len(rows)}.jsonl", rows)

    # Write combined test_all.jsonl (all datasets in a fixed order for reproducibility).
    test_all: list[dict[str, Any]] = []
    for ds_name in sorted(tests.keys()):
        test_all.extend(tests[ds_name])
    write_jsonl(output_dir / "test" / "test_all.jsonl", test_all)

    stats = {
        "seed": args.seed,
        "train_target_size": args.train_target_size,
        "val_per_dataset": args.val_per_dataset,
        "train_initial_per_dataset": args.train_initial_per_dataset,
        "musique_variant": args.musique_variant,
        "normalized_counts": {
            ds_name: {split: len(rows) for split, rows in splits.items()}
            for ds_name, splits in normalized.items()
        },
        "mixed_train_count": len(train_mixed),
        "mixed_val_count": len(val_mixed),
        "max_test_size": args.max_test_size,
        "test_counts_before_cap": test_counts_before_cap,
        "test_counts": {k: len(v) for k, v in tests.items()},
        "test_all_count": len(test_all),
        "output_dir": str(output_dir.resolve()),
    }
    (output_dir / "stats.json").parent.mkdir(parents=True, exist_ok=True)
    with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))

    # ------------------------------------------------------------------
    # Optional: wiki corpus dedup + supplementation
    # ------------------------------------------------------------------
    if args.wiki_corpus is not None:
        wiki_corpus_path: Path = args.wiki_corpus
        wiki_out_path: Path = (
            args.wiki_out
            if args.wiki_out is not None
            else wiki_corpus_path.parent / "wiki-supplemented.jsonl"
        )

        print(f"\n[wiki] Loading corpus from {wiki_corpus_path} …")
        base_docs, seen_keys = load_and_dedup_corpus(wiki_corpus_path)
        n_after_dedup = len(base_docs)
        print(f"[wiki] Entries after dedup: {n_after_dedup}")

        # Collect supporting paragraphs from all available datasets.
        supporting_sources: list[tuple[str, list[dict[str, Any]]]] = []

        for split_path, label in [
            (p_hotpot_train, "hotpotqa/train"),
            (p_hotpot_dev, "hotpotqa/dev"),
            (p_2wiki_train, "2wiki/train"),
            (p_2wiki_dev, "2wiki/dev"),
        ]:
            if split_path.exists():
                print(f"[wiki] Extracting supporting paragraphs from {label} …")
                docs = extract_supporting_docs_hotpot_like(split_path)
                supporting_sources.append((label, docs))
            else:
                print(f"[wiki] Skipping {label} (file not found: {split_path})")

        for split_path, label in [
            (p_musique_train, "musique/train"),
            (p_musique_dev, "musique/dev"),
        ]:
            if split_path.exists():
                print(f"[wiki] Extracting supporting paragraphs from {label} …")
                docs = extract_supporting_docs_musique(split_path)
                supporting_sources.append((label, docs))
            else:
                print(f"[wiki] Skipping {label} (file not found: {split_path})")

        # Use a generator so supporting_sources is consumed lazily and we
        # never hold a second full copy of all paragraphs in memory.
        all_new_docs = (doc for _, docs in supporting_sources for doc in docs)

        merged_docs, n_added = supplement_corpus(base_docs, seen_keys, all_new_docs)

        print(f"[wiki] New paragraphs added: {n_added}")
        print(f"[wiki] Total corpus size:     {len(merged_docs)}")
        print(f"[wiki] Writing supplemented corpus to {wiki_out_path} …")
        write_jsonl(wiki_out_path, merged_docs)
        print("[wiki] Done.")

        # Append corpus stats to the stats dict and re-write stats.json.
        stats["wiki_corpus"] = {
            "source": str(wiki_corpus_path),
            "output": str(wiki_out_path),
            "n_after_dedup": n_after_dedup,
            "n_added_from_datasets": n_added,
            "n_total": len(merged_docs),
            "supporting_sources": {label: len(docs) for label, docs in supporting_sources},
        }
        with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()


