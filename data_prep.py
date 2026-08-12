"""Flattens the doc-level jsonl files in data/ (one row per document, with a
list of question/answer "samples") into row-per-question HF Datasets ready for
SFTTrainer / GRPOTrainer, and caches the SFT-flattened form under
data/sft_splits/*.jsonl for reuse/inspection.

Run standalone to (re)build the cached splits without touching a GPU:

    python data_prep.py --reasoning-format old
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, Features, Sequence, Value
from datasets import Image as HFImage

from common.paths import DATA_DIR, SPLITS_DIR
from common.prompts import THINKING_PROMPT_TEMPLATE

IMAGE_EXT_FALLBACKS = [".jpg", ".jpeg", ".png"]


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def resolve_image_path(doc: dict, data_dir: Path = DATA_DIR) -> Path | None:
    """doc["image"] is a path relative to data/ (e.g. "images/real/foo.jpg").
    Some sources record a different extension in the jsonl than what's
    actually on disk, so fall back to the other common extensions on the same
    stem before giving up.
    """
    if not doc.get("image"):
        return None
    image_path = Path(doc["image"])
    full_path = image_path if image_path.is_absolute() else data_dir / image_path
    if full_path.exists():
        return full_path
    for ext in IMAGE_EXT_FALLBACKS:
        candidate = full_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# SFT rows: {"images": [...], "prompt": [...], "completion": [...]}
# ---------------------------------------------------------------------------

def build_sft_row(sample: dict, image_path: Path | None, category: str, reasoning_format: str) -> dict:
    question = str(sample.get("prompt", "")).strip()
    answer = str(sample.get("answer", "")).strip()
    prompt_text = THINKING_PROMPT_TEMPLATE.format(question=question)

    use_new = (
        reasoning_format == "new"
        and sample.get("relevance_reasoning")
        and sample.get("derivation_reasoning")
    )

    if use_new:
        relevance = str(sample["relevance_reasoning"]).strip()
        derivation = str(sample["derivation_reasoning"]).strip()
        raw_data = sample.get("raw_data", {})
        raw_str = raw_data if isinstance(raw_data, str) else json.dumps(raw_data, ensure_ascii=False)
        assistant_text = (
            f"<think>{relevance}<raw>{raw_str}</raw>{derivation}</think>"
            f"{json.dumps({'answer': answer}, ensure_ascii=False)}"
        )
    else:
        thinking = str(sample.get("derivation_reasoning", "")).strip()
        assistant_text = f"<think>{thinking}</think>\n{answer}"

    return {
        "images": [str(image_path)] if image_path is not None else [],
        "category": category,
        "prompt": [{"role": "user", "content": prompt_text}],
        "completion": [{"role": "assistant", "content": assistant_text}],
    }


def load_sft_rows(jsonl_file: Path, data_dir: Path, reasoning_format: str) -> list[dict]:
    """Each split jsonl is already a dedicated, non-overlapping split (built
    once and hand-curated) -- no shuffling or doc-level sample-count targeting
    needed here, just flatten docs -> rows."""
    rows = []
    for doc in iter_jsonl(jsonl_file):
        image_path = resolve_image_path(doc, data_dir)
        category = doc.get("category", "")
        for sample in doc.get("samples", []):
            rows.append(build_sft_row(sample, image_path, category, reasoning_format))
    return rows


SFT_ROW_FEATURES = Features(
    {
        "images": Sequence(HFImage()),
        "category": Value("string"),
        "prompt": [{"role": Value("string"), "content": Value("string")}],
        "completion": [{"role": Value("string"), "content": Value("string")}],
    }
)


# ---------------------------------------------------------------------------
# GRPO rows: {"images": [...], "prompt": [...], "solution": ..., "reference_*": ...}
# ---------------------------------------------------------------------------

def build_grpo_row(sample: dict, image_path: Path | None, category: str) -> dict | None:
    """None if the sample lacks the "new"-format fields -- the GRPO reward
    functions score the relevance/raw/derivation structure, so a sample
    without a reference to compare against is skipped rather than silently
    scored on empty strings."""
    if not (sample.get("relevance_reasoning") and sample.get("derivation_reasoning") and sample.get("raw_data")):
        return None

    question = str(sample.get("prompt", "")).strip()
    answer = str(sample.get("answer", "")).strip()
    raw_data = sample.get("raw_data", {})
    raw_str = raw_data if isinstance(raw_data, str) else json.dumps(raw_data, ensure_ascii=False)

    return {
        "images": [str(image_path)] if image_path is not None else [],
        "category": category,
        "prompt": [{"role": "user", "content": THINKING_PROMPT_TEMPLATE.format(question=question)}],
        "solution": answer,
        "reference_raw_data": raw_str,
        "reference_derivation": str(sample["derivation_reasoning"]).strip(),
    }


def load_grpo_rows(jsonl_file: Path, data_dir: Path) -> list[dict]:
    """TRAIN/VAL jsonl are already dedicated, non-overlapping splits (built
    once and hand-curated) -- no shuffling or doc-level sample-count targeting
    needed here, just flatten docs -> rows (dropping samples build_grpo_row()
    rejects)."""
    rows = []
    for doc in iter_jsonl(jsonl_file):
        image_path = resolve_image_path(doc, data_dir)
        category = doc.get("category", "")
        for sample in doc.get("samples", []):
            row = build_grpo_row(sample, image_path, category)
            if row is not None:
                rows.append(row)
    return rows


GRPO_ROW_FEATURES = Features(
    {
        "images": Sequence(HFImage()),
        "category": Value("string"),
        "prompt": [{"role": Value("string"), "content": Value("string")}],
        "solution": Value("string"),
        "reference_raw_data": Value("string"),
        "reference_derivation": Value("string"),
    }
)


def to_dataset(rows: list[dict], features: Features) -> Dataset:
    if not rows:
        empty = {name: [] for name in features}
        return Dataset.from_dict(empty, features=features)
    return Dataset.from_list(rows, features=features)


# ---------------------------------------------------------------------------
# Split builders
# ---------------------------------------------------------------------------

def build_sft_splits(
    train_jsonl: Path,
    val_jsonl: Path,
    data_dir: Path,
    splits_dir: Path,
    reasoning_format: str,
) -> dict[str, Dataset]:
    train_rows = load_sft_rows(train_jsonl, data_dir, reasoning_format)
    val_rows = load_sft_rows(val_jsonl, data_dir, reasoning_format)

    splits_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(splits_dir / "train.jsonl", train_rows)
    write_jsonl(splits_dir / "val.jsonl", val_rows)

    datasets = {
        "train": to_dataset(train_rows, SFT_ROW_FEATURES),
        "val": to_dataset(val_rows, SFT_ROW_FEATURES),
    }
    print(f"Train: {len(datasets['train'])} -> {splits_dir / 'train.jsonl'}")
    print(f"Val:   {len(datasets['val'])} -> {splits_dir / 'val.jsonl'}")
    return datasets


def build_grpo_splits(train_jsonl: Path, val_jsonl: Path, data_dir: Path) -> dict[str, Dataset]:
    train_rows = load_grpo_rows(train_jsonl, data_dir)
    val_rows = load_grpo_rows(val_jsonl, data_dir)

    datasets = {
        "train": to_dataset(train_rows, GRPO_ROW_FEATURES),
        "val": to_dataset(val_rows, GRPO_ROW_FEATURES),
    }
    print(f"Train: {len(datasets['train'])}")
    print(f"Val:   {len(datasets['val'])}")
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=DATA_DIR / "train_sft.jsonl")
    parser.add_argument("--val-jsonl", type=Path, default=DATA_DIR / "val.jsonl")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--reasoning-format", choices=["old", "new"], default="new")
    args = parser.parse_args()

    build_sft_splits(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        data_dir=args.data_dir,
        splits_dir=args.splits_dir,
        reasoning_format=args.reasoning_format,
    )


if __name__ == "__main__":
    main()
