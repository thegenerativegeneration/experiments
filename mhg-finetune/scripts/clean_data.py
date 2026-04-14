#!/usr/bin/env python3
"""
clean_data.py — Quality-filter and deduplicate the raw synthetic dataset.

Filtering steps (applied in order):
  1. Schema validation  — record must have system + messages with user/assistant turns
  2. Length filter      — assistant response must be ≥ 30 characters
  3. MHG heuristics     — for MHG-output scenarios, the assistant response must
                          pass the MHG language detector (skipped for translation/
                          explanation scenarios where the output is Modern German
                          or English)
  4. Modern contamination — flag responses with too many Modern German markers
  5. Back-translation   — if back_translation.enabled, re-score semantic
                          consistency (requires OPENAI_API_KEY)
  6. Deduplication      — MinHash near-duplicate removal on assistant text
  7. Train/eval split   — 90 % train / 10 % eval written to separate files

Usage::

    python scripts/clean_data.py \\
        --input  data/synthetic_raw.jsonl \\
        --train  data/train.jsonl \\
        --eval   data/eval.jsonl \\
        --config configs/generation_config.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

# Local utilities (importable when running from the mhg-finetune root)
sys.path.insert(0, str(Path(__file__).parent))
from utils.mhg_heuristics import has_modern_contamination, is_likely_mhg
from utils.dedup import Deduplicator

# Scenarios whose assistant output is expected to be MHG
MHG_OUTPUT_SCENARIOS = {"mhg_conversation", "paraphrase"}

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _get_assistant_text(record: dict[str, Any]) -> str | None:
    messages = record.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return None


def _validate_schema(record: dict[str, Any]) -> bool:
    if not isinstance(record.get("system"), str) or not record["system"].strip():
        return False
    if "messages" not in record:
        return False
    messages = record["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    roles = [m.get("role") for m in messages]
    return "user" in roles and "assistant" in roles


def _validate_length(record: dict[str, Any], min_chars: int = 30) -> bool:
    text = _get_assistant_text(record)
    return text is not None and len(text.strip()) >= min_chars


def _validate_mhg_output(record: dict[str, Any]) -> bool:
    scenario = record.get("scenario", "")
    if scenario not in MHG_OUTPUT_SCENARIOS:
        return True  # not applicable
    text = _get_assistant_text(record)
    if not text:
        return False
    return is_likely_mhg(text)


def _validate_no_modern_contamination(record: dict[str, Any]) -> bool:
    scenario = record.get("scenario", "")
    if scenario not in MHG_OUTPUT_SCENARIOS:
        return True
    text = _get_assistant_text(record)
    if not text:
        return False
    return not has_modern_contamination(text)


# ---------------------------------------------------------------------------
# Back-translation validation (optional, requires API key)
# ---------------------------------------------------------------------------


def _back_translate(text: str, model: str = "gpt-4o-mini") -> str:
    """Translate *text* (MHG) to Modern German using the OpenAI API."""
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Translate the following Middle High German text into modern "
                    "standard German. Output only the translation, no commentary."
                ),
            },
            {"role": "user", "content": text},
        ],
        max_tokens=512,
        temperature=0.0,
    )
    return response.choices[0].message.content or ""


def _chrf_score(hypothesis: str, reference: str) -> float:
    """Compute chrF score between hypothesis and reference strings."""
    from sacrebleu.metrics import CHRF
    metric = CHRF()
    result = metric.corpus_score([hypothesis], [[reference]])
    return result.score


def _validate_back_translation(
    record: dict[str, Any],
    min_chrf: float,
) -> bool:
    scenario = record.get("scenario", "")
    if scenario not in MHG_OUTPUT_SCENARIOS:
        return True
    assistant_text = _get_assistant_text(record)
    context_text = record.get("context_text", "")
    if not assistant_text or not context_text:
        return True  # can't validate, pass through
    try:
        back_translated = _back_translate(assistant_text)
        score = _chrf_score(back_translated, context_text)
        return score >= min_chrf
    except Exception:  # noqa: BLE001
        return True  # API failure → keep the record


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",  default="data/synthetic_raw.jsonl")
    parser.add_argument("--train",  default="data/train.jsonl")
    parser.add_argument("--eval",   default="data/eval.jsonl")
    parser.add_argument("--config", default="configs/generation_config.yaml")
    parser.add_argument(
        "--eval-fraction", type=float, default=0.1,
        help="Fraction of clean examples to use for evaluation (default: 0.1)",
    )
    parser.add_argument(
        "--dedup-threshold", type=float, default=0.85,
        help="MinHash Jaccard threshold for deduplication (default: 0.85)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    random.seed(args.seed)

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    bt_cfg = config.get("back_translation", {})
    back_translation_enabled: bool = bt_cfg.get("enabled", False)
    min_chrf: float = bt_cfg.get("min_chrf", 20.0)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # ── Load ─────────────────────────────────────────────────────────────────
    records: list[dict[str, Any]] = []
    with input_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} raw records from {input_path}")

    # ── Filter pipeline ──────────────────────────────────────────────────────
    stats: dict[str, int] = {
        "schema": 0,
        "length": 0,
        "mhg_heuristic": 0,
        "modern_contamination": 0,
        "back_translation": 0,
        "duplicate": 0,
    }

    def _apply(records: list, fn, label: str) -> list:
        filtered = [r for r in records if fn(r)]
        dropped = len(records) - len(filtered)
        stats[label] = dropped
        print(f"  {label}: dropped {dropped} records → {len(filtered)} remain")
        return filtered

    records = _apply(records, _validate_schema, "schema")
    records = _apply(records, _validate_length, "length")
    records = _apply(records, _validate_mhg_output, "mhg_heuristic")
    records = _apply(records, _validate_no_modern_contamination, "modern_contamination")

    if back_translation_enabled:
        print(f"  Running back-translation validation (min chrF={min_chrf})…")
        records = _apply(
            records,
            lambda r: _validate_back_translation(r, min_chrf),
            "back_translation",
        )

    # ── Deduplication ────────────────────────────────────────────────────────
    dedup = Deduplicator(threshold=args.dedup_threshold)

    # Build a flat text field for dedup comparison
    for r in records:
        asst = _get_assistant_text(r) or ""
        user_msgs = [m["content"] for m in r.get("messages", []) if m.get("role") == "user"]
        r["_dedup_text"] = " ".join(user_msgs) + " " + asst

    before = len(records)
    records = dedup.filter(records, text_field="_dedup_text")
    stats["duplicate"] = before - len(records)
    for r in records:
        r.pop("_dedup_text", None)
    print(f"  deduplication: dropped {stats['duplicate']} records → {len(records)} remain")

    # ── Strip internal keys not needed downstream ────────────────────────────
    clean_keys = {"source", "chunk_id", "scenario", "system", "messages"}
    records = [{k: v for k, v in r.items() if k in clean_keys} for r in records]

    # ── Train / eval split ───────────────────────────────────────────────────
    random.shuffle(records)
    n_eval = max(1, int(len(records) * args.eval_fraction))
    eval_records  = records[:n_eval]
    train_records = records[n_eval:]

    for path_str, split in [(args.train, train_records), (args.eval, eval_records)]:
        out = Path(path_str)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fout:
            for r in split:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        f"\nDone: {len(train_records)} train → {args.train} | "
        f"{len(eval_records)} eval → {args.eval}"
    )


if __name__ == "__main__":
    main()
