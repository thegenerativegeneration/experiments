#!/usr/bin/env python3
"""
preprocess.py — Clean and chunk raw MHG text files into overlapping passages.

Each raw .txt file in data/raw/ is:
  1. Cleaned  – line numbers, footnote markers, and editorial insertions removed
  2. Chunked  – split into passages of ~TARGET_TOKENS tokens with OVERLAP overlap
  3. Written  – appended to data/chunks.jsonl

Each JSONL record has the shape::

    {
        "source": "nibelungenlied",
        "chunk_id": 42,
        "text": "Uns ist in alten mæren wunders vil geseit ...",
        "token_count": 312
    }

Usage::

    python scripts/preprocess.py [--raw-dir data/raw] [--output data/chunks.jsonl]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import tiktoken
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET_TOKENS = 350      # target chunk size in tokens
OVERLAP_TOKENS = 50      # overlap between consecutive chunks
MIN_TOKENS = 80          # discard chunks shorter than this

# tiktoken encoding close enough to most modern LLM tokenisers
_ENCODING = tiktoken.get_encoding("cl100k_base")

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

# Match verse line numbers like "1.", "12.", "(13)", "[14]", "1234."
_LINE_NUMBER_RE = re.compile(r"^\s*[\(\[]?\d{1,4}[\)\]]?\.?\s*$")

# Remove editorial insertions in square brackets (e.g., "[sic]", "[fehlt]")
_EDITORIAL_RE = re.compile(r"\[(?!.*\].*\[).*?\]")

# Collapse multiple blank lines into one
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def _clean(text: str) -> str:
    lines = text.splitlines()
    cleaned: list[str] = []
    for line in lines:
        # Skip pure line-number lines
        if _LINE_NUMBER_RE.match(line):
            continue
        # Remove inline editorial notes
        line = _EDITORIAL_RE.sub("", line)
        cleaned.append(line)
    result = "\n".join(cleaned)
    result = _MULTI_BLANK_RE.sub("\n\n", result)
    return result.strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _token_count(text: str) -> int:
    return len(_ENCODING.encode(text))


def _chunk_text(
    text: str,
    target: int = TARGET_TOKENS,
    overlap: int = OVERLAP_TOKENS,
    min_tokens: int = MIN_TOKENS,
) -> list[str]:
    """Split *text* into overlapping token-bounded passages."""
    # Work at word level for clean boundaries
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        # Grow window until we hit the target token count
        end = start + 1
        while end <= len(words):
            candidate = " ".join(words[start:end])
            if _token_count(candidate) >= target:
                break
            end += 1

        passage = " ".join(words[start:end])
        if _token_count(passage) >= min_tokens:
            chunks.append(passage)

        if end >= len(words):
            break

        # Move start forward, keeping `overlap` tokens of context
        # Walk backward from `end` until we've kept ~overlap tokens
        overlap_start = end - 1
        while overlap_start > start:
            overlap_text = " ".join(words[overlap_start:end])
            if _token_count(overlap_text) >= overlap:
                break
            overlap_start -= 1
        start = overlap_start

    return chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output", default="data/chunks.jsonl")
    parser.add_argument(
        "--target-tokens", type=int, default=TARGET_TOKENS,
        help=f"Target tokens per chunk (default: {TARGET_TOKENS})",
    )
    parser.add_argument(
        "--overlap-tokens", type=int, default=OVERLAP_TOKENS,
        help=f"Overlap tokens between chunks (default: {OVERLAP_TOKENS})",
    )
    args = parser.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        print(f"ERROR: raw directory not found: {raw_dir}", file=sys.stderr)
        print("Run scripts/collect_texts.py first.", file=sys.stderr)
        sys.exit(1)

    txt_files = sorted(raw_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {raw_dir}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for txt_file in tqdm(txt_files, desc="Preprocessing"):
            raw = txt_file.read_text(encoding="utf-8", errors="replace")
            cleaned = _clean(raw)
            chunks = _chunk_text(
                cleaned,
                target=args.target_tokens,
                overlap=args.overlap_tokens,
            )
            for i, chunk in enumerate(chunks):
                record = {
                    "source": txt_file.stem,
                    "chunk_id": i,
                    "text": chunk,
                    "token_count": _token_count(chunk),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            tqdm.write(f"  {txt_file.stem}: {len(chunks)} chunks")
            total_chunks += len(chunks)

    print(f"\nTotal: {total_chunks} chunks → {out_path}")


if __name__ == "__main__":
    main()
