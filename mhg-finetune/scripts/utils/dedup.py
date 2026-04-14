"""
MinHash-based near-duplicate detection for JSONL chat datasets.

Usage::

    from scripts.utils.dedup import Deduplicator

    dedup = Deduplicator(threshold=0.85)
    kept = dedup.filter(records, text_field="assistant_text")
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

from datasketch import MinHash, MinHashLSH


def _normalise(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _shingles(text: str, k: int = 5) -> set[str]:
    """Return character k-shingles of *text*."""
    return {text[i : i + k] for i in range(max(1, len(text) - k + 1))}


def _build_minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for shingle in _shingles(_normalise(text)):
        m.update(shingle.encode("utf-8"))
    return m


class Deduplicator:
    """Remove near-duplicate records from an iterable of dicts.

    Parameters
    ----------
    threshold:
        Jaccard similarity threshold above which two records are considered
        duplicates.  0.85 is a good default.
    num_perm:
        Number of MinHash permutations.  Higher = more accurate but slower.
    """

    def __init__(self, threshold: float = 0.85, num_perm: int = 128) -> None:
        self.threshold = threshold
        self.num_perm = num_perm
        self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._counter = 0

    def filter(
        self,
        records: Iterable[dict[str, Any]],
        text_field: str,
    ) -> list[dict[str, Any]]:
        """Return a deduplicated list of *records* keyed on *text_field*."""
        kept: list[dict[str, Any]] = []
        for record in records:
            text = record.get(text_field, "")
            if not text:
                kept.append(record)
                continue
            mh = _build_minhash(text, self.num_perm)
            key = f"doc_{self._counter}"
            if not self._lsh.query(mh):
                self._lsh.insert(key, mh)
                self._counter += 1
                kept.append(record)
        return kept
