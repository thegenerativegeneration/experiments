#!/usr/bin/env python3
"""
collect_texts.py — Download public-domain Middle High German source texts.

Texts are fetched from Project Gutenberg and Wikisource, stripped of front/back
matter, and saved as plain UTF-8 .txt files in data/raw/.

Usage::

    python scripts/collect_texts.py [--output-dir data/raw]

The script is idempotent: already-downloaded files are skipped.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import NamedTuple

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Source catalogue
# ---------------------------------------------------------------------------

class TextSource(NamedTuple):
    slug: str           # filename stem used to save the file
    url: str            # direct plain-text or HTML URL
    format: str         # "gutenberg" | "wikisource_html" | "plain"
    description: str


SOURCES: list[TextSource] = [
    # Project Gutenberg plain-text files (UTF-8)
    TextSource(
        slug="nibelungenlied",
        url="https://www.gutenberg.org/cache/epub/7420/pg7420.txt",
        format="gutenberg",
        description="Das Nibelungenlied (MHG original, Gutenberg #7420)",
    ),
    TextSource(
        slug="parzival_wolfram",
        url="https://www.gutenberg.org/cache/epub/19393/pg19393.txt",
        format="gutenberg",
        description="Parzival – Wolfram von Eschenbach (Gutenberg #19393)",
    ),
    TextSource(
        slug="tristan_gottfried",
        url="https://www.gutenberg.org/cache/epub/8970/pg8970.txt",
        format="gutenberg",
        description="Tristan – Gottfried von Strassburg (Gutenberg #8970)",
    ),
    # Wikisource HTML pages (article body extraction)
    TextSource(
        slug="iwein_hartmann",
        url="https://de.wikisource.org/wiki/Iwein",
        format="wikisource_html",
        description="Iwein – Hartmann von Aue (Wikisource)",
    ),
    TextSource(
        slug="arme_heinrich",
        url="https://de.wikisource.org/wiki/Der_arme_Heinrich",
        format="wikisource_html",
        description="Der arme Heinrich – Hartmann von Aue (Wikisource)",
    ),
    TextSource(
        slug="walther_lieder",
        url="https://de.wikisource.org/wiki/Walther_von_der_Vogelweide",
        format="wikisource_html",
        description="Walther von der Vogelweide – Lieder (Wikisource)",
    ),
    TextSource(
        slug="minnesang_fruehling",
        url="https://de.wikisource.org/wiki/Minnesangs_Fr%C3%BChling",
        format="wikisource_html",
        description="Minnesangs Frühling (Wikisource)",
    ),
]

# ---------------------------------------------------------------------------
# Gutenberg helpers
# ---------------------------------------------------------------------------

# Regex patterns to strip Gutenberg header / footer boilerplate
_GUTENBERG_START = re.compile(
    r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG", re.IGNORECASE
)
_GUTENBERG_END = re.compile(
    r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG", re.IGNORECASE
)


def _strip_gutenberg(raw: str) -> str:
    """Remove Project Gutenberg header and footer."""
    start_match = _GUTENBERG_START.search(raw)
    end_match = _GUTENBERG_END.search(raw)
    if start_match:
        raw = raw[start_match.end():]
    if end_match:
        raw = raw[: end_match.start()]
    return raw.strip()


# ---------------------------------------------------------------------------
# Wikisource helpers
# ---------------------------------------------------------------------------


def _extract_wikisource(html: str) -> str:
    """Extract article body text from a Wikisource HTML page."""
    soup = BeautifulSoup(html, "lxml")
    # Remove navigation, TOC, edit links, references
    for tag in soup.select(
        "sup, .mw-editsection, #toc, .navbox, .sister-project, "
        ".mw-references-wrap, table.wikitable"
    ):
        tag.decompose()
    content_div = (
        soup.select_one("#mw-content-text .mw-parser-output")
        or soup.select_one("#mw-content-text")
    )
    if content_div is None:
        return soup.get_text(separator="\n")
    lines = [line.strip() for line in content_div.get_text(separator="\n").splitlines()]
    # Drop very short lines (page numbers, single characters)
    lines = [l for l in lines if len(l) > 2]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "mhg-finetune-collector/1.0 "
        "(academic research; github.com/thegenerativegeneration/experiments)"
    )
}


def fetch_text(source: TextSource, session: requests.Session) -> str:
    response = session.get(source.url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    if source.format == "gutenberg":
        raw = response.text
        return _strip_gutenberg(raw)

    if source.format == "wikisource_html":
        return _extract_wikisource(response.text)

    # plain
    return response.text.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Directory to save raw text files (default: data/raw)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files that already exist",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    errors: list[str] = []

    for source in tqdm(SOURCES, desc="Downloading"):
        dest = out_dir / f"{source.slug}.txt"
        if dest.exists() and not args.force:
            tqdm.write(f"  skip  {source.slug} (already exists)")
            continue

        tqdm.write(f"  fetch {source.description}")
        try:
            text = fetch_text(source, session)
            dest.write_text(text, encoding="utf-8")
            tqdm.write(f"  saved {dest}  ({len(text):,} chars)")
        except Exception as exc:
            msg = f"  ERROR {source.slug}: {exc}"
            tqdm.write(msg, file=sys.stderr)
            errors.append(msg)

        time.sleep(1)  # be polite to servers

    if errors:
        print(f"\n{len(errors)} error(s):", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\nAll texts saved to {out_dir}/")


if __name__ == "__main__":
    main()
