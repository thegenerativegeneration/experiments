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
    slug: str                           # filename stem used to save the file
    url: str                            # direct plain-text or HTML URL
    format: str                         # "gutenberg" | "wikisource_html" | "plain"
    description: str
    fallback_urls: tuple[str, ...] = () # alternative URLs tried in order on 404
    search_term: str = ""               # if set, search de.wikisource.org API as last resort


SOURCES: list[TextSource] = [
    # All sourced from de.wikisource.org; search_term ensures the API can find
    # the correct page if the primary URL returns 404 or redirects elsewhere.
    TextSource(
        slug="nibelungenlied",
        url="https://de.wikisource.org/wiki/Das_Nibelungenlied",
        format="wikisource_html",
        description="Das Nibelungenlied (MHG original, Wikisource)",
        search_term="Das Nibelungenlied",
    ),
    TextSource(
        slug="parzival_wolfram",
        url="https://de.wikisource.org/wiki/Parzival_(Wolfram_von_Eschenbach)",
        format="wikisource_html",
        description="Parzival – Wolfram von Eschenbach (Wikisource)",
        fallback_urls=("https://de.wikisource.org/wiki/Parzival",),
        search_term="Parzival Wolfram von Eschenbach",
    ),
    TextSource(
        slug="tristan_gottfried",
        url="https://de.wikisource.org/wiki/Tristan_(Gottfried_von_Stra%C3%9Fburg)",
        format="wikisource_html",
        description="Tristan – Gottfried von Strassburg (Wikisource)",
        search_term="Tristan Gottfried von Strassburg",
    ),
    TextSource(
        slug="iwein_hartmann",
        url="https://de.wikisource.org/wiki/Iwein_(Hartmann_von_Aue)",
        format="wikisource_html",
        description="Iwein – Hartmann von Aue (Wikisource)",
        search_term="Iwein Hartmann von Aue",
    ),
    TextSource(
        slug="arme_heinrich",
        url="https://de.wikisource.org/wiki/Der_arme_Heinrich_(Hartmann_von_Aue)",
        format="wikisource_html",
        description="Der arme Heinrich – Hartmann von Aue (Wikisource)",
        search_term="Der arme Heinrich Hartmann von Aue",
    ),
    TextSource(
        slug="walther_lieder",
        url="https://de.wikisource.org/wiki/Walther_von_der_Vogelweide",
        format="wikisource_html",
        description="Walther von der Vogelweide – Lieder (Wikisource)",
        search_term="Walther von der Vogelweide",
    ),
    TextSource(
        slug="minnesang_fruehling",
        url="https://de.wikisource.org/wiki/Des_Minnesangs_Fr%C3%BChling",
        format="wikisource_html",
        description="Minnesangs Frühling (Wikisource)",
        search_term="Des Minnesangs Frühling",
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
    # Remove navigation, TOC, edit links, images, references, and other non-text elements
    for tag in soup.select(
        "sup, .mw-editsection, #toc, .navbox, .sister-project, "
        ".mw-references-wrap, table.wikitable, "
        ".thumb, figure, figcaption, .thumbcaption, .gallery, "
        ".mw-file-description-page, .floatnone, .floatleft, .floatright"
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


_WIKISOURCE_MIN_CHARS = 1_000  # minimum content length to accept a search result


def _wikisource_fetch_by_search(
    query: str, session: requests.Session
) -> str:
    """Search de.wikisource.org via its API and return text of the best match.

    Tries up to 5 search results in order, accepting the first one with
    substantial content (>= ``_WIKISOURCE_MIN_CHARS`` characters).
    """
    resp = session.get(
        "https://de.wikisource.org/w/api.php",
        headers=HEADERS,
        timeout=30,
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srnamespace": 0,
            "srlimit": 5,
            "format": "json",
        },
    )
    resp.raise_for_status()
    results = resp.json().get("query", {}).get("search", [])
    if not results:
        raise ValueError(f"Wikisource API: no results for {query!r}")

    for result in results:
        title = result["title"]
        url = "https://de.wikisource.org/wiki/" + title.replace(" ", "_")
        try:
            page_resp = session.get(url, headers=HEADERS, timeout=30)
            page_resp.raise_for_status()
            text = _extract_wikisource(page_resp.text)
            if len(text) >= _WIKISOURCE_MIN_CHARS:
                return text
        except requests.RequestException:
            continue

    raise ValueError(f"Wikisource API: no usable page found for {query!r}")


# ---------------------------------------------------------------------------
# Content validation
# ---------------------------------------------------------------------------

# Unicode ranges for CJK (Chinese / Japanese / Korean) characters
_CJK_RE = re.compile(
    r"[\u2E80-\u2EFF\u2F00-\u2FDF\u3000-\u303F\u3040-\u309F\u30A0-\u30FF"
    r"\u3100-\u312F\u3200-\u32FF\u3300-\u33FF\u3400-\u4DBF\u4E00-\u9FFF"
    r"\uF900-\uFAFF\uFE30-\uFE4F\u20000-\u2A6DF\u2A700-\u2B73F]"
)

_MIN_CONTENT_CHARS = 5_000   # shortest acceptable text
_MAX_CJK_RATIO = 0.05        # ≤5 % CJK characters allowed


def _validate_content(text: str, source_slug: str) -> None:
    """Raise ``ValueError`` if *text* looks like the wrong document.

    Checks:
    * Minimum length — catches empty or near-empty pages.
    * CJK character ratio — rejects accidentally downloaded Chinese/Japanese texts.
    """
    if len(text) < _MIN_CONTENT_CHARS:
        raise ValueError(
            f"{source_slug}: content too short ({len(text):,} chars < {_MIN_CONTENT_CHARS:,})"
        )
    cjk_count = len(_CJK_RE.findall(text))
    ratio = cjk_count / max(len(text), 1)
    if ratio > _MAX_CJK_RATIO:
        raise ValueError(
            f"{source_slug}: {ratio:.1%} CJK characters — this is not a German text"
        )


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
    urls_to_try = [source.url, *source.fallback_urls]
    last_error: Exception | None = None
    for url in urls_to_try:
        try:
            response = session.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                last_error = exc
                continue
            raise

        if source.format == "gutenberg":
            text = _strip_gutenberg(response.text)
        elif source.format == "wikisource_html":
            text = _extract_wikisource(response.text)
        else:
            text = response.text.strip()

        try:
            _validate_content(text, source.slug)
        except ValueError as exc:
            tqdm.write(f"  warn  {exc} — trying next source")
            last_error = exc
            continue

        return text

    # All hard-coded URLs failed or produced invalid content; try Wikisource API search
    if source.format == "wikisource_html" and source.search_term:
        text = _wikisource_fetch_by_search(source.search_term, session)
        _validate_content(text, source.slug)
        return text

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{source.slug}: no URL produced valid content")


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
