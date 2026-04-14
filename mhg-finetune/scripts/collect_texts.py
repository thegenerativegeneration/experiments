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
    format: str                         # "gutenberg" | "gutenberg_html" | "wikisource_html" | "plain"
    description: str
    fallback_urls: tuple[str, ...] = () # alternative URLs tried in order on 404
    search_term: str = ""               # if set, search de.wikisource.org API as last resort


SOURCES: list[TextSource] = [
    # Wikisource HTML pages for the major MHG epics
    TextSource(
        slug="nibelungenlied",
        url="https://de.wikisource.org/wiki/Das_Nibelungenlied",
        format="wikisource_html",
        description="Das Nibelungenlied (MHG original, Wikisource)",
        search_term="Das Nibelungenlied",
    ),
    TextSource(
        slug="parzival_wolfram",
        url="https://de.wikisource.org/wiki/Parzival",
        format="wikisource_html",
        description="Parzival – Wolfram von Eschenbach (Wikisource)",
        fallback_urls=("https://de.wikisource.org/wiki/Parzival_(Wolfram_von_Eschenbach)",),
        search_term="Parzival Wolfram von Eschenbach",
    ),
    TextSource(
        slug="tristan_gottfried",
        url="https://www.gutenberg.org/files/8970/8970-h/8970-h.htm",
        format="gutenberg_html",
        description="Tristan – Gottfried von Strassburg (Gutenberg #8970)",
        fallback_urls=(
            "https://www.gutenberg.org/files/8970/8970-0.txt",
            "https://www.gutenberg.org/files/8970/8970.txt",
        ),
    ),
    # Wikisource HTML pages (article body extraction)
    TextSource(
        slug="iwein_hartmann",
        url="https://de.wikisource.org/wiki/Iwein_(Hartmann_von_Aue)",
        format="wikisource_html",
        description="Iwein – Hartmann von Aue (Wikisource)",
        fallback_urls=("https://de.wikisource.org/wiki/Iwein",),
        search_term="Iwein Hartmann von Aue",
    ),
    TextSource(
        slug="arme_heinrich",
        url="https://de.wikisource.org/wiki/Der_arme_Heinrich_(Hartmann_von_Aue)",
        format="wikisource_html",
        description="Der arme Heinrich – Hartmann von Aue (Wikisource)",
        fallback_urls=("https://de.wikisource.org/wiki/Der_arme_Heinrich",),
        search_term="Der arme Heinrich Hartmann von Aue",
    ),
    TextSource(
        slug="walther_lieder",
        url="https://de.wikisource.org/wiki/Walther_von_der_Vogelweide",
        format="wikisource_html",
        description="Walther von der Vogelweide – Lieder (Wikisource)",
    ),
    TextSource(
        slug="minnesang_fruehling",
        url="https://de.wikisource.org/wiki/Des_Minnesangs_Fr%C3%BChling",
        format="wikisource_html",
        description="Minnesangs Frühling (Wikisource)",
        fallback_urls=("https://de.wikisource.org/wiki/Minnesangs_Fr%C3%BChling",),
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
# Gutenberg HTML helper
# ---------------------------------------------------------------------------


def _extract_gutenberg_html(html: str) -> str:
    """Extract plain text from a Gutenberg HTML file, then strip boilerplate."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.select("script, style"):
        tag.decompose()
    body = soup.find("body") or soup
    lines = [line.strip() for line in body.get_text(separator="\n").splitlines()]
    lines = [l for l in lines if len(l) > 2]
    return _strip_gutenberg("\n".join(lines))


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
            return _strip_gutenberg(response.text)

        if source.format == "gutenberg_html":
            return _extract_gutenberg_html(response.text)

        if source.format == "wikisource_html":
            return _extract_wikisource(response.text)

        # plain
        return response.text.strip()

    # All hard-coded URLs failed; try Wikisource API search as last resort
    if source.format == "wikisource_html" and source.search_term:
        return _wikisource_fetch_by_search(source.search_term, session)

    assert last_error is not None
    raise last_error


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
