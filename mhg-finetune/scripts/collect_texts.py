#!/usr/bin/env python3
"""
collect_texts.py — Download public-domain Middle High German source texts.

Texts are fetched from the Mittelhochdeutsche Begriffsdatenbank (MHDBDB),
University of Salzburg, which publishes its TEI-encoded corpus on GitHub at
https://github.com/DigitalHumanitiesCraft/mhdbdb-tei-only (CC BY-NC-SA 4.0).

The raw `<w>` token forms are extracted from the TEI XML and saved as plain
UTF-8 .txt files in data/raw/.

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
from xml.etree import ElementTree as ET

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Source catalogue
# ---------------------------------------------------------------------------

# Base URL for raw MHDBDB TEI files on GitHub (main branch).
# Each file is named <SIGLE>.tei.xml where SIGLE is the short work identifier
# used by the MHDBDB (see authority-files/works.xml in the same repo).
_MHDBDB_BASE = (
    "https://raw.githubusercontent.com/"
    "DigitalHumanitiesCraft/mhdbdb-tei-only/main/tei"
)


class TextSource(NamedTuple):
    slug: str         # output filename stem
    url: str          # raw GitHub URL of the TEI XML file
    description: str


SOURCES: list[TextSource] = [
    TextSource(
        slug="nibelungenlied",
        url=f"{_MHDBDB_BASE}/NBB.tei.xml",
        description="Nibelungenlied (MHDBDB siglum NBB)",
    ),
    TextSource(
        slug="parzival_wolfram",
        url=f"{_MHDBDB_BASE}/PZ.tei.xml",
        description="Parzival – Wolfram von Eschenbach (MHDBDB siglum PZ)",
    ),
    TextSource(
        slug="tristan_gottfried",
        url=f"{_MHDBDB_BASE}/TR.tei.xml",
        description="Tristan – Gottfried von Straßburg (MHDBDB siglum TR)",
    ),
    TextSource(
        slug="iwein_hartmann",
        url=f"{_MHDBDB_BASE}/IW.tei.xml",
        description="Iwein – Hartmann von Aue (MHDBDB siglum IW)",
    ),
    TextSource(
        slug="arme_heinrich",
        url=f"{_MHDBDB_BASE}/DAH.tei.xml",
        description="Der arme Heinrich – Hartmann von Aue (MHDBDB siglum DAH)",
    ),
    TextSource(
        slug="walther_lieder",
        url=f"{_MHDBDB_BASE}/WVV.tei.xml",
        description="Walther von der Vogelweide – Lyrik (MHDBDB siglum WVV)",
    ),
    TextSource(
        slug="minnesang_fruehling",
        url=f"{_MHDBDB_BASE}/MNL.tei.xml",
        description="Namenlose Lieder / Minnesangs Frühling (MHDBDB siglum MNL)",
    ),
]

# ---------------------------------------------------------------------------
# TEI extraction
# ---------------------------------------------------------------------------

_TEI_NS = "http://www.tei-c.org/ns/1.0"
_W   = f"{{{_TEI_NS}}}w"
_PC  = f"{{{_TEI_NS}}}pc"
_L   = f"{{{_TEI_NS}}}l"
_LG  = f"{{{_TEI_NS}}}lg"
_DIV = f"{{{_TEI_NS}}}div"
_P   = f"{{{_TEI_NS}}}p"
_BODY = f"{{{_TEI_NS}}}body"


def _line_tokens(container: ET.Element) -> list[str]:
    """Return word-form tokens from a single <l> or <p> element."""
    tokens: list[str] = []
    for child in container.iter():
        if child.tag == _W:
            # itertext() handles any inline <hi> elements (enlarged initials)
            text = "".join(child.itertext()).strip()
            if text:
                tokens.append(text)
        elif child.tag == _PC:
            punct = "".join(child.itertext()).strip()
            if punct:
                join = child.get("join", "")
                if join in ("left", "both") and tokens:
                    tokens[-1] += punct  # attach to preceding word
                else:
                    tokens.append(punct)
    return tokens


def _extract_mhdbdb_tei(xml: str) -> str:
    """Extract plain MHG verse text from an MHDBDB TEI-XML file.

    Processes ``<l>`` (verse lines) and ``<p>`` (prose paragraphs) inside
    ``<body>``, inserting blank lines between stanzas/chapters (``<lg>`` /
    ``<div>`` boundaries).
    """
    root = ET.fromstring(xml)
    body = root.find(f".//{_BODY}")
    if body is None:
        raise ValueError("No <body> element found in TEI XML")

    result: list[str] = []

    def _walk(el: ET.Element) -> None:
        tag = el.tag
        if tag == _L:
            tokens = _line_tokens(el)
            if tokens:
                result.append(" ".join(tokens))
        elif tag in (_LG, _DIV, _P):
            # Blank line between stanzas / chapter / paragraph sections.
            # NOTE: <p> is treated as a structural wrapper (like <lg>/<div>)
            # because in MHDBDB TEI files the verse body is encoded as
            # <body><p><l>…</l>…</p></body>.
            if result and result[-1] != "":
                result.append("")
            for child in el:
                _walk(child)
        else:
            for child in el:
                _walk(child)

    _walk(body)
    return "\n".join(result).strip()


# ---------------------------------------------------------------------------
# Content validation
# ---------------------------------------------------------------------------

# Unicode ranges for CJK (Chinese / Japanese / Korean) characters
_CJK_RE = re.compile(
    r"[\u2E80-\u2EFF\u2F00-\u2FDF\u3000-\u303F\u3040-\u309F\u30A0-\u30FF"
    r"\u3100-\u312F\u3200-\u32FF\u3300-\u33FF\u3400-\u4DBF\u4E00-\u9FFF"
    r"\uF900-\uFAFF\uFE30-\uFE4F]"
)

_MIN_CONTENT_CHARS = 1_000   # shortest acceptable text (guards against empty/error pages)
_MAX_CJK_RATIO = 0.05        # ≤5 % CJK characters allowed


def _validate_content(text: str, source_slug: str) -> None:
    """Raise ``ValueError`` if *text* is too short or contains too many CJK chars."""
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
    response = session.get(source.url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    text = _extract_mhdbdb_tei(response.text)
    _validate_content(text, source.slug)
    return text


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
