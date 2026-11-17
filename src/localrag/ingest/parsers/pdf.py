from __future__ import annotations

from pathlib import Path

import pymupdf

from .base import ParseError, Section


def parse_pdf(path: Path) -> list[Section]:
    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        raise ParseError(f"PDF nicht lesbar: {exc}") from exc
    with doc:
        if doc.needs_pass:
            raise ParseError("PDF ist passwortgeschützt")
        sections = []
        for page_no, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                sections.append(Section(text=text, page=page_no))
    # Leere Liste = vermutlich Scan ohne Textebene → später OCR (Phase 4)
    return sections
