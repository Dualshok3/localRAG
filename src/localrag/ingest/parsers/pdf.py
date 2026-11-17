from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import pymupdf

from .base import ParseError, Section

# MuPDF schreibt Reparaturhinweise kaputter PDFs direkt auf stderr („syntax error: invalid key in dict“).
# Beim Indexieren tausender Dateien überflutet das die Ausgabe; echte Fehler kommen weiter als Ausnahme.
pymupdf.TOOLS.mupdf_display_errors(False)

if TYPE_CHECKING:
    from localrag.ingest.ocr import OcrEngine

# Seiten mit weniger Text gelten als Scan (Kopf-/Fußzeilen eines Scanners zählen nicht)
MIN_TEXT_CHARS = 25
OCR_DPI = 200
OCR_MAX_PAGES = 60  # gescannte Bücher würden sonst Stunden kosten
OCR_MAX_PIXELS = 4000  # längste Bildkante
GOOD_OCR_QUALITY = 0.6  # ab hier gilt das OCR-Ergebnis als lesbarer Text


def parse_pdf(path: Path, ocr: OcrEngine | None = None) -> list[Section]:
    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        raise ParseError(f"PDF nicht lesbar: {exc}") from exc
    with doc:
        if doc.needs_pass:
            raise ParseError("PDF ist passwortgeschützt")
        sections = []
        ocr_pages = 0
        for page_no, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if len(text.strip()) < MIN_TEXT_CHARS and ocr is not None and ocr_pages < OCR_MAX_PAGES:
                ocr_text = _ocr_page(page, ocr)
                ocr_pages += 1
                if len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
            if text.strip():
                sections.append(Section(text=text, page=page_no))
    return sections


_WORD = re.compile(r"[^\W\d_]{3,}")


def text_quality(text: str) -> float:
    """Anteil der Zeichen, die zu echten Wörtern (≥ 3 Buchstaben) gehören. Zeichensalat liegt nahe 0."""
    chars = sum(1 for c in text if not c.isspace())
    return sum(len(w) for w in _WORD.findall(text)) / chars if chars else 0.0


def _ocr_page(page: pymupdf.Page, ocr: OcrEngine) -> str:
    """OCR einer Seite. Gedrehte Scans (realer Fall: Urkunde quer eingescannt) liefern bei 0° nur
    Zeichensalat → dann die anderen Drehungen probieren und die mit den meisten echten Wörtern nehmen."""
    longest = max(page.rect.width, page.rect.height) / 72  # Zoll
    zoom = (min(OCR_DPI, int(OCR_MAX_PIXELS / longest)) if longest else OCR_DPI) / 72

    def run(angle: int) -> str:
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom).prerotate(angle), alpha=False)
        return ocr.recognize(pix.tobytes("png"))

    best = run(0)
    if text_quality(best) >= GOOD_OCR_QUALITY:
        return best
    for angle in (90, 270, 180):
        candidate = run(angle)
        if text_quality(candidate) * len(candidate) > text_quality(best) * len(best):
            best = candidate
        if text_quality(best) >= GOOD_OCR_QUALITY:
            break
    return best
