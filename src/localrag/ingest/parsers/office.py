from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph

from .base import HeadingStack, ParseError, Section

# Englische und deutsche Word-Formatvorlagen: "Heading 2", "Überschrift 2", "Title", "Titel"
_HEADING_RE = re.compile(r"^(?:heading|überschrift)\s*(\d)$", re.IGNORECASE)


def _heading_level(par: Paragraph) -> int | None:
    name = (par.style.name if par.style is not None else "") or ""
    if name.lower() in {"title", "titel"}:
        return 0
    m = _HEADING_RE.match(name.strip())
    return int(m.group(1)) if m else None


def _table_text(table: Table) -> str:
    rows = []
    for row in table.rows:
        cells = [c.text.strip() for c in row.cells]
        # Verbundene Zellen tauchen mehrfach auf → Duplikate direkt nebeneinander entfernen
        deduped = [c for i, c in enumerate(cells) if c and (i == 0 or c != cells[i - 1])]
        if deduped:
            rows.append(" | ".join(deduped))
    return "\n".join(rows)


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _parse_docx_raw(path: Path) -> list[Section]:
    """Notlösung: Text direkt aus word/document.xml lesen.

    python-docx bricht ab, wenn z. B. ein eingebettetes Bild einen unbekannten Content-Type hat
    (realer Fall: Bild mit Endung ".undefined"). Der Text ist davon nicht betroffen.
    """
    try:
        with zipfile.ZipFile(path) as zf:
            root = ElementTree.fromstring(zf.read("word/document.xml"))
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError, OSError) as exc:
        raise ParseError(f"DOCX nicht lesbar: {exc}") from exc
    paragraphs = []
    for par in root.iter(f"{_W}p"):
        text = "".join(
            (node.text or "") if node.tag == f"{_W}t" else " "
            for node in par.iter()
            if node.tag in (f"{_W}t", f"{_W}tab")
        )
        if text.strip():
            paragraphs.append(text)
    text = "\n\n".join(paragraphs)
    return [Section(text=text)] if text else []


def parse_docx(path: Path) -> list[Section]:
    try:
        document = docx.Document(str(path))
    except Exception as exc:
        try:
            return _parse_docx_raw(path)
        except ParseError:
            raise ParseError(f"DOCX nicht lesbar: {exc}") from exc

    sections: list[Section] = []
    headings = HeadingStack()
    buffer: list[str] = []

    def flush() -> None:
        text = "\n\n".join(b for b in buffer if b.strip())
        if text:
            sections.append(Section(text=text, heading_path=headings.path))
        buffer.clear()

    for block in document.iter_inner_content():
        if isinstance(block, Paragraph):
            level = _heading_level(block)
            if level is not None and block.text.strip():
                flush()
                headings.push(level, block.text)
            else:
                buffer.append(block.text)
        elif isinstance(block, Table):
            buffer.append(_table_text(block))
    flush()
    return sections
