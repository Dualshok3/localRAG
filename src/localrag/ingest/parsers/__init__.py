"""Parser-Registry: Dateiendung → Funktion, die eine Liste von Sections liefert."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .base import ParseError, Section
from .office import parse_docx
from .pdf import parse_pdf
from .text import parse_markdown, parse_txt

Parser = Callable[[Path], list[Section]]

PARSERS: dict[str, Parser] = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".md": parse_markdown,
    ".markdown": parse_markdown,
    ".txt": parse_txt,
}

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(PARSERS)


def parse(path: Path) -> list[Section]:
    parser = PARSERS.get(path.suffix.lower())
    if parser is None:
        raise ParseError(f"Dateityp {path.suffix} wird nicht unterstützt")
    return parser(path)


__all__ = ["PARSERS", "SUPPORTED_EXTENSIONS", "ParseError", "Section", "parse"]
