"""Parser-Registry: Dateiendung → Funktion, die eine Liste von Sections liefert."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from .base import ParseError, Section, SkippedFile
from .mail import parse_eml, parse_msg
from .office import parse_docx
from .pdf import parse_pdf
from .slides import parse_pptx
from .tables import parse_csv, parse_xlsx
from .text import parse_code, parse_markdown, parse_txt
from .web import parse_html

if TYPE_CHECKING:
    from localrag.ingest.ocr import OcrEngine

Parser = Callable[[Path], list[Section]]

PARSERS: dict[str, Parser] = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".md": parse_markdown,
    ".markdown": parse_markdown,
    ".txt": parse_txt,
    ".csv": parse_csv,
    ".xlsx": parse_xlsx,
    ".xlsm": parse_xlsx,
    ".pptx": parse_pptx,
    ".html": parse_html,
    ".htm": parse_html,
    ".eml": parse_eml,
    ".msg": parse_msg,
}

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(PARSERS)

# Quellcode nur auf Wunsch (`code = true` in der Konfiguration): Wer ganze Laufwerke indexiert, hätte
# sonst Bibliotheken und Abhängigkeiten im Index, die eigene Dokumente verdrängen.
CODE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py", ".js", ".mjs", ".ts", ".tsx", ".jsx", ".java", ".kt", ".cs", ".c", ".h", ".cpp", ".hpp",
        ".go", ".rs", ".rb", ".php", ".swift", ".ps1", ".sh", ".bat", ".sql", ".r", ".lua", ".vue",
    }
)  # fmt: skip


def parse(path: Path, ocr: OcrEngine | None = None) -> list[Section]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path, ocr=ocr)  # nur PDFs können Scans ohne Textebene sein
    parser = PARSERS.get(suffix)
    if parser is None and suffix in CODE_EXTENSIONS:
        parser = parse_code
    if parser is None:
        raise ParseError(f"Dateityp {path.suffix} wird nicht unterstützt")
    return parser(path)


__all__ = [
    "CODE_EXTENSIONS",
    "PARSERS",
    "SUPPORTED_EXTENSIONS",
    "ParseError",
    "Section",
    "SkippedFile",
    "parse",
]
