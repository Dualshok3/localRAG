from __future__ import annotations

import re
from pathlib import Path

from .base import HeadingStack, ParseError, Section

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def read_text(path: Path) -> str:
    """UTF-8 (mit/ohne BOM), sonst Windows-1252 – deckt fast alle deutschen Textdateien ab."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ParseError(f"Datei nicht lesbar: {exc}") from exc
    if b"\x00" in raw[:4096]:
        raise ParseError("Sieht nach Binärdatei aus")
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def parse_txt(path: Path) -> list[Section]:
    text = read_text(path)
    return [Section(text=text)] if text.strip() else []


def parse_markdown(path: Path) -> list[Section]:
    sections: list[Section] = []
    headings = HeadingStack()
    buffer: list[str] = []
    in_code = False

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            sections.append(Section(text=text, heading_path=headings.path))
        buffer.clear()

    for line in read_text(path).splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            in_code = not in_code
        m = None if in_code else _MD_HEADING_RE.match(line)
        if m:
            flush()
            headings.push(len(m.group(1)), m.group(2))
        else:
            buffer.append(line)
    flush()
    return sections
