"""HTML: sichtbarer Text, gegliedert nach Überschriften (h1–h6). Skripte, Styles und Navigation fallen weg."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from .base import HeadingStack, ParseError, Section
from .text import read_text

_SKIP = {"script", "style", "noscript", "template", "svg", "nav", "head", "iframe", "button", "select"}
_BLOCK = {
    "p", "div", "br", "li", "tr", "section", "article", "header", "footer", "main", "aside", "blockquote",
    "pre", "table", "ul", "ol", "dl", "dt", "dd", "figure", "figcaption", "form", "hr", "address",
}  # fmt: skip
_HEADING = re.compile(r"^h([1-6])$")
_CHARSET = re.compile(rb"""<meta[^>]+charset=["']?([\w-]+)""", re.IGNORECASE)


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections: list[Section] = []
        self.headings = HeadingStack()
        self.title = ""
        self._buffer: list[str] = []
        self._skip_depth = 0
        self._heading_level: int | None = None
        self._heading_text: list[str] = []
        self._in_title = False

    def _flush(self) -> None:
        text = re.sub(r"[ \t\r\f\v]+", " ", "".join(self._buffer))
        text = re.sub(r" ?\n[ \n]*", "\n", text).strip()
        if text:
            self.sections.append(Section(text=text, heading_path=self.headings.path))
        self._buffer.clear()

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _SKIP:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif m := _HEADING.match(tag):
            self._flush()
            self._heading_level, self._heading_text = int(m.group(1)), []
        elif tag in _BLOCK:
            self._buffer.append("\n")
        elif tag in ("td", "th"):
            self._buffer.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag == "title":
            self._in_title = False
        elif _HEADING.match(tag) and self._heading_level is not None:
            text = " ".join("".join(self._heading_text).split())
            if text:
                self.headings.push(self._heading_level, text)
            self._heading_level = None
        elif tag in _BLOCK:
            self._buffer.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:  # <title> steht im übersprungenen <head>
            self.title += data
        elif self._skip_depth:
            return
        elif self._heading_level is not None:
            self._heading_text.append(data)
        else:
            self._buffer.append(data)

    def close(self) -> None:
        super().close()
        self._flush()


def html_to_sections(html: str) -> list[Section]:
    parser = _Extractor()
    parser.feed(html)
    parser.close()
    title = " ".join(parser.title.split())
    if title and parser.sections and parser.sections[0].heading_path is None:
        first = parser.sections[0]
        parser.sections[0] = Section(text=first.text, page=first.page, heading_path=title)
    return parser.sections


def html_to_text(html: str) -> str:
    return "\n\n".join(s.text for s in html_to_sections(html))


def _decode(raw: bytes, path: Path) -> str:
    m = _CHARSET.search(raw[:4096])
    if m:
        try:
            return raw.decode(m.group(1).decode("ascii"), errors="replace")
        except LookupError:
            pass
    return read_text(path)


def parse_html(path: Path) -> list[Section]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ParseError(f"Datei nicht lesbar: {exc}") from exc
    return html_to_sections(_decode(raw, path))
