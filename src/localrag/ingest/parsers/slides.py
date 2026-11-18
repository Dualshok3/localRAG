"""PowerPoint (.pptx): ein Abschnitt pro Folie, mit Titel als Überschrift und den Sprechernotizen."""

from __future__ import annotations

from pathlib import Path

from .base import ParseError, Section


def _shape_texts(shape) -> list[str]:
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    out: list[str] = []
    if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
        for inner in shape.shapes:
            out.extend(_shape_texts(inner))
        return out
    if getattr(shape, "has_table", False) and shape.has_table:
        for row in shape.table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                out.append(" | ".join(cells))
        return out
    if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
        text = "\n".join(p.text for p in shape.text_frame.paragraphs if p.text.strip())
        if text.strip():
            out.append(text)
    return out


def parse_pptx(path: Path) -> list[Section]:
    from pptx import Presentation

    try:
        prs = Presentation(str(path))
    except Exception as exc:
        raise ParseError(f"PowerPoint-Datei nicht lesbar: {exc}") from exc

    sections: list[Section] = []
    for number, slide in enumerate(prs.slides, start=1):
        title_shape = slide.shapes.title
        title = title_shape.text.strip() if title_shape is not None and title_shape.has_text_frame else ""
        parts: list[str] = []
        for shape in slide.shapes:
            if title_shape is not None and shape.shape_id == title_shape.shape_id:
                continue
            parts.extend(_shape_texts(shape))
        if slide.has_notes_slide:
            notes = (
                slide.notes_slide.notes_text_frame.text.strip() if slide.notes_slide.notes_text_frame else ""
            )
            if notes:
                parts.append(f"Notizen: {notes}")
        body = "\n\n".join(parts)
        if title and body:
            body = f"{title}\n\n{body}"
        elif title:
            body = title
        if body.strip():
            sections.append(Section(text=body, page=number, heading_path=title or None))
    return sections
