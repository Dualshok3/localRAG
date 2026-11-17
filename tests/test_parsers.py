from pathlib import Path

import pytest

from localrag.ingest.parsers import ParseError, parse

from .conftest import make_docx, make_pdf


def test_pdf_pages_are_numbered(tmp_path: Path):
    sections = parse(make_pdf(tmp_path / "a.pdf", ["Seite eins", "", "Seite drei"]))
    assert [(s.page, s.text.strip()) for s in sections] == [(1, "Seite eins"), (3, "Seite drei")]


def test_broken_pdf_raises_parse_error(tmp_path: Path):
    bad = tmp_path / "kaputt.pdf"
    bad.write_bytes(b"%PDF-1.4 kein echtes pdf")
    with pytest.raises(ParseError):
        parse(bad)


def test_docx_headings_become_heading_path(tmp_path: Path):
    path = make_docx(
        tmp_path / "v.docx",
        [("h1", "Vertrag"), ("p", "Einleitung"), ("h2", "Urlaub"), ("p", "30 Tage"), ("h2", "Gehalt")],
    )
    sections = parse(path)
    assert [(s.heading_path, s.text) for s in sections] == [
        ("Vertrag", "Einleitung"),
        ("Vertrag > Urlaub", "30 Tage"),
    ]


def test_docx_tables_are_included(tmp_path: Path):
    import docx

    document = docx.Document()
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "Posten", "Betrag"
    table.cell(1, 0).text, table.cell(1, 1).text = "Strom", "84 EUR"
    document.save(str(tmp_path / "t.docx"))
    assert "Strom | 84 EUR" in parse(tmp_path / "t.docx")[0].text


def test_markdown_headings_and_code_blocks(tmp_path: Path):
    md = tmp_path / "n.md"
    md.write_text("# Titel\nText\n```\n# kein heading\n```\n## Unter\nMehr\n", encoding="utf-8")
    sections = parse(md)
    assert [s.heading_path for s in sections] == ["Titel", "Titel > Unter"]
    assert "# kein heading" in sections[0].text


def test_txt_falls_back_to_cp1252(tmp_path: Path):
    txt = tmp_path / "alt.txt"
    txt.write_bytes("Größe".encode("cp1252"))
    assert parse(txt)[0].text == "Größe"


def test_unsupported_type(tmp_path: Path):
    f = tmp_path / "x.xyz"
    f.write_text("x")
    with pytest.raises(ParseError):
        parse(f)
