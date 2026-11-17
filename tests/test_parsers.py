from pathlib import Path

import pytest

from localrag.ingest.parsers import ParseError, SkippedFile, parse

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


def test_docx_with_unknown_media_type_falls_back_to_raw_text(tmp_path: Path):
    # Realer Fall: eingebettetes Bild mit Endung ".undefined" → python-docx bricht ab
    import zipfile

    good = make_docx(tmp_path / "gut.docx", [("h1", "Reiseplan"), ("p", "Hotel in Florenz")])
    broken = tmp_path / "kaputt.docx"
    rel = (
        '<Relationship Id="rIdX" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
        'relationships/image" Target="media/bild.undefined"/></Relationships>'
    )
    with zipfile.ZipFile(good) as src, zipfile.ZipFile(broken, "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "word/_rels/document.xml.rels":
                data = data.decode("utf-8").replace("</Relationships>", rel).encode("utf-8")
            dst.writestr(item, data)
        dst.writestr("word/media/bild.undefined", b"\x89PNG")

    text = "\n".join(s.text for s in parse(broken))
    assert "Reiseplan" in text and "Hotel in Florenz" in text

    from localrag.ingest.parsers.office import _parse_docx_raw

    raw = "\n".join(s.text for s in _parse_docx_raw(broken))
    assert "Reiseplan" in raw and "Hotel in Florenz" in raw


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


@pytest.mark.parametrize("encoding", ["utf-16", "utf-16-le"])  # mit BOM bzw. ohne BOM
def test_utf16_text_files_are_read(tmp_path: Path, encoding: str):
    path = tmp_path / "einstellungen.txt"
    path.write_bytes("Auflösung: 1920x1080\r\nSprache: Deutsch\r\n".encode(encoding))
    assert "Auflösung: 1920x1080" in parse(path)[0].text


def test_real_binary_in_txt_is_still_rejected(tmp_path: Path):
    path = tmp_path / "kaputt.txt"
    path.write_bytes(b"\r\x00\x00\x001.89.214\x00\x07\x00\xff\x10binary\x00\x00data" * 20)
    with pytest.raises(ParseError):
        parse(path)


def test_word_lists_are_skipped(tmp_path: Path):
    path = tmp_path / "wörterbuch.txt"
    path.write_text("\n".join(f"Wort{i}" for i in range(500)), encoding="utf-8")
    with pytest.raises(SkippedFile, match="Wort- oder Datenliste"):
        parse(path)


def test_single_line_data_dumps_are_skipped(tmp_path: Path):
    path = tmp_path / "meta.txt"
    path.write_text("{'a': 1, " * 10_000 + "}", encoding="utf-8")
    with pytest.raises(SkippedFile, match="Daten-Dump"):
        parse(path)


def test_long_notes_with_sentences_are_kept(tmp_path: Path):
    lines = []
    for i in range(400):
        lines.append(f"Punkt {i}")
        lines.append(f"Heute habe ich die Aufgabe Nummer {i} fertig erledigt und abgegeben.")
    path = tmp_path / "tagebuch.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    assert parse(path)
