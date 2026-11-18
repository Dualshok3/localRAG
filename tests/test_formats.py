"""Weitere Formate: Excel, CSV, PowerPoint, HTML, E-Mails. Alle Testdateien entstehen zur Laufzeit."""

import datetime as dt
import sys
import types
from email.message import EmailMessage
from pathlib import Path

import openpyxl
import pytest
from pptx import Presentation
from pptx.util import Inches

from localrag.index.db import Database
from localrag.ingest.indexer import index_paths
from localrag.ingest.parsers import ParseError, SkippedFile, parse
from localrag.ingest.parsers.tables import HEADER_EVERY, MAX_ROWS
from localrag.retrieval.hybrid import search

from .conftest import make_docx


def test_xlsx_one_section_per_sheet_with_values_not_formulas(tmp_path: Path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Haushalt 2024"
    ws.append(["Datum", "Posten", "Betrag"])
    ws.append([dt.date(2024, 3, 1), "Stromabschlag", 70.5])
    ws.append([dt.date(2024, 4, 1), "Stromabschlag", 70.5])
    ws["C4"] = "=C2+C3"
    other = wb.create_sheet("Leer")
    other["A1"] = None
    path = tmp_path / "budget.xlsx"
    wb.save(path)

    sections = parse(path)
    assert len(sections) == 1  # leeres Blatt fällt weg
    s = sections[0]
    assert s.heading_path == "Blatt Haushalt 2024"
    assert "01.03.2024 | Stromabschlag | 70,5" in s.text
    assert "=C2+C3" not in s.text  # Formel nie als Text (ohne Excel gibt es keinen berechneten Wert)


def test_long_tables_repeat_the_header(tmp_path: Path):
    path = tmp_path / "konto.csv"
    lines = ["Datum;Empfänger;Betrag"] + [f"0{i % 9 + 1}.01.2024;Händler {i};{i},00" for i in range(100)]
    path.write_text("\n".join(lines), encoding="utf-8")
    text = parse(path)[0].text
    assert text.count("Datum | Empfänger | Betrag") == 1 + 100 // HEADER_EVERY
    assert "Händler 7 | 7,00" in text


def test_csv_with_commas_and_quotes(tmp_path: Path):
    path = tmp_path / "liste.csv"
    path.write_text('Name,Notiz\n"Muster, Erika","Termin am 3. Mai"\nB,C\n', encoding="utf-8")
    assert "Muster, Erika | Termin am 3. Mai" in parse(path)[0].text


def test_pptx_slides_with_title_tables_and_notes(tmp_path: Path):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Projektstatus"
    slide.placeholders[1].text = "Meilenstein 2 erreicht"
    table = slide.shapes.add_table(2, 2, Inches(1), Inches(4), Inches(4), Inches(1)).table
    table.cell(0, 0).text, table.cell(0, 1).text = "Risiko", "Lieferverzug"
    slide.notes_slide.notes_text_frame.text = "Budget nicht erwähnen"
    prs.slides.add_slide(prs.slide_layouts[6])  # leere Folie
    path = tmp_path / "status.pptx"
    prs.save(path)

    sections = parse(path)
    assert len(sections) == 1
    s = sections[0]
    assert s.page == 1 and s.heading_path == "Projektstatus"
    assert "Meilenstein 2 erreicht" in s.text
    assert "Risiko | Lieferverzug" in s.text
    assert "Notizen: Budget nicht erwähnen" in s.text


def test_html_keeps_headings_drops_scripts_and_navigation(tmp_path: Path):
    path = tmp_path / "seite.html"
    path.write_bytes(
        """<html><head><meta charset="windows-1252"><title>Garantie</title>
        <style>body{color:red}</style><script>alert('x')</script></head>
        <body><nav>Startseite | Kontakt</nav>
        <h1>Garantiebedingungen</h1><p>Die Garantie gilt 24&nbsp;Monate.</p>
        <h2>Ausschlüsse</h2><ul><li>Wasserschäden</li><li>Stürze</li></ul>
        <table><tr><td>Modell</td><td>X1</td></tr></table></body></html>""".encode("cp1252")
    )
    sections = parse(path)
    joined = "\n".join(s.text for s in sections)
    assert "alert" not in joined and "color:red" not in joined and "Startseite" not in joined
    assert sections[0].heading_path == "Garantiebedingungen"
    assert "Garantie gilt 24\xa0Monate" in sections[0].text
    assert sections[1].heading_path == "Garantiebedingungen > Ausschlüsse"
    assert "Wasserschäden" in sections[1].text and "Stürze" in sections[1].text
    assert "Modell | X1" in sections[1].text


def make_eml(path: Path, *, html: bool = False, attachment: tuple[str, bytes] | None = None) -> Path:
    msg = EmailMessage()
    msg["Subject"] = "Ihre Jahresabrechnung"
    msg["From"] = "Stadtwerke Beispielstadt <service@example.org>"
    msg["To"] = "erika@example.org"
    msg["Date"] = "Tue, 05 Mar 2024 09:30:00 +0100"
    if html:
        msg.set_content("<p>Ihr neuer <b>Abschlag</b> beträgt 70 EUR.</p>", subtype="html")
    else:
        msg.set_content("Ihr neuer Abschlag beträgt 70 EUR.")
    if attachment:
        name, data = attachment
        msg.add_attachment(data, maintype="application", subtype="octet-stream", filename=name)
    path.write_bytes(msg.as_bytes())
    return path


def test_eml_headers_body_and_attachments(tmp_path: Path):
    docx = make_docx(tmp_path / "x.docx", [("h1", "Abrechnung"), ("p", "Verbrauch 2.340 kWh")])
    path = make_eml(tmp_path / "mail.eml", attachment=("Abrechnung 2023.docx", docx.read_bytes()))
    sections = parse(path)
    first = sections[0]
    assert first.heading_path == "Ihre Jahresabrechnung"
    for part in ("Von: Stadtwerke Beispielstadt", "Datum: 05.03.2024 09:30", "Abschlag beträgt 70 EUR"):
        assert part in first.text
    att = sections[1]
    assert att.heading_path == "Ihre Jahresabrechnung > Anhang: Abrechnung 2023.docx > Abrechnung"
    assert "2.340 kWh" in att.text


def test_eml_html_only_and_unsupported_attachment(tmp_path: Path):
    path = make_eml(tmp_path / "mail.eml", html=True, attachment=("bild.png", b"\x89PNG"))
    sections = parse(path)
    assert len(sections) == 1
    assert "Ihr neuer Abschlag beträgt 70 EUR." in sections[0].text
    assert "<b>" not in sections[0].text


def test_msg_uses_outlook_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """.msg-Dateien lassen sich ohne Outlook nicht erzeugen → extract_msg durch eine Attrappe ersetzen."""
    attachment = types.SimpleNamespace(longFilename="notiz.txt", data="Zählerstand 4711".encode())
    fake = types.SimpleNamespace(
        subject="Zählerstand",
        body="",
        htmlBody=b"<p>Bitte <i>bis Freitag</i> melden.</p>",
        date=dt.datetime(2024, 3, 5, 9, 30),
        sender="Hausverwaltung",
        to="Erika",
        cc=None,
        attachments=[attachment],
        close=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "extract_msg", types.SimpleNamespace(openMsg=lambda p: fake))
    path = tmp_path / "x.msg"
    path.write_bytes(bytes.fromhex("d0cf11e0a1b11ae1") + bytes(100))  # OLE-Kennung wie bei echten .msg
    sections = parse(path)
    assert "Bitte bis Freitag melden." in sections[0].text
    assert "Datum: 05.03.2024 09:30" in sections[0].text
    assert sections[1].text == "Zählerstand 4711"


def test_broken_office_files_raise_parse_error(tmp_path: Path):
    for name in ("kaputt.xlsx", "kaputt.pptx"):
        (tmp_path / name).write_bytes(b"kein zip")
        with pytest.raises(ParseError):
            parse(tmp_path / name)


def test_new_formats_are_indexed_and_searchable(db: Database, tmp_path: Path):
    root = tmp_path / "docs"
    root.mkdir()
    make_eml(root / "mail.eml")
    (root / "notiz.html").write_text("<h1>Fahrrad</h1><p>Kettenschloss gekauft</p>", encoding="utf-8")
    stats = index_paths(db, [root])
    assert stats.added == 2 and stats.failed == 0
    assert search(db, "Kettenschloss", mode="keyword")[0].path.endswith("notiz.html")
    assert search(db, "Stadtwerke Abschlag", mode="keyword")[0].path.endswith("mail.eml")


def test_msg_that_is_not_outlook_is_skipped_or_read_as_mail(tmp_path: Path):
    """.msg heißen auch Programmdateien (ROS-Nachrichten, Tcl-Kataloge) – die sind kein Fehler."""
    ros = tmp_path / "Pose.msg"
    ros.write_text("float64 x" + chr(10) + "float64 y" + chr(10), encoding="utf-8")
    with pytest.raises(SkippedFile):
        parse(ros)
    as_text = tmp_path / "mail.msg"
    as_text.write_bytes(make_eml(tmp_path / "x.eml").read_bytes())
    assert "Abschlag beträgt 70 EUR" in parse(as_text)[0].text


def test_huge_csv_is_skipped_as_data_export(tmp_path: Path):
    path = tmp_path / "messwerte.csv"
    rows = ["t;wert"] + [f"{i};{i * 0.5}" for i in range(MAX_ROWS + 1)]
    path.write_text(chr(10).join(rows), encoding="utf-8")
    with pytest.raises(SkippedFile):
        parse(path)
