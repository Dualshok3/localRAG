"""E-Mails: .eml (Standardformat, z. B. Thunderbird, Export aus Webmailern) und .msg (Outlook).

Ein Abschnitt mit Kopfzeilen und Text, dazu ein Abschnitt je Anhang in einem unterstützten Format.
Kopfzeilen kommen bewusst in den Text: „Mail von den Stadtwerken im März“ soll Absender und Datum finden.
"""

from __future__ import annotations

import email
import email.policy
import os
import re
import tempfile
from collections.abc import Iterable
from email.message import EmailMessage
from email.utils import parsedate_to_datetime
from pathlib import Path

from .base import ParseError, Section, SkippedFile
from .web import html_to_text

MAX_ATTACHMENT_MB = 25


def _date(value: object) -> str:
    if not value:
        return ""
    try:
        return parsedate_to_datetime(str(value)).strftime("%d.%m.%Y %H:%M")
    except (TypeError, ValueError):
        return str(value)


def mail_section(*, subject: str, sender: str, to: str, cc: str, date: str, body: str) -> Section | None:
    fields = (("Betreff", subject), ("Von", sender), ("An", to), ("Cc", cc), ("Datum", date))
    head = [f"{label}: {value}" for label, value in fields if value]
    text = "\n".join(head) + ("\n\n" + body.strip() if body.strip() else "")
    return Section(text=text, heading_path=subject or None) if text.strip() else None


def attachment_sections(subject: str, attachments: Iterable[tuple[str, bytes]]) -> list[Section]:
    """Anhänge über die normalen Parser lesen (Umweg über eine temporäre Datei)."""
    from . import SUPPORTED_EXTENSIONS, parse

    sections: list[Section] = []
    for name, data in attachments:
        ext = os.path.splitext(name)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS or ext in (".eml", ".msg"):
            continue  # keine Mails in Mails – sonst drohen Schleifen
        if not data or len(data) > MAX_ATTACHMENT_MB * 1024 * 1024:
            continue
        with tempfile.TemporaryDirectory(prefix="localrag-") as tmp:
            path = Path(tmp) / f"anhang{ext}"
            path.write_bytes(data)
            try:
                parsed = parse(path)
            except (ParseError, SkippedFile):
                continue
            except Exception:  # ein kaputter Anhang darf die Mail nicht unlesbar machen
                continue
        label = f"Anhang: {name}"
        for s in parsed:
            heading = " > ".join(p for p in (subject, label, s.heading_path) if p)
            sections.append(Section(text=s.text, page=s.page, heading_path=heading))
    return sections


def parse_eml(path: Path) -> list[Section]:
    try:
        with path.open("rb") as f:
            msg = email.message_from_binary_file(f, policy=email.policy.default)
    except OSError as exc:
        raise ParseError(f"Datei nicht lesbar: {exc}") from exc
    assert isinstance(msg, EmailMessage)
    subject = str(msg.get("subject", "") or "")

    body = ""
    part = msg.get_body(preferencelist=("plain", "html"))
    if part is not None:
        try:
            content = part.get_content()
        except (LookupError, ValueError):  # unbekannter Zeichensatz
            content = (part.get_payload(decode=True) or b"").decode("latin-1", errors="replace")
        body = html_to_text(content) if part.get_content_subtype() == "html" else content

    attachments: list[tuple[str, bytes]] = []
    for att in msg.iter_attachments():
        name = att.get_filename()
        if name:
            attachments.append((name, att.get_payload(decode=True) or b""))

    first = mail_section(
        subject=subject,
        sender=str(msg.get("from", "") or ""),
        to=str(msg.get("to", "") or ""),
        cc=str(msg.get("cc", "") or ""),
        date=_date(msg.get("date")),
        body=body,
    )
    return ([first] if first else []) + attachment_sections(subject, attachments)


_OLE_MAGIC = bytes.fromhex("d0cf11e0a1b11ae1")
_RFC822 = re.compile(rb"^(From|Subject|Received|Date|To|Return-Path):", re.MULTILINE | re.IGNORECASE)


def parse_msg(path: Path) -> list[Section]:
    import extract_msg

    try:
        with path.open("rb") as f:
            head = f.read(4096)
    except OSError as exc:
        raise ParseError(f"Datei nicht lesbar: {exc}") from exc
    if not head.startswith(_OLE_MAGIC):
        # .msg ist auch Endung für Programmdateien (ROS-Nachrichten, Tcl-Kataloge …); manche Programme
        # speichern Mails als Text unter .msg → dann wie .eml lesen
        if _RFC822.search(head):
            return parse_eml(path)
        raise SkippedFile("keine Outlook-Nachricht (andere Datei mit Endung .msg)")
    try:
        msg = extract_msg.openMsg(str(path))
    except Exception as exc:
        raise ParseError(f"Outlook-Nachricht nicht lesbar: {exc}") from exc
    try:
        subject = msg.subject or ""
        body = msg.body or ""
        if not body.strip() and getattr(msg, "htmlBody", None):
            raw = msg.htmlBody
            body = html_to_text(raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw)
        date = msg.date.strftime("%d.%m.%Y %H:%M") if getattr(msg, "date", None) else ""
        attachments: list[tuple[str, bytes]] = []
        for att in getattr(msg, "attachments", []):
            name = getattr(att, "longFilename", None) or getattr(att, "shortFilename", None)
            data = getattr(att, "data", None)
            if name and isinstance(data, bytes):
                attachments.append((name, data))
        first = mail_section(
            subject=subject,
            sender=msg.sender or "",
            to=msg.to or "",
            cc=msg.cc or "",
            date=date,
            body=body,
        )
    finally:
        msg.close()
    return ([first] if first else []) + attachment_sections(subject, attachments)
