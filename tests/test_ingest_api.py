"""Sofort indexieren: /index (für „Senden an“ und `localrag add`) und /upload (Ziehen & Ablegen)."""

import sys
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from localrag.api.client import ServiceClient
from localrag.api.server import create_app, safe_filename, unique_target
from localrag.cli import app as cli_app
from localrag.cli import sendto_script
from localrag.index.db import Database
from localrag.index.embedder import HashingEmbedder

from .test_service import Bridge

UI = {"X-LocalRAG": "1"}


@pytest.fixture
def service(tmp_path: Path):
    db_path = tmp_path / "svc.db"
    Database(db_path).close()
    app = create_app(
        db_path,
        embedder=HashingEmbedder(),
        load_model=False,
        allowed_hosts=("testserver",),
        inbox=tmp_path / "Eingang",
        max_file_mb=1,
    )
    client = TestClient(app)
    return client, ServiceClient("http://testserver", transport=Bridge(client)), tmp_path


def upload(client: TestClient, name: str, data: bytes, headers: dict | None = None):
    return client.post("/upload", content=data, headers={**UI, "X-Filename": name, **(headers or {})})


def test_upload_saves_to_inbox_and_is_searchable_at_once(service):
    client, _, tmp = service
    resp = upload(client, "Rechnung%20M%C3%A4rz.md", b"Heizungswartung 180 EUR")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert Path(data["saved"]) == tmp / "Eingang" / "Rechnung März.md"
    assert data["added"] == 1 and data["documents"][0]["status"] == "ok"
    hits = client.get("/search", params={"q": "Heizungswartung", "mode": "hybrid"}).json()["hits"]
    assert hits[0]["path"].endswith("Rechnung März.md")


def test_upload_same_file_twice_is_unchanged_different_content_gets_new_name(service):
    client, _, _ = service
    upload(client, "notiz.md", b"Version eins")
    assert upload(client, "notiz.md", b"Version eins").json()["unchanged"] == 1
    second = upload(client, "notiz.md", b"Version zwei").json()
    assert Path(second["saved"]).name == "notiz (2).md"


@pytest.mark.parametrize(
    ("name", "data", "headers", "status"),
    [
        ("bild.png", b"x", None, 415),
        ("leer.md", b"", None, 400),
        ("gross.md", b"x" * (1024 * 1024 + 1), None, 413),
        ("ohne-header.md", b"x", {"X-LocalRAG": ""}, 403),
    ],
)
def test_upload_rejections(service, name, data, headers, status):
    client, _, tmp = service
    assert upload(client, name, data, headers).status_code == status
    assert not (tmp / "Eingang" / name).exists()


def test_upload_cannot_escape_the_inbox(service):
    client, _, tmp = service
    resp = upload(client, "..%2F..%2Fboese.md", b"x")
    assert resp.status_code == 200
    assert Path(resp.json()["saved"]).parent == tmp / "Eingang"


@pytest.mark.parametrize("name", ["", ".env", "..", "%2E%2E"])
def test_safe_filename_rejects_hidden_and_empty(name):
    with pytest.raises(HTTPException):
        safe_filename(name)


def test_safe_filename_strips_paths_and_forbidden_characters():
    assert safe_filename("C:%5Cfremd%5Cbericht.pdf") == "bericht.pdf"
    assert safe_filename('a<b>c:"d|e?.md') == "abcde.md"


def test_unique_target(tmp_path: Path):
    (tmp_path / "a.md").write_bytes(b"1")
    assert unique_target(tmp_path, "a.md", b"1") == tmp_path / "a.md"
    assert unique_target(tmp_path, "a.md", b"2") == tmp_path / "a (2).md"


def test_index_endpoint_indexes_files_outside_any_configured_folder(service):
    client, svc, tmp = service
    usb = tmp / "USB-Stick"
    usb.mkdir()
    (usb / "garantie.md").write_text("Garantie Waschmaschine fünf Jahre", encoding="utf-8")
    (usb / "foto.png").write_bytes(b"x")
    result = svc.index([str(usb)])
    assert result["added"] == 1
    assert [Path(d["path"]).name for d in result["documents"]] == ["garantie.md"]
    assert client.post("/index", json={"paths": [str(usb)]}).status_code == 403  # Header fehlt
    with pytest.raises(RuntimeError, match="Nicht gefunden"):
        svc.index([str(tmp / "gibtsnicht")])


def test_add_command_without_service_indexes_locally(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("localrag.cli._service_for", lambda db: None)
    doc = tmp_path / "brief.md"
    doc.write_text("Kündigung Fitnessstudio", encoding="utf-8")
    db = tmp_path / "i.db"
    result = CliRunner().invoke(cli_app, ["add", str(doc), "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert "1 neu" in result.output and "✓ brief.md" in result.output
    assert CliRunner().invoke(cli_app, ["add", str(tmp_path / "fehlt.md"), "--db", str(db)]).exit_code == 1


def test_sendto_script_passes_every_selected_file():
    script = sendto_script(r"C:\py\python.exe", {"LOCALRAG_MODELS": "D:/m"})
    assert '"""C:\\py\\python.exe"" -m localrag.cli add --popup"' in script
    assert "For Each a In WScript.Arguments" in script
    assert ", 0, False" in script


@pytest.mark.skipif(sys.platform != "win32", reason="„Senden an“ gibt es nur unter Windows")
def test_sendto_enable_disable(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    runner = CliRunner()
    assert runner.invoke(cli_app, ["sendto", "--enable"]).exit_code == 0
    target = tmp_path / "Microsoft" / "Windows" / "SendTo" / "LocalRAG (indexieren).vbs"
    assert "localrag.cli add --popup" in target.read_text(encoding="utf-16")
    assert runner.invoke(cli_app, ["sendto", "--disable"]).exit_code == 0
    assert not target.exists()
