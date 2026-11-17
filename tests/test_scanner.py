import subprocess
import sys
from pathlib import Path

import pytest

from localrag.ingest.scanner import iter_files


def names(root: Path) -> list[str]:
    return sorted(str(f.path.relative_to(root)).replace("\\", "/") for f in iter_files([root]))


def test_skips_excluded_hidden_and_unsupported(tmp_path: Path):
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.md").write_text("x", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "y.md").write_text("y", encoding="utf-8")
    (tmp_path / "bild.png").write_bytes(b"png")
    (tmp_path / "leer.txt").write_text("", encoding="utf-8")
    (tmp_path / "notiz.md").write_text("Inhalt", encoding="utf-8")
    assert names(tmp_path) == ["notiz.md"]


@pytest.mark.skipif(sys.platform != "win32", reason="Junctions gibt es nur unter Windows")
def test_does_not_follow_junctions(tmp_path: Path):
    """Sonst würde z. B. „Dokumente und Einstellungen“ → C:/Users beim Laufwerks-Scan alles doppelt lesen."""
    real = tmp_path / "echt"
    real.mkdir()
    (real / "brief.txt").write_text("Hallo", encoding="utf-8")
    link = tmp_path / "verknuepfung"
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(real)], check=True, capture_output=True)
    assert (link / "brief.txt").exists()
    assert names(tmp_path) == ["echt/brief.txt"]
