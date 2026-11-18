import subprocess
import sys
from pathlib import Path

import pytest

from localrag.ingest.scanner import Exclusions, iter_files


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


def test_exclude_rules_names_patterns_and_paths(tmp_path: Path):
    for rel in ["a/notiz.md", "Backup_2020/b.md", "alt/c.md", "x/alt/d.md", "e_entwurf.md", "f.md"]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("Inhalt", encoding="utf-8")
    rules = ["*backup*", str(tmp_path / "alt"), "*_entwurf.md"]
    found = sorted(
        str(f.path.relative_to(tmp_path)).replace("\\", "/")
        for f in iter_files([tmp_path], excluded_dirs=rules)
    )
    # Pfadregel trifft nur genau diesen Ordner, nicht jeden namens „alt“
    assert found == ["a/notiz.md", "f.md", "x/alt/d.md"]


def test_path_excluded_ignores_rules_on_the_root_itself(tmp_path: Path):
    rules = Exclusions(["backup"])
    root = tmp_path / "backup"
    assert not rules.path_excluded(str(root / "a.md"), root)
    assert rules.path_excluded(str(root / "backup" / "a.md"), root)
    assert rules.path_excluded(str(tmp_path / "backup" / "a.md"))
