from pathlib import Path

import pytest
from typer.testing import CliRunner

from localrag.cli import app
from localrag.config import ConfigError, default_db_path, load_config
from localrag.index.db import Database
from localrag.ingest.indexer import index_paths


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_missing_file_means_defaults(tmp_path: Path):
    cfg = load_config(tmp_path / "gibtsnicht.toml")
    assert cfg.folders == () and cfg.source is None
    assert "node_modules" in cfg.exclusions()


def test_reads_folders_excludes_and_db(tmp_path: Path, monkeypatch):
    p = write(
        tmp_path / "c.toml",
        'folders = ["D:/Dokumente"]\nexclude = ["*backup*"]\nmax_file_mb = 10\ndb = "D:/x/index.db"\n',
    )
    cfg = load_config(p)
    assert cfg.folders == (Path("D:/Dokumente"),)
    assert cfg.max_file_mb == 10
    assert cfg.exclusions(["extra"])[-2:] == ["*backup*", "extra"]
    monkeypatch.setenv("LOCALRAG_CONFIG", str(p))
    assert default_db_path() == Path("D:/x/index.db")
    monkeypatch.setenv("LOCALRAG_DB", str(tmp_path / "env.db"))
    assert default_db_path() == tmp_path / "env.db"  # Umgebungsvariable gewinnt


def test_default_excludes_can_be_turned_off(tmp_path: Path):
    cfg = load_config(write(tmp_path / "c.toml", "default_excludes = false\nexclude = ['a']\n"))
    assert cfg.exclusions() == ["a"]


@pytest.mark.parametrize(
    "text", ["folders = 'D:/x'\n", "max_file_mb = -1\n", "foldrs = []\n", "folders = [\n"]
)
def test_invalid_config_gives_clear_error(tmp_path: Path, text: str):
    with pytest.raises(ConfigError):
        load_config(write(tmp_path / "c.toml", text))


def test_index_without_paths_uses_configured_folders_and_drops_newly_excluded(tmp_path: Path, monkeypatch):
    docs = tmp_path / "docs"
    (docs / "Backup").mkdir(parents=True)
    write(docs / "brief.md", "Kündigung der Wohnung")
    write(docs / "Backup" / "brief.md", "Kündigung der Wohnung")
    cfg = tmp_path / "config.toml"
    db = tmp_path / "index.db"
    monkeypatch.setenv("LOCALRAG_CONFIG", str(cfg))
    runner = CliRunner()

    write(cfg, f"folders = ['{docs.as_posix()}']\ndb = '{db.as_posix()}'\n")
    result = runner.invoke(app, ["index", "--no-embed"])
    assert result.exit_code == 0, result.output
    with Database(db) as d:
        assert len(list(d.iter_document_paths())) == 2

    # Ausschluss nachträglich ergänzt → schon indexierte Datei verschwindet beim nächsten Lauf
    write(cfg, f"folders = ['{docs.as_posix()}']\ndb = '{db.as_posix()}'\nexclude = ['backup']\n")
    result = runner.invoke(app, ["index", "--no-embed"])
    assert result.exit_code == 0, result.output
    assert "1 entfernt" in result.output
    with Database(db) as d:
        assert [Path(p).name for p in d.iter_document_paths()] == ["brief.md"]


def test_index_without_paths_or_config_explains_what_to_do(tmp_path: Path):
    result = CliRunner().invoke(app, ["index", "--db", str(tmp_path / "i.db")])
    assert result.exit_code == 2
    assert "config --init" in result.output


def test_config_init_writes_template_that_loads(tmp_path: Path, monkeypatch):
    path = tmp_path / "sub" / "config.toml"
    monkeypatch.setenv("LOCALRAG_CONFIG", str(path))
    assert CliRunner().invoke(app, ["config", "--init"]).exit_code == 0
    assert load_config(path).folders == ()
    assert CliRunner().invoke(app, ["config"]).exit_code == 0


def test_excluded_files_are_removed_only_below_the_indexed_root(db: Database, tmp_path: Path):
    root = tmp_path / "Backup"
    root.mkdir()
    write(root / "a.md", "Text")
    index_paths(db, [root], excluded_dirs=["backup"])
    assert len(list(db.iter_document_paths())) == 1  # Wurzel selbst ausdrücklich gewollt
