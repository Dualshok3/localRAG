"""Zentrale Einstellungen, Standardwerte und die optionale Konfigurationsdatei.

Konfigurationsdatei (TOML), Standard ``~/.localrag/config.toml``, per ``LOCALRAG_CONFIG`` änderbar::

    folders = ["D:/Dokumente", "C:/Users/ich/Desktop"]   # was `localrag index` ohne Angabe indexiert
    exclude = ["python_projects", "*backup*", "D:/Dokumente/Alt"]
    max_file_mb = 50
    db = "D:/localrag/index.db"

``exclude``: Einträge ohne Pfadtrenner sind Namen oder Muster (``*``, ``?``) für Ordner *und* Dateien an
beliebiger Stelle. Einträge mit ``/``, ``\\`` oder Laufwerk schließen genau diesen Ordner aus. Die
eingebauten Ausschlüsse (``node_modules``, ``AppData``, ``Windows`` …) gelten zusätzlich, außer
``default_excludes = false``.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ENV_DB = "LOCALRAG_DB"
ENV_CONFIG = "LOCALRAG_CONFIG"

# Ordnernamen (case-insensitive), die beim Scannen komplett übersprungen werden.
DEFAULT_EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".cache",
        ".idea",
        ".vscode",
        "site-packages",
        "appdata",
        "$recycle.bin",
        "system volume information",
        "windows",
        "program files",
        "program files (x86)",
        "programdata",
        "steamapps",
    }
)

# Dateien größer als das werden ignoriert (große PDFs sind oft Scans oder Handbücher).
DEFAULT_MAX_FILE_MB = 50

# Chunking: ~350 Wörter ≈ 450–550 Tokens für deutsche Texte.
CHUNK_MAX_WORDS = 350
CHUNK_OVERLAP_WORDS = 50

CONFIG_TEMPLATE = """\
# LocalRAG-Konfiguration
# Ordner, die `localrag index` (ohne Angabe) indexiert und `localrag watch` beobachtet
folders = []

# Ausschließen – Name/Muster gilt überall (Ordner und Dateien), ein Pfad genau für diesen Ordner.
# Die eingebauten Ausschlüsse (node_modules, .git, AppData, Windows, Program Files …) gelten zusätzlich.
exclude = [
    # "python_projects",
    # "*backup*",
    # "D:/Dokumente/Alt",
]

# Quellcode (.py, .js, .cs …) mitindexieren
code = false

# Größere Dateien überspringen (MB)
max_file_mb = 50

# Sprachmodell für `localrag ask` (OpenAI-kompatibel; Standard: LM Studio auf Port 1234)
# llm_url = "http://127.0.0.1:1234/v1"
# llm_model = "gemma-4-26b-a4b-it"

# Index-Datenbank (sonst ~/.localrag/localrag.db oder LOCALRAG_DB)
# db = "D:/localrag/index.db"
"""


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Config:
    folders: tuple[Path, ...] = ()
    exclude: tuple[str, ...] = ()
    default_excludes: bool = True
    max_file_mb: float = DEFAULT_MAX_FILE_MB
    db: Path | None = None
    code: bool = False  # Quellcode (.py, .js, …) mitindexieren
    llm_url: str | None = None  # OpenAI-kompatibler Server für `ask`
    llm_model: str | None = None
    source: Path | None = field(default=None, compare=False)

    def extensions(self) -> frozenset[str]:
        from localrag.ingest.parsers import CODE_EXTENSIONS, SUPPORTED_EXTENSIONS

        return SUPPORTED_EXTENSIONS | CODE_EXTENSIONS if self.code else SUPPORTED_EXTENSIONS

    def exclusions(self, extra: list[str] | None = None) -> list[str]:
        base = list(DEFAULT_EXCLUDED_DIRS) if self.default_excludes else []
        return base + list(self.exclude) + list(extra or [])


def config_path() -> Path:
    env = os.environ.get(ENV_CONFIG)
    if env:
        return Path(env).expanduser()
    return Path.home() / ".localrag" / "config.toml"


def load_config(path: Path | None = None) -> Config:
    """Liest die Konfiguration; fehlt die Datei, gelten die Standardwerte."""
    path = path or config_path()
    if not path.exists():
        return Config()
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Konfiguration {path} nicht lesbar: {exc}") from exc

    known = {"folders", "exclude", "default_excludes", "max_file_mb", "db", "code", "llm_url", "llm_model"}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"Unbekannte Einträge in {path}: {', '.join(sorted(unknown))}")

    def str_list(key: str) -> list[str]:
        value = data.get(key, [])
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ConfigError(f"{key} in {path} muss eine Liste von Texten sein")
        return value

    max_mb = data.get("max_file_mb", DEFAULT_MAX_FILE_MB)
    if not isinstance(max_mb, int | float) or max_mb <= 0:
        raise ConfigError(f"max_file_mb in {path} muss eine positive Zahl sein")
    db = data.get("db")
    if db is not None and not isinstance(db, str):
        raise ConfigError(f"db in {path} muss ein Pfad (Text) sein")
    for key in ("llm_url", "llm_model"):
        if data.get(key) is not None and not isinstance(data[key], str):
            raise ConfigError(f"{key} in {path} muss ein Text sein")
    return Config(
        folders=tuple(Path(f).expanduser() for f in str_list("folders")),
        exclude=tuple(str_list("exclude")),
        default_excludes=bool(data.get("default_excludes", True)),
        max_file_mb=float(max_mb),
        db=Path(db).expanduser() if db else None,
        code=bool(data.get("code", False)),
        llm_url=data.get("llm_url"),
        llm_model=data.get("llm_model"),
        source=path,
    )


def default_db_path() -> Path:
    """Index-Datenbank: LOCALRAG_DB, sonst ``db`` aus der Konfiguration, sonst ~/.localrag/localrag.db."""
    env = os.environ.get(ENV_DB)
    if env:
        return Path(env).expanduser()
    try:
        configured = load_config().db
    except ConfigError:
        configured = None  # Fehler meldet `localrag index`/`config`, hier nicht abbrechen
    return configured or Path.home() / ".localrag" / "localrag.db"
