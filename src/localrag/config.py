"""Zentrale Einstellungen und Standardwerte."""

from __future__ import annotations

import os
from pathlib import Path

ENV_DB = "LOCALRAG_DB"

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
    }
)

# Dateien größer als das werden ignoriert (große PDFs sind oft Scans oder Handbücher).
DEFAULT_MAX_FILE_MB = 50

# Chunking: ~350 Wörter ≈ 450–550 Tokens für deutsche Texte.
CHUNK_MAX_WORDS = 350
CHUNK_OVERLAP_WORDS = 50


def default_db_path() -> Path:
    """Pfad der Index-Datenbank. Per Umgebungsvariable LOCALRAG_DB überschreibbar (z. B. auf D:)."""
    env = os.environ.get(ENV_DB)
    if env:
        return Path(env).expanduser()
    return Path.home() / ".localrag" / "localrag.db"
