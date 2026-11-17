"""Findet indexierbare Dateien unter den angegebenen Wurzelordnern."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from localrag.config import DEFAULT_EXCLUDED_DIRS, DEFAULT_MAX_FILE_MB
from localrag.ingest.parsers import SUPPORTED_EXTENSIONS


@dataclass(frozen=True)
class FileInfo:
    path: Path
    mtime: float
    size: int

    @property
    def filetype(self) -> str:
        return self.path.suffix.lower().lstrip(".")


def iter_files(
    roots: Iterable[Path],
    *,
    excluded_dirs: Iterable[str] = DEFAULT_EXCLUDED_DIRS,
    extensions: Iterable[str] = SUPPORTED_EXTENSIONS,
    max_file_mb: float = DEFAULT_MAX_FILE_MB,
) -> Iterator[FileInfo]:
    excluded = {d.casefold() for d in excluded_dirs}
    exts = {e.lower() for e in extensions}
    max_bytes = int(max_file_mb * 1024 * 1024)

    for root in roots:
        root = Path(root).resolve()
        if root.is_file():
            candidates: Iterable[tuple[str, list[str], list[str]]] = [(str(root.parent), [], [root.name])]
        else:
            candidates = os.walk(root, onerror=lambda _err: None)
        for dirpath, dirnames, filenames in candidates:
            # In-place filtern → os.walk steigt in ausgeschlossene Ordner gar nicht erst ab
            dirnames[:] = [
                d
                for d in dirnames
                if d.casefold() not in excluded and not d.startswith(".") and not _is_link(dirpath, d)
            ]
            for name in filenames:
                if name.startswith("~$"):  # Office-Sperrdateien
                    continue
                if os.path.splitext(name)[1].lower() not in exts:
                    continue
                full = Path(dirpath) / name
                try:
                    st = full.stat()
                except OSError:
                    continue
                if st.st_size == 0 or st.st_size > max_bytes:
                    continue
                yield FileInfo(path=full, mtime=st.st_mtime, size=st.st_size)


def _is_link(dirpath: str, name: str) -> bool:
    """Symlinks und Windows-Junctions überspringen (z. B. „Dokumente und Einstellungen“ → C:/Users).

    Sonst würden Ordner doppelt oder in Schleifen gelesen, sobald man ein ganzes Laufwerk indexiert.
    """
    full = os.path.join(dirpath, name)
    return os.path.islink(full) or os.path.isjunction(full)


def is_under(path: str, root: Path) -> bool:
    """Liegt ``path`` innerhalb von ``root``? (Groß-/Kleinschreibung wie im Dateisystem.)"""
    p = os.path.normcase(os.path.abspath(path))
    r = os.path.normcase(str(Path(root).resolve()))
    return p == r or p.startswith(r.rstrip("\\/") + os.sep)
