"""Findet indexierbare Dateien unter den angegebenen Wurzelordnern."""

from __future__ import annotations

import fnmatch
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


def _norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(path)).rstrip("\\/")


class Exclusions:
    """Ausschlussregeln: Namen/Muster (``node_modules``, ``*backup*``) gelten für Ordner und Dateien an
    beliebiger Stelle, Pfade (``D:/Dokumente/Alt``) genau für diesen Ordner samt Inhalt."""

    def __init__(self, rules: Iterable[str] = DEFAULT_EXCLUDED_DIRS) -> None:
        self.names: set[str] = set()
        self.patterns: list[str] = []
        self.paths: list[str] = []
        for rule in rules:
            rule = rule.strip()
            if not rule:
                continue
            if "/" in rule or "\\" in rule or (len(rule) >= 2 and rule[1] == ":"):
                self.paths.append(_norm(os.path.expanduser(rule)))
            elif any(ch in rule for ch in "*?["):
                self.patterns.append(rule.casefold())
            else:
                self.names.add(rule.casefold())

    def name_excluded(self, name: str) -> bool:
        folded = name.casefold()
        return folded in self.names or any(fnmatch.fnmatchcase(folded, p) for p in self.patterns)

    def dir_excluded(self, full: str) -> bool:
        name = os.path.basename(full.rstrip("\\/"))
        if self.name_excluded(name):
            return True
        if not self.paths:
            return False
        norm = _norm(full)
        return any(norm == p or norm.startswith(p + os.sep) for p in self.paths)

    def path_excluded(self, path: str, root: Path | None = None) -> bool:
        """Liegt die Datei unter einem ausgeschlossenen Ordner oder passt ihr Name auf eine Regel?

        Mit ``root`` werden nur Ordner *unterhalb* der Wurzel geprüft – wer ``D:/Backup`` ausdrücklich
        indexiert, will dessen Inhalt, auch wenn „backup“ ausgeschlossen ist.
        """
        p = Path(path)
        if self.name_excluded(p.name) or p.name.startswith("~$"):
            return True
        parents = p.parents
        if root is not None:
            r = _norm(str(root))
            parents = [q for q in parents if _norm(str(q)).startswith(r + os.sep)]  # type: ignore[assignment]
        for parent in parents:
            if parent.name and (self.name_excluded(parent.name) or parent.name.startswith(".")):
                return True
        norm = _norm(path)
        return any(norm.startswith(q + os.sep) for q in self.paths)


def iter_files(
    roots: Iterable[Path],
    *,
    excluded_dirs: Iterable[str] = DEFAULT_EXCLUDED_DIRS,
    extensions: Iterable[str] = SUPPORTED_EXTENSIONS,
    max_file_mb: float = DEFAULT_MAX_FILE_MB,
) -> Iterator[FileInfo]:
    rules = excluded_dirs if isinstance(excluded_dirs, Exclusions) else Exclusions(excluded_dirs)
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
                if not d.startswith(".")
                and not rules.dir_excluded(os.path.join(dirpath, d))
                and not _is_link(dirpath, d)
            ]
            for name in filenames:
                if name.startswith("~$"):  # Office-Sperrdateien
                    continue
                if os.path.splitext(name)[1].lower() not in exts:
                    continue
                if rules.name_excluded(name) and not root.is_file():
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
