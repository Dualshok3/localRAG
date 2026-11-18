"""Hält den Index aktuell, während Dateien entstehen, sich ändern oder verschwinden.

``localrag watch`` (eigener Prozess) oder ``localrag serve --watch`` (im Dienst, teilt sich das geladene
Modell mit der Suche). Ereignisse des Dateisystems werden gesammelt und nach einer kurzen Ruhepause
gemeinsam verarbeitet: Ein Programm, das eine Datei in mehreren Schritten speichert, löst so nur eine
Neuindexierung aus. Verarbeitet wird über denselben inkrementellen Weg wie ``localrag index`` – nur eben
für die betroffenen Dateien statt für ganze Ordner.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from localrag.config import DEFAULT_MAX_FILE_MB
from localrag.index.db import Database
from localrag.index.embedder import Embedder
from localrag.ingest.indexer import IndexStats, index_paths
from localrag.ingest.ocr import OcrEngine
from localrag.ingest.parsers import SUPPORTED_EXTENSIONS
from localrag.ingest.scanner import Exclusions, is_under

DEFAULT_DEBOUNCE_S = 2.0


@dataclass
class WatchBatch:
    paths: list[Path]
    stats: IndexStats
    seconds: float
    error: str | None = None
    changed: list[str] = field(default_factory=list)


class FolderWatcher:
    def __init__(
        self,
        db_path: Path,
        roots: Iterable[Path],
        *,
        exclusions: Exclusions | None = None,
        embedder: Embedder | None = None,
        ocr: OcrEngine | None = None,
        max_file_mb: float = DEFAULT_MAX_FILE_MB,
        debounce: float = DEFAULT_DEBOUNCE_S,
        lock: threading.Lock | None = None,
        on_batch: Callable[[WatchBatch], None] | None = None,
    ) -> None:
        self.db_path = Path(db_path).resolve()
        self.roots = [Path(r).resolve() for r in roots]
        self.rules = exclusions or Exclusions()
        self.embedder = embedder
        self.ocr = ocr
        self.max_file_mb = max_file_mb
        self.debounce = debounce
        self.lock = lock  # z. B. die Modell-Sperre des Dienstes: Einbetten und Suchen nicht gleichzeitig
        self.on_batch = on_batch
        self._pending: set[Path] = set()
        self._last_event = 0.0
        self._cond = threading.Condition()
        self._stopping = False
        self._observer = None
        self._thread: threading.Thread | None = None
        self.batches = 0

    # --- Ereignisse ----------------------------------------------------------------------------

    def relevant(self, path: str, is_directory: bool) -> bool:
        root = next((r for r in self.roots if is_under(path, r)), None)
        if root is None:
            return False
        if os.path.normcase(os.path.abspath(path)).startswith(os.path.normcase(str(self.db_path))):
            return False  # eigene Datenbank inkl. -wal/-shm
        if not is_directory and os.path.splitext(path)[1].lower() not in SUPPORTED_EXTENSIONS:
            return False
        return not self.rules.path_excluded(path, root)

    def notify(self, path: str, is_directory: bool = False) -> None:
        if not self.relevant(path, is_directory):
            return
        with self._cond:
            self._pending.add(Path(path))
            self._last_event = time.monotonic()
            self._cond.notify()

    # --- Verarbeitung --------------------------------------------------------------------------

    def process(self, paths: Iterable[Path]) -> WatchBatch:
        """Betroffene Dateien/Ordner neu einlesen bzw. aus dem Index entfernen."""
        batch = sorted(set(paths))
        started = time.perf_counter()
        error = None
        stats = IndexStats()
        try:
            with self.lock or nullcontext(), Database(self.db_path) as db:
                stats = index_paths(
                    db,
                    batch,
                    excluded_dirs=self.rules,
                    max_file_mb=self.max_file_mb,
                    embedder=self.embedder,
                    ocr=self.ocr,
                )
        except Exception as exc:  # ein kaputter Durchlauf darf den Watcher nicht beenden
            error = f"{type(exc).__name__}: {exc}"
        self.batches += 1
        result = WatchBatch(paths=batch, stats=stats, seconds=time.perf_counter() - started, error=error)
        if self.on_batch:
            self.on_batch(result)
        return result

    def _loop(self, catch_up: bool) -> None:
        if catch_up:
            # Was sich geändert hat, während der Watcher nicht lief
            self.process(self.roots)
        while True:
            with self._cond:
                while not self._stopping and not self._pending:
                    self._cond.wait()
                if self._stopping:
                    return
                # Ruhepause abwarten: solange weitere Ereignisse kommen, wird gesammelt
                while not self._stopping:
                    quiet = time.monotonic() - self._last_event
                    if quiet >= self.debounce:
                        break
                    self._cond.wait(self.debounce - quiet)
                if self._stopping:
                    return
                batch, self._pending = self._pending, set()
            self.process(batch)

    # --- Start/Stopp ---------------------------------------------------------------------------

    def start(self, *, catch_up: bool = False) -> None:
        """Beobachten beginnen; ``catch_up`` = vorher einmal alle Wurzeln abgleichen (im Hintergrund)."""
        handler = _Handler(self)
        observer = Observer()
        for root in self.roots:
            observer.schedule(handler, str(root if root.is_dir() else root.parent), recursive=True)
        observer.daemon = True
        observer.start()
        self._observer = observer
        self._thread = threading.Thread(
            target=self._loop, args=(catch_up,), name="localrag-watch", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
        if self._thread is not None:
            self._thread.join(timeout=30)

    def __enter__(self) -> FolderWatcher:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


class _Handler(FileSystemEventHandler):
    def __init__(self, watcher: FolderWatcher) -> None:
        self.watcher = watcher

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.event_type in ("opened", "closed_no_write"):
            return
        if event.is_directory and event.event_type == "modified":
            return  # Ordner „ändert“ sich bei jeder Datei darin – die Datei selbst meldet sich ohnehin
        # Unter Windows ist bei gelöschten/verschobenen Ordnern nicht bekannt, dass es Ordner waren →
        # vorsichtshalber wie einen Ordner behandeln (räumt dann alles darunter aus dem Index)
        gone = event.event_type in ("deleted", "moved")
        self.watcher.notify(os.fsdecode(event.src_path), event.is_directory or gone)
        dest = getattr(event, "dest_path", "")
        if dest:
            self.watcher.notify(os.fsdecode(dest), event.is_directory)
