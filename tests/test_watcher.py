import shutil
import threading
import time
from pathlib import Path

from localrag.index.db import Database
from localrag.index.embedder import HashingEmbedder
from localrag.ingest.scanner import Exclusions
from localrag.ingest.watcher import FolderWatcher, WatchBatch
from localrag.retrieval.hybrid import search


def names(db_path: Path) -> list[str]:
    if not db_path.exists():
        return []
    with Database(db_path) as db:
        return sorted(Path(p).name for p in db.iter_document_paths())


def wait_for(condition, timeout: float = 15.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.05)
    return False


def make_watcher(tmp_path: Path, root: Path, **kw) -> FolderWatcher:
    return FolderWatcher(tmp_path / "index.db", [root], debounce=0.3, **kw)


def test_relevant_skips_unsupported_excluded_and_own_database(tmp_path: Path):
    root = tmp_path / "docs"
    root.mkdir()
    w = FolderWatcher(root / "index.db", [root], exclusions=Exclusions(["*backup*"]))
    assert w.relevant(str(root / "a.md"), False)
    assert not w.relevant(str(root / "bild.png"), False)
    assert not w.relevant(str(root / "Backup" / "a.md"), False)
    assert not w.relevant(str(root / "index.db-wal"), False)
    assert not w.relevant(str(tmp_path / "anderswo.md"), False)
    assert w.relevant(str(root / "Unterordner"), True)


def test_process_adds_updates_and_removes(tmp_path: Path):
    root = tmp_path / "docs"
    root.mkdir()
    w = make_watcher(tmp_path, root, embedder=HashingEmbedder())
    note = root / "notiz.md"
    note.write_text("Kühlschrank reparieren", encoding="utf-8")
    first = w.process([note])
    assert first.stats.added == 1 and first.stats.embedded == 1
    note.write_text("Waschmaschine reparieren lassen", encoding="utf-8")
    assert w.process([note]).stats.updated == 1
    note.unlink()
    assert w.process([note]).stats.removed == 1
    assert names(tmp_path / "index.db") == []


def test_deleted_folder_removes_everything_below(tmp_path: Path):
    root = tmp_path / "docs"
    (root / "Projekt").mkdir(parents=True)
    for n in ("a.md", "b.md"):
        (root / "Projekt" / n).write_text("Text " + n, encoding="utf-8")
    w = make_watcher(tmp_path, root)
    w.process([root])
    shutil.rmtree(root / "Projekt")
    assert w.process([root / "Projekt"]).stats.removed == 2


def test_errors_are_reported_and_do_not_stop_the_watcher(tmp_path: Path):
    root = tmp_path / "docs"
    root.mkdir()
    seen: list[WatchBatch] = []
    w = make_watcher(tmp_path, root, on_batch=seen.append)

    class Boom:
        def __enter__(self):
            raise RuntimeError("Platte voll")

        def __exit__(self, *exc):
            return False

    w.lock = Boom()  # type: ignore[assignment]
    result = w.process([root / "x.md"])
    assert result.error and "Platte voll" in result.error
    assert seen == [result]


def test_watches_real_file_events_end_to_end(tmp_path: Path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "alt.md").write_text("Altes Rezept für Linsensuppe", encoding="utf-8")
    db_path = tmp_path / "index.db"
    batches: list[WatchBatch] = []
    lock = threading.Lock()
    with make_watcher(tmp_path, root, lock=lock, on_batch=batches.append) as w:
        w.stop()  # nur zum Prüfen, dass stop() vor dem ersten Ereignis sauber ist
    w = make_watcher(tmp_path, root, lock=lock, on_batch=batches.append)
    w.start(catch_up=True)
    try:
        assert wait_for(lambda: names(db_path) == ["alt.md"]), "Abgleich beim Start fehlt"

        (root / "Neu").mkdir()
        (root / "Neu" / "rechnung.md").write_text("Rechnung Heizungswartung", encoding="utf-8")
        assert wait_for(lambda: "rechnung.md" in names(db_path)), "neue Datei nicht indexiert"

        (root / "alt.md").unlink()
        assert wait_for(lambda: names(db_path) == ["rechnung.md"]), "gelöschte Datei noch im Index"

        (root / "Neu" / "rechnung.md").rename(root / "Neu" / "heizung.md")
        assert wait_for(lambda: names(db_path) == ["heizung.md"]), "Umbenennung nicht übernommen"

        with Database(db_path) as db:
            assert search(db, "Heizungswartung", mode="keyword")
    finally:
        w.stop()
    assert all(b.error is None for b in batches)


def test_many_quick_saves_are_processed_together(tmp_path: Path):
    root = tmp_path / "docs"
    root.mkdir()
    batches: list[WatchBatch] = []
    w = FolderWatcher(tmp_path / "index.db", [root], debounce=0.5, on_batch=batches.append)
    with w:
        for i in range(5):
            (root / "entwurf.md").write_text(f"Entwurf Version {i}", encoding="utf-8")
            time.sleep(0.05)
        assert wait_for(lambda: batches)
        time.sleep(0.8)
    assert len(batches) == 1
    with Database(tmp_path / "index.db") as db:
        assert search(db, "Version 4", mode="keyword")[0].text == "Entwurf Version 4"
