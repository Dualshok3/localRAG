"""Pre-commit guard: blocks files that could contain private data.

LocalRAG indexes personal documents, so real documents must never be committed.
Allowed exception: synthetic test files under tests/fixtures/ or eval/fixtures/.
"""

from __future__ import annotations

import sys
from pathlib import PurePosixPath

BLOCKED_DIRS = {"data", "index", "models", "private", ".lancedb"}

DOCUMENT_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".odt", ".ods",
    ".eml", ".msg", ".pst", ".ost", ".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff",
    ".zip", ".7z", ".rar",
}

DATABASE_EXTENSIONS = {".db", ".sqlite", ".sqlite3", ".lance", ".parquet", ".npy", ".faiss"}

ALLOWED_PREFIXES = ("tests/fixtures/", "eval/fixtures/", "docs/images/")


def problem(path: str) -> str | None:
    p = PurePosixPath(path.replace("\\", "/"))
    if p.parts and p.parts[0] in BLOCKED_DIRS:
        return f"liegt im gesperrten Ordner '{p.parts[0]}/'"
    suffix = p.suffix.lower()
    if suffix in DATABASE_EXTENSIONS:
        return "sieht nach Index/Datenbank aus"
    if suffix in DOCUMENT_EXTENSIONS and not str(p).startswith(ALLOWED_PREFIXES):
        return f"Dokument/Bild ({suffix}) ausserhalb von {', '.join(ALLOWED_PREFIXES)}"
    if p.name.lower().startswith(".env") and p.name.lower() != ".env.example":
        return "Umgebungsdatei mit moeglichen Secrets"
    return None


def main(argv: list[str]) -> int:
    hits = [(f, reason) for f in argv if (reason := problem(f))]
    for f, reason in hits:
        print(f"BLOCKIERT: {f} -> {reason}")
    if hits:
        print("\nPrivate Daten gehoeren nicht ins Repo. Synthetische Testdateien nach tests/fixtures/ legen.")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
