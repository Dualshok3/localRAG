import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "check_private_files", Path(__file__).parents[1] / "scripts" / "check_private_files.py"
)
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)


def test_blocks_documents_indexes_and_env():
    assert guard.problem("Mietvertrag.pdf")
    assert guard.problem("data/notes.md")
    assert guard.problem("index\\localrag.db")
    assert guard.problem(".env")


def test_allows_code_docs_and_fixtures():
    assert guard.problem("src/localrag/cli.py") is None
    assert guard.problem("eval/corpus/mietvertrag.md") is None
    assert guard.problem("tests/fixtures/sample.pdf") is None
    assert guard.problem(".env.example") is None
