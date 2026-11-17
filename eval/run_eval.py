"""Misst die Retrieval-Qualität auf einem synthetischen Testkorpus.

Jede Frage hat genau eine erwartete Quelldatei. Gemessen wird, an welcher Stelle diese Datei
in der Trefferliste steht:

- Recall@k: Anteil der Fragen, bei denen die richtige Datei unter den Top k ist
- MRR (Mean Reciprocal Rank): Durchschnitt von 1/Rang (1.0 = immer Platz 1)

Fragetypen:
- keyword:    Frage nutzt Wörter, die im Dokument stehen
- paraphrase: Frage umschreibt den Inhalt mit anderen Wörtern (Schwäche von BM25)

Aufruf:  uv run python eval/run_eval.py [--write-report] [--min-recall-at-5 0.4]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from localrag.index.db import Database
from localrag.ingest.indexer import index_paths
from localrag.retrieval.keyword import keyword_search

EVAL_DIR = Path(__file__).resolve().parent
CORPUS = EVAL_DIR / "corpus"
QUESTIONS = EVAL_DIR / "questions.jsonl"
REPORT = EVAL_DIR / "results" / "bm25_baseline.md"
K_VALUES = (1, 3, 5)


@dataclass
class Result:
    question: str
    expected: str
    type: str
    rank: int | None  # 1-basiert, None = nicht in den Top 10
    top: str | None
    latency_ms: float


def run() -> list[Result]:
    questions = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line]
    results: list[Result] = []
    with tempfile.TemporaryDirectory() as tmp, Database(Path(tmp) / "eval.db") as db:
        index_paths(db, [CORPUS])
        for q in questions:
            started = time.perf_counter()
            hits = keyword_search(db, q["question"], limit=10)
            latency = (time.perf_counter() - started) * 1000
            names = [Path(h.path).name for h in hits]
            rank = names.index(q["expected"]) + 1 if q["expected"] in names else None
            results.append(
                Result(q["question"], q["expected"], q["type"], rank, names[0] if names else None, latency)
            )
    return results


def metrics(results: list[Result]) -> dict[str, float]:
    n = len(results) or 1
    out = {f"recall@{k}": sum(1 for r in results if r.rank and r.rank <= k) / n for k in K_VALUES}
    out["mrr"] = sum(1 / r.rank for r in results if r.rank) / n
    return out


def report(results: list[Result]) -> str:
    groups = {"gesamt": results} | {
        t: [r for r in results if r.type == t] for t in sorted({r.type for r in results})
    }
    lines = [
        "# Eval: BM25-Baseline",
        "",
        f"Korpus: {len(list(CORPUS.glob('*')))} synthetische Dokumente · {len(results)} Fragen · "
        f"Latenz p50 {statistics.median(r.latency_ms for r in results):.1f} ms",
        "",
        "| Fragen | n | " + " | ".join(f"Recall@{k}" for k in K_VALUES) + " | MRR |",
        "|---|---|" + "---|" * (len(K_VALUES) + 1),
    ]
    for name, rs in groups.items():
        m = metrics(rs)
        cells = " | ".join(f"{m[f'recall@{k}']:.2f}" for k in K_VALUES)
        lines.append(f"| {name} | {len(rs)} | {cells} | {m['mrr']:.2f} |")
    lines += [
        "",
        "## Einzelergebnisse",
        "",
        "| Typ | Frage | Erwartet | Rang | Top-Treffer |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        rank = str(r.rank) if r.rank else "–"
        lines.append(f"| {r.type} | {r.question} | {r.expected} | {rank} | {r.top or '–'} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", action="store_true", help=f"Ergebnis nach {REPORT} schreiben")
    parser.add_argument("--min-recall-at-5", type=float, default=None, help="Exit 1, wenn darunter (für CI)")
    args = parser.parse_args()

    results = run()
    text = report(results)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    print(text)
    if args.write_report:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(text, encoding="utf-8")
    if args.min_recall_at_5 is not None and metrics(results)["recall@5"] < args.min_recall_at_5:
        print(f"Recall@5 unter Schwelle {args.min_recall_at_5}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
