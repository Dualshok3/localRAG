"""Misst die Retrieval-Qualität auf einem synthetischen Testkorpus – pro Suchmodus.

Jede Frage hat genau eine erwartete Quelldatei. Gemessen wird, an welcher Stelle diese Datei
in der Trefferliste steht:

- Recall@k: Anteil der Fragen, bei denen die richtige Datei unter den Top k ist
- MRR (Mean Reciprocal Rank): Durchschnitt von 1/Rang (1.0 = immer Platz 1)

Fragetypen:
- keyword:    Frage nutzt Wörter, die im Dokument stehen
- paraphrase: Frage umschreibt den Inhalt mit anderen Wörtern (Schwäche von BM25)

Aufruf:
  uv run python eval/run_eval.py                       # alle verfügbaren Modi
  uv run python eval/run_eval.py --modes keyword       # nur BM25 (z. B. in CI)
  uv run python eval/run_eval.py --write-report        # nach eval/results/latest.md schreiben
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
from localrag.index.embedder import Embedder, load_embedder, semantic_available
from localrag.ingest.indexer import index_paths
from localrag.retrieval.hybrid import search

EVAL_DIR = Path(__file__).resolve().parent
CORPUS = EVAL_DIR / "corpus"
QUESTIONS = EVAL_DIR / "questions.jsonl"
REPORT = EVAL_DIR / "results" / "latest.md"
K_VALUES = (1, 3, 5)
ALL_MODES = ("keyword", "semantic", "hybrid")


@dataclass
class Result:
    mode: str
    question: str
    expected: str
    type: str
    rank: int | None  # 1-basiert, None = nicht in den Top 10
    top: str | None
    latency_ms: float


def load_questions() -> list[dict[str, str]]:
    return [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line]


def run(modes: list[str], embedder: Embedder | None) -> tuple[list[Result], float]:
    questions = load_questions()
    results: list[Result] = []
    with tempfile.TemporaryDirectory() as tmp, Database(Path(tmp) / "eval.db") as db:
        started = time.perf_counter()
        index_paths(db, [CORPUS], embedder=embedder)
        index_seconds = time.perf_counter() - started
        if embedder is not None:  # Aufwärmen, damit die erste Messung nicht CUDA-Initialisierung enthält
            embedder.embed_query("Aufwärmen")
        for mode in modes:
            for q in questions:
                started = time.perf_counter()
                hits = search(db, q["question"], mode=mode, embedder=embedder, limit=10)  # type: ignore[arg-type]
                latency = (time.perf_counter() - started) * 1000
                names = [Path(h.path).name for h in hits]
                rank = names.index(q["expected"]) + 1 if q["expected"] in names else None
                results.append(
                    Result(
                        mode,
                        q["question"],
                        q["expected"],
                        q["type"],
                        rank,
                        names[0] if names else None,
                        latency,
                    )
                )
    return results, index_seconds


def metrics(results: list[Result]) -> dict[str, float]:
    n = len(results) or 1
    out = {f"recall@{k}": sum(1 for r in results if r.rank and r.rank <= k) / n for k in K_VALUES}
    out["mrr"] = sum(1 / r.rank for r in results if r.rank) / n
    return out


def report(results: list[Result], modes: list[str], embedder: Embedder | None, index_seconds: float) -> str:
    types = sorted({r.type for r in results})
    n_questions = len(load_questions())
    lines = [
        "# Eval: Suchmodi im Vergleich",
        "",
        f"Korpus: {len(list(CORPUS.glob('*')))} synthetische Dokumente · {n_questions} Fragen · "
        f"Embedding-Modell: {embedder.name if embedder else '–'}"
        + (f" ({getattr(embedder, 'device', 'cpu')})" if embedder else "")
        + f" · Indexierung inkl. Vektoren: {index_seconds:.1f} s",
        "",
        "| Modus | Fragen | n | " + " | ".join(f"Recall@{k}" for k in K_VALUES) + " | MRR | Latenz p50 |",
        "|---|---|---|" + "---|" * (len(K_VALUES) + 2),
    ]
    for mode in modes:
        mode_results = [r for r in results if r.mode == mode]
        for group in ["gesamt", *types]:
            rs = mode_results if group == "gesamt" else [r for r in mode_results if r.type == group]
            m = metrics(rs)
            cells = " | ".join(f"{m[f'recall@{k}']:.2f}" for k in K_VALUES)
            latency = statistics.median(r.latency_ms for r in rs)
            label = f"**{mode}**" if group == "gesamt" else ""
            lines.append(f"| {label} | {group} | {len(rs)} | {cells} | {m['mrr']:.2f} | {latency:.1f} ms |")

    lines += [
        "",
        "## Rang der richtigen Datei pro Frage",
        "",
        "| Typ | Frage | Erwartet | " + " | ".join(modes) + " |",
        "|---|---|---|" + "---|" * len(modes),
    ]
    for q in load_questions():
        ranks = []
        for mode in modes:
            r = next(x for x in results if x.mode == mode and x.question == q["question"])
            ranks.append(str(r.rank) if r.rank else "–")
        lines.append(f"| {q['type']} | {q['question']} | {q['expected']} | " + " | ".join(ranks) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default=None, help="Kommagetrennt: keyword,semantic,hybrid")
    parser.add_argument("--model", default=None, help="Embedding-Modell (Standard: BAAI/bge-m3)")
    parser.add_argument("--write-report", action="store_true", help=f"Ergebnis nach {REPORT} schreiben")
    parser.add_argument(
        "--min-recall-at-5", type=float, default=None, help="Exit 1, wenn ein Modus darunter liegt (für CI)"
    )
    args = parser.parse_args()

    if args.modes:
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    else:
        modes = (
            list(ALL_MODES)
            if semantic_available() or (args.model or "").startswith("hashing-")
            else ["keyword"]
        )
    needs_vectors = any(m != "keyword" for m in modes)
    embedder = load_embedder(args.model) if needs_vectors else None

    results, index_seconds = run(modes, embedder)
    text = report(results, modes, embedder, index_seconds)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    print(text)
    if args.write_report:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(text, encoding="utf-8")

    if args.min_recall_at_5 is not None:
        worst = min(metrics([r for r in results if r.mode == m])["recall@5"] for m in modes)
        if worst < args.min_recall_at_5:
            print(f"Recall@5 {worst:.2f} unter Schwelle {args.min_recall_at_5}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
