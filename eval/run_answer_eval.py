"""Misst die Qualität der *Antworten* (nicht nur der Suche) auf dem synthetischen Korpus.

Für jede Frage aus ``answers.jsonl``: Quellen suchen wie ``localrag ask``, Antwort vom lokalen Modell holen,
dann bewerten:

- **richtig**: Ein Prüfmodell vergleicht die Antwort mit der Kernaussage der Referenz (LLM-as-Judge,
  Temperatur 0). Optionale Details der Referenz dürfen fehlen, falsche Angaben nicht.
- **belegt**: Das Prüfmodell prüft, ob jede Aussage durch die übergebenen Quellen gedeckt ist
- **Zitat korrekt**: Die Antwort zitiert mit ``[n]`` mindestens einmal die Datei, aus der die Referenz stammt
- **Weiß-nicht korrekt**: Bei Fragen, die der Korpus nicht beantwortet, sagt das Modell genau das
  (gezählt über den festen Satz aus dem Prompt, ohne Prüfmodell)

Modelle als ``ID`` (Server aus LOCALRAG_LLM_URL, Standard LM Studio) oder ``ID@http://host:port/v1``.
Bei LM Studio (Port 1234) wird jedes Modell vorher exklusiv geladen.

  uv run python eval/run_answer_eval.py --models gemma-4-26b-a4b-it,qwen3.5-9b --judge gemma-4-26b-a4b-it
  # gespeicherte Antworten behalten, nur ein neues Modell dazu und alles neu bewerten:
  uv run python eval/run_answer_eval.py --reuse --models crit@http://127.0.0.1:8080/v1 --judge crit@...
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from localrag.generation.answer import AnswerResult, retrieve_sources, stream_answer
from localrag.generation.llm import ENV_LLM_URL, ChatClient, Message
from localrag.index.db import Database
from localrag.index.embedder import load_embedder, semantic_available
from localrag.ingest.indexer import index_paths
from localrag.retrieval.keyword import SearchHit

ROOT = Path(__file__).parent
CORPUS = ROOT / "corpus"
QUESTIONS = ROOT / "answers.jsonl"
REPORT = ROOT / "results" / "answers.md"
RAW = ROOT / "results" / "answers_raw.json"

JUDGE_PROMPT = """Du bewertest die Antwort eines Assistenten auf eine Frage zu persönlichen Dokumenten.

Frage: {question}
Kernaussage (muss inhaltlich enthalten sein): {reference}
Optionale Details (dürfen fehlen, dürfen aber nicht falsch wiedergegeben werden): {details}

Quellen, die der Assistent bekommen hat:
{sources}

Antwort des Assistenten:
{answer}

Bewerte zwei Dinge:
1. RICHTIG: Trifft die Antwort die Kernaussage und widerspricht weder ihr noch den optionalen Details?
   Andere Formulierungen und zusätzliche korrekte Angaben sind in Ordnung. Fehlende optionale Details
   machen eine Antwort NICHT falsch. Falsche Zahlen, Daten oder eine weggelassene Bedingung aus der
   Kernaussage schon.
2. BELEGT: Ist jede Aussage der Antwort durch die Quellen gedeckt (nichts erfunden)?

Antworte ausschließlich mit einer Zeile JSON:
{{"richtig": true|false, "belegt": true|false, "grund": "..."}}"""


@dataclass
class Row:
    model: str
    question: str
    qtype: str
    answer: str
    seconds: float
    abstained: bool
    cites_source: bool | None
    correct: bool | None = None
    grounded: bool | None = None
    reason: str = ""
    sources: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Endpoint:
    model: str
    url: str

    @classmethod
    def parse(cls, spec: str) -> Endpoint:
        model, _, url = spec.partition("@")
        return cls(model, url or os.environ.get(ENV_LLM_URL) or "http://localhost:1234/v1")

    @property
    def is_lmstudio(self) -> bool:
        return ":1234" in self.url

    def client(self) -> ChatClient:
        return ChatClient(self.url, self.model)


def lms_load(endpoint: Endpoint) -> None:
    """Modell in LM Studio exklusiv laden (16 GB VRAM reichen nicht für zwei große Modelle)."""
    lms = shutil.which("lms")
    if lms is None or not endpoint.is_lmstudio:
        return
    subprocess.run([lms, "unload", "--all"], check=False, capture_output=True)
    subprocess.run([lms, "load", endpoint.model, "-y"], check=True, capture_output=True)


def load_questions() -> list[dict[str, str]]:
    return [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]


def judge(
    client: ChatClient, q: dict[str, str], answer: str, sources: list[SearchHit]
) -> tuple[bool, bool, str]:
    source_text = "\n\n".join(f"[{i}] {Path(h.path).name}: {h.text}" for i, h in enumerate(sources, 1))
    prompt = JUDGE_PROMPT.format(
        question=q["question"],
        reference=q["reference"],
        details=q.get("details") or "–",
        sources=source_text,
        answer=answer,
    )
    raw = client.chat([Message("user", prompt)], temperature=0.0, max_tokens=400)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        data = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        data = {}
    return bool(data.get("richtig")), bool(data.get("belegt")), str(data.get("grund", raw[:200]))


def retrieve_all(questions: list[dict[str, str]], embedder_name: str | None) -> dict[str, list[SearchHit]]:
    embedder = load_embedder(embedder_name) if semantic_available() else load_embedder("hashing-256")
    with tempfile.TemporaryDirectory() as tmp, Database(Path(tmp) / "eval.db") as db:
        index_paths(db, [CORPUS], embedder=embedder)
        sources = {
            q["question"]: retrieve_sources(db, q["question"], embedder=embedder, mode="hybrid")
            for q in questions
        }
    del embedder  # Grafikspeicher für das Sprachmodell freigeben
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass
    return sources


def answer_all(
    endpoint: Endpoint, questions: list[dict[str, str]], sources_for: dict[str, list[SearchHit]]
) -> list[Row]:
    rows: list[Row] = []
    lms_load(endpoint)
    with endpoint.client() as client:
        for q in questions:
            result = AnswerResult(question=q["question"], sources=sources_for[q["question"]])
            started = time.perf_counter()
            for _ in stream_answer(client, result):
                pass
            cited_files = {Path(h.path).name for _, h in result.cited_sources}
            rows.append(
                Row(
                    model=endpoint.model,
                    question=q["question"],
                    qtype=q["type"],
                    answer=result.text,
                    seconds=time.perf_counter() - started,
                    abstained=result.is_no_answer,
                    cites_source=(q["source"] in cited_files) if q["source"] else None,
                    sources=[Path(h.path).name for h in result.sources],
                )
            )
            print(
                f"{endpoint.model:28s} {q['type']:15s} {rows[-1].seconds:5.1f}s  {result.text[:70]!r}",
                flush=True,
            )
    return rows


def judge_all(
    endpoint: Endpoint,
    rows: list[Row],
    questions: list[dict[str, str]],
    sources_for: dict[str, list[SearchHit]],
) -> None:
    lms_load(endpoint)
    by_q = {q["question"]: q for q in questions}
    with endpoint.client() as client:
        for row in rows:
            q = by_q[row.question]
            row.correct = row.grounded = None
            row.reason = ""
            if q["type"] == "unanswerable" or row.abstained:
                continue
            row.correct, row.grounded, row.reason = judge(client, q, row.answer, sources_for[row.question])


def pct(values: list[bool]) -> str:
    return f"{sum(values)}/{len(values)}" if values else "–"


def report(rows: list[Row], judge_model: str) -> str:
    lines = [
        "# Antwortqualität",
        "",
        f"Erzeugt von `eval/run_answer_eval.py`. Prüfmodell: `{judge_model}`. Fragen: `eval/answers.jsonl` "
        "(erfundener Korpus). Suche: hybrid, dieselben Quellen für alle Modelle.",
        "",
        "| Modell | richtig | belegt | Zitat auf richtige Datei | „weiß ich nicht“ korrekt "
        "| fälschlich verweigert | Antwortzeit p50 |",
        "|---|---|---|---|---|---|---|",
    ]
    for model in dict.fromkeys(r.model for r in rows):
        mine = [r for r in rows if r.model == model]
        answerable = [r for r in mine if r.qtype != "unanswerable"]
        unanswerable = [r for r in mine if r.qtype == "unanswerable"]
        lines.append(
            f"| {model} | {pct([bool(r.correct) for r in answerable])} "
            f"| {pct([bool(r.grounded) for r in answerable if not r.abstained])} "
            f"| {pct([bool(r.cites_source) for r in answerable])} "
            f"| {pct([r.abstained for r in unanswerable])} "
            f"| {pct([r.abstained for r in answerable])} "
            f"| {statistics.median(r.seconds for r in mine):.1f} s |"
        )
    lines += ["", "## Falsch oder unbelegt", ""]
    for r in rows:
        if r.qtype == "unanswerable" and not r.abstained:
            lines.append(
                f"- **{r.model}** · {r.question} → antwortet statt „weiß ich nicht“: {r.answer[:160]!r}"
            )
        elif r.qtype != "unanswerable" and (r.abstained or r.correct is False or r.grounded is False):
            what = "verweigert" if r.abstained else ("falsch" if r.correct is False else "unbelegt")
            lines.append(f"- **{r.model}** · {r.question} → {what}: {r.answer[:160]!r} ({r.reason[:160]})")
    return "\n".join(lines) + "\n"


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="", help="Kommagetrennt: ID oder ID@URL")
    parser.add_argument("--judge", required=True, help="Prüfmodell: ID oder ID@URL")
    parser.add_argument(
        "--reuse", action="store_true", help=f"Antworten aus {RAW.name} behalten, neu bewerten"
    )
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    questions = load_questions()
    known = {q["question"] for q in questions}
    endpoints = [Endpoint.parse(s) for s in args.models.split(",") if s.strip()]
    rows: list[Row] = []
    if args.reuse and RAW.exists():
        regenerate = {e.model for e in endpoints}
        rows = [
            Row(**r)
            for r in json.loads(RAW.read_text(encoding="utf-8"))
            if r["model"] not in regenerate and r["question"] in known
        ]
    sources_for = retrieve_all(questions, args.embedding_model)
    for endpoint in endpoints:
        rows += answer_all(endpoint, questions, sources_for)
    judge_endpoint = Endpoint.parse(args.judge)
    judge_all(judge_endpoint, rows, questions, sources_for)

    text = report(rows, judge_endpoint.model)
    if args.write_report:  # zuerst speichern – die Konsolenausgabe kann unter Windows scheitern
        REPORT.parent.mkdir(exist_ok=True)
        REPORT.write_text(text, encoding="utf-8")
    RAW.parent.mkdir(exist_ok=True)
    RAW.write_text(
        json.dumps([dataclasses.asdict(r) for r in rows], ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
