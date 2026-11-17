# LocalRAG – Semantische Suche über die eigenen Dateien

> Frag deinen Computer in natürlicher Sprache – komplett lokal, mit Quellenangabe.

**Status:** 🚧 Phase 1 fertig: Hybrid-Suche aus BM25 und Vektoren (bge-m3) über PDF, DOCX, Markdown und TXT. Als Nächstes kommen Antworten per lokalem LLM, siehe [PLAN.md](PLAN.md).

## Quickstart

Voraussetzung: [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Dualshok3/RAG.git && cd RAG
uv sync --extra semantic                      # inkl. PyTorch/CUDA; ohne Extra: nur Stichwortsuche
uv run localrag index "D:\Dokumente"          # inkrementell, beim zweiten Lauf nur Änderungen
uv run localrag search "Darf ich eine Katze halten?"
uv run localrag search "PF-1234" --mode keyword   # schnell, ohne Modell
uv run localrag status --errors
```

| Variable | Bedeutung | Standard |
|---|---|---|
| `LOCALRAG_DB` | Index-Datenbank (auch `--db`) | `~/.localrag/localrag.db` |
| `LOCALRAG_MODELS` | Speicherort der Modelle (~4,5 GB) | Hugging-Face-Cache |
| `LOCALRAG_EMBEDDING_MODEL` | Embedding-Modell | `BAAI/bge-m3` |

Beim ersten Start wird bge-m3 heruntergeladen. Die CLI lädt das Modell bei jeder Suche neu, das dauert
einige Sekunden. Die Suche selbst braucht danach ~30 ms (RTX 5060 Ti). Ein dauerhaft laufender Dienst
folgt in Phase 5.

## Messung: Stichwort vs. Vektor vs. Hybrid

Synthetischer Testkorpus mit 10 Dokumenten und 25 Fragen ([eval/results/latest.md](eval/results/latest.md)):

| Fragen | n | BM25 | Vektor (bge-m3) | **Hybrid (RRF)** |
|---|---|---|---|---|
| Stichworte aus dem Dokument | 11 | 1.00 | 1.00 | **1.00** |
| Umschreibungen („Darf ich eine Katze halten?“ → *Haustiere*) | 10 | 0.00 | 1.00 | **0.90** |
| Exakte Kennungen (Seriennummer, IP, Betrag) | 4 | 1.00 | 0.50 | **1.00** |
| **Gesamt** | 25 | 0.60 | 0.92 | **0.96** |

Angegeben ist Recall@1, also wie oft die richtige Datei auf Platz 1 steht. Bei Recall@5 erreichen
Vektor und Hybrid jeweils 1.00. Das Bild ist klar: BM25 scheitert an Umschreibungen, Vektoren an exakten
Kennungen, und Hybrid gleicht beides aus.

Einschränkung: Mit 10 Dokumenten ist der Korpus klein und leicht. Ein größeres, schwierigeres Eval-Set
ist für Phase 3 geplant. Selbst messen: `uv run python eval/run_eval.py`

## Idee

Die Windows-Suche findet Dateinamen und exakte Wörter. LocalRAG versteht den **Inhalt**:

```text
> localrag ask "Wie lange ist die Kündigungsfrist in meinem Mietvertrag?"

Die Kündigungsfrist beträgt drei Monate zum Monatsende [1].

[1] D:\Dokumente\Wohnung\Mietvertrag_2021.pdf – Seite 4, §5 Kündigung
```

## Geplante Features

- 🔒 **100 % lokal** – keine Daten verlassen den Rechner (Ollama + lokale Embeddings)
- 🔎 **Hybrid-Suche** – semantische Vektorsuche + BM25-Stichwortsuche + Reranking
- 📄 PDF, Word, Excel, PowerPoint, Markdown, E-Mails, Quellcode, OCR für Scans
- ⚡ **Inkrementelle Indexierung** – nur geänderte Dateien werden neu verarbeitet
- 📊 **Messbare Qualität** – Eval-Set mit Recall@k / MRR, Benchmark gegen die Windows-Suche
- 🤖 **MCP-Server** – Claude & Co. können direkt in den eigenen Dateien suchen

## Architektur

```mermaid
flowchart LR
    F[Dateien] --> P[Parser + OCR] --> C[Chunker] --> E[Embeddings bge-m3]
    E --> V[(SQLite sqlite-vec)]
    C --> B[(SQLite FTS5)]
    Q[Frage] --> V & B --> R[RRF + Reranker] --> L[LLM via Ollama] --> A[Antwort + Zitate]
```

## Tech-Stack

Python 3.12 · uv · PyTorch/CUDA · sentence-transformers · SQLite (FTS5 + sqlite-vec) · Ollama · FastAPI · Typer · MCP

## Lizenz

Siehe [LICENSE](LICENSE).
