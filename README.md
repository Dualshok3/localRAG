# LocalRAG – Semantische Suche über die eigenen Dateien

> Frag deinen Computer in natürlicher Sprache – komplett lokal, mit Quellenangabe.

**Status:** 🚧 Hybrid-Suche aus BM25 und Vektoren (bge-m3) über PDF, DOCX, Markdown und TXT. Antworten mit Quellenangabe per lokalem LLM (LM Studio) funktionieren, die Antwortqualität ist aber noch nicht gemessen. Details in [PLAN.md](PLAN.md).

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
| `LOCALRAG_LLM_URL` | OpenAI-kompatibler Chat-Server | `http://localhost:1234/v1` (LM Studio) |
| `LOCALRAG_LLM_MODEL` | Modell-ID auf dem Server | bereits geladenes Modell |

### Fragen beantworten (lokales LLM)

`localrag ask` sucht die passenden Textstellen und lässt ein lokales Sprachmodell daraus eine Antwort mit
Quellenangaben formulieren. Getestet mit [LM Studio](https://lmstudio.ai). Jeder OpenAI-kompatible Server
funktioniert ebenso (llama.cpp, vLLM, Ollama).

```bash
lms server start                     # LM-Studio-Server
lms load qwen3.5-9b                  # oder ein anderes Modell
uv run localrag ask "Wie hoch ist der neue Stromabschlag?"
```

Das Modell bekommt nur die gefundenen Textstellen und soll jede Aussage mit `[n]` belegen. Steht die
Antwort nicht in den Quellen, sagt es „Dazu habe ich in deinen Dateien nichts gefunden.“

Beim ersten Start wird bge-m3 heruntergeladen. Ohne Dienst lädt die CLI das Modell bei jeder Suche neu,
das dauert rund 20 s. Schneller geht es mit dem Hintergrunddienst:

```bash
uv run localrag serve            # hält das Modell geladen, nur unter 127.0.0.1 erreichbar
uv run localrag search "…"       # nutzt den Dienst automatisch → Antwort in Millisekunden
uv run localrag search "…" --local   # Dienst ignorieren
```

Der Dienst bietet auch eine HTTP-API (`/search`, `/ask`, `/health`, Doku unter `http://127.0.0.1:8765/docs`).
Weil er Inhalte deiner Dateien herausgibt, bindet er standardmäßig nur an 127.0.0.1.

### Als Werkzeug für Claude (MCP)

Mit `localrag mcp` bekommen Claude Desktop, Claude Code und andere MCP-Clients drei Werkzeuge:
`search_files`, `read_passage` (Volltext eines Treffers samt Nachbarabschnitten) und `index_status`.
Der Assistent kann damit mehrfach suchen und gezielt nachlesen.

```bash
uv sync --extra semantic --extra mcp
claude mcp add localrag -- uv --directory C:\pfad\zu\RAG run localrag mcp --allow "D:\Dokumente"
```

Für Claude Desktop in `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "localrag": {
      "command": "uv",
      "args": ["--directory", "C:\\pfad\\zu\\RAG", "run", "localrag", "mcp", "--allow", "D:\\Dokumente"],
      "env": { "LOCALRAG_DB": "D:\\localrag\\localrag.db", "LOCALRAG_MODELS": "D:\\localrag\\models" }
    }
  }
}
```

> **Datenschutz:** Was die Werkzeuge zurückgeben, landet im Kontext des Assistenten. Bei einem
> Cloud-Modell verlassen diese Ausschnitte deinen Rechner, anders als bei `localrag ask` mit LM Studio.
> Mit `--allow` lässt sich die Herausgabe auf bestimmte Ordner begrenzen. Treffer aus anderen Ordnern
> verschweigt der Server dann komplett.

## Messung: Stichwort vs. Vektor vs. Hybrid

Synthetischer Testkorpus mit 16 Dokumenten und 35 Fragen ([eval/results/latest.md](eval/results/latest.md)).
Er enthält absichtlich ähnliche Dokumente als Fallen: Stromrechnung 2023 und 2024, Wohnungs- und
Garagenmietvertrag, alter und aktueller Arbeitsvertrag.

| Fragen | n | BM25 | Vektor (bge-m3) | Hybrid (RRF) |
|---|---|---|---|---|
| Stichworte aus dem Dokument | 12 | 1.00 | 1.00 | 1.00 |
| Umschreibungen („Darf ich eine Katze halten?“ → *Haustiere*) | 13 | 0.23 | **0.92** | 0.69 |
| Exakte Kennungen (Seriennummer, IP, Betrag) | 5 | **1.00** | 0.40 | **1.00** |
| Ähnliche Dokumente unterscheiden („Strom **2023**“) | 5 | 0.20 | **1.00** | 0.60 |
| **Gesamt** | 35 | 0.60 | **0.89** | 0.83 |

Angegeben ist Recall@1, also wie oft die richtige Datei auf Platz 1 steht. Bei Recall@3 erreichen
Vektor und Hybrid jeweils 1.00, die richtige Datei ist also immer unter den ersten drei. Latenz pro Suche
auf der GPU: rund 22 ms.

Was man daraus lernt:
- **BM25** scheitert an Umschreibungen, **Vektoren** an exakten Kennungen.
- **Hybrid** ist bei Kennungen perfekt, verliert aber bei Umschreibungen Plätze. Einfaches RRF gewichtet
  beide Listen gleich. Findet BM25 nur zufällige Wortübereinstimmungen in falschen Dateien, schieben
  diese den richtigen Vektor-Treffer auf Platz 2 oder 3.
- Hybrid bleibt der Standard, weil es nie komplett danebenliegt. Für Platz 1 ist der nächste Schritt ein
  Reranker (Phase 3), der die Top-Kandidaten noch einmal genau bewertet.

Einschränkung: 35 Fragen sind wenig, ein bis zwei Fragen verschieben die Werte um mehrere Prozentpunkte.
Selbst messen: `uv run python eval/run_eval.py`

## Idee

Die Windows-Suche findet Dateinamen und exakte Wörter. LocalRAG versteht den **Inhalt**:

```text
> localrag ask "Wie lange ist die Kündigungsfrist in meinem Mietvertrag?"

Die Kündigungsfrist beträgt drei Monate zum Monatsende [1].

[1] D:\Dokumente\Wohnung\Mietvertrag_2021.pdf – Seite 4, §5 Kündigung
```

## Geplante Features

- 🔒 **100 % lokal** – keine Daten verlassen den Rechner (LM Studio oder jeder andere OpenAI-kompatible Server + lokale Embeddings)
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
    Q[Frage] --> V & B --> R[RRF + Reranker] --> L[LLM via LM Studio] --> A[Antwort + Zitate]
```

## Tech-Stack

Python 3.12 · uv · PyTorch/CUDA · sentence-transformers · SQLite (FTS5 + sqlite-vec) · LM Studio (OpenAI-API) · httpx · FastAPI · Typer · MCP

## Lizenz

Siehe [LICENSE](LICENSE).
