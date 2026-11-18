# Changelog

Format nach [Keep a Changelog](https://keepachangelog.com/de/1.1.0/), Versionen nach
[Semantic Versioning](https://semver.org/lang/de/).

## [Unreleased]

### Neu
- `localrag autostart --enable/--disable`: Dienst samt Watcher startet bei der Windows-Anmeldung unsichtbar.
- Sprachmodell in der Konfiguration: `llm_url` und `llm_model` (Umgebungsvariablen haben Vorrang).
- Quellcode als Format, nur mit `code = true`, aufgeteilt an Funktionen und Klassen.
- Messung der Antwortqualität (`eval/run_answer_eval.py`) mit Prüfmodell, Zitat-Prüfung und Fallen-Fragen.

### Geändert
- Von Werkzeugen erzeugte HTML-Dokumentation (Sphinx, Doxygen, Javadoc …) wird übersprungen, `steamapps` ist
  standardmäßig ausgeschlossen.
- Präsentationen zeigen „Folie N“ statt „Seite N“.

## [0.2.0] – 2026-11-18

Alltagstauglich: Filter, Konfiguration, automatische Aktualisierung, Web-Oberfläche, mehr Formate.

### Neu
- **Filter in der Anfrage:** `type:pdf`, `in:Rechnungen`, `-in:Backup`, `after:2024-03`, `before:2024-07`.
  Sie funktionieren in CLI, Web-Oberfläche, HTTP-API und MCP. Bei Filtern rechnet die Vektorsuche exakt
  in der passenden Teilmenge. Filter gehen nicht an das Sprachmodell.
- **Konfigurationsdatei** `~/.localrag/config.toml` (`LOCALRAG_CONFIG`) mit `folders`, `exclude`,
  `max_file_mb` und `db`. `localrag config` zeigt sie an, `localrag config --init` legt eine Vorlage an.
- **Ausschlussregeln:** Namen, Muster (`*backup*`) und Pfade. Dateien, die nachträglich ausgeschlossen
  werden, verschwinden beim nächsten Lauf aus dem Index.
- **Watcher:** `localrag watch` oder `localrag serve --watch` hält den Index aktuell. Beim Start wird einmal
  abgeglichen, danach werden Ereignisse 2 s gesammelt. Gelöschte und umbenannte Dateien und Ordner werden
  aus dem Index entfernt.
- **Web-Oberfläche** unter `http://127.0.0.1:8765`: Suche, Antworten mit klickbaren Zitaten, Quellen-Karten,
  „Öffnen“ und „Im Ordner zeigen“, PDFs im Browser an der gefundenen Seite. Helles und dunkles Farbschema,
  auch für schmale Fenster.
- **Formate:** Excel (`.xlsx`/`.xlsm`, je Tabellenblatt, Kopfzeile wird in langen Tabellen wiederholt), CSV,
  PowerPoint (`.pptx`, je Folie mit Tabellen und Notizen), HTML (gegliedert nach Überschriften),
  E-Mails (`.eml`, Outlook `.msg`) mit Kopfzeilen und Anhängen. Programmdateien, die ebenfalls auf `.msg`
  enden, und CSV-Datenexporte mit über 5.000 Zeilen werden übersprungen statt als Fehler gemeldet. Geprüft
  an allen Dateien auf C: und D:: 1.900 Dateien in den neuen Formaten, ein echter Fehler.
- API-Endpunkte `/stats`, `/file/{id}` und `/open`.

### Sicherheit
- Der Dienst lehnt Anfragen mit fremdem Host-Header ab (Schutz vor DNS-Rebinding). `/open` verlangt einen
  eigenen Header, den fremde Webseiten nicht senden können. Ausgeliefert werden nur indexierte Dateien.

### Geändert
- Datenbankverbindungen warten bis zu 30 s auf Sperren und schreiben beim Öffnen nichts mehr, wenn das
  Schema aktuell ist. So kommen sich Dienst, Watcher und CLI nicht mehr in die Quere.

## [0.1.0] – 2026-11-17

Erste veröffentlichte Version.

- Hybrid-Suche: BM25 (SQLite FTS5, deutsches Stemming) plus bge-m3-Vektoren (sqlite-vec), verbunden per
  Reciprocal Rank Fusion. Bei ganzen Sätzen zählt die Stichwortliste weniger.
- `localrag ask`: Antworten mit Zitaten über ein lokales Sprachmodell (LM Studio oder ein anderer
  OpenAI-kompatibler Server).
- Optionaler Cross-Encoder-Reranker (`--rerank`).
- `localrag serve`: Hintergrunddienst mit geladenem Modell und HTTP-API.
- `localrag mcp`: MCP-Server für Claude Desktop und Claude Code, begrenzbar auf Ordner mit `--allow`.
- Formate: PDF (OCR für Scans über die Windows-OCR), DOCX, Markdown, TXT (auch UTF-16).
- Inkrementelle Indexierung, Zusammenfassen nahezu gleicher Treffer.
- Skaliert auf ganze Laufwerke: Junctions werden übersprungen, Wort- und Datenlisten ausgelassen,
  Vektoren im Arbeitsspeicher.
- Eval-Set mit Messungen: Stichwort, Vektor, Hybrid und Reranker, dazu ein Vergleich mit der Windows-Suche.

[0.2.0]: https://github.com/Dualshok3/RAG/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Dualshok3/RAG/releases/tag/v0.1.0
