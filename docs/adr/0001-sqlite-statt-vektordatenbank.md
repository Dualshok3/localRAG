# ADR 0001: Eine SQLite-Datei statt separater Vektordatenbank

**Status:** Angenommen · **Datum:** 2026-11-17

## Kontext

LocalRAG braucht drei Arten von Daten:

1. Ein Manifest der indexierten Dateien (Pfad, Hash, Änderungsdatum)
2. Einen Volltext-Index für die Stichwortsuche (BM25)
3. Vektoren für die semantische Suche

Zur Wahl standen eine eigenständige Vektordatenbank (LanceDB, Qdrant, Chroma) zusätzlich zu SQLite
oder alles in SQLite mit FTS5 und der Erweiterung `sqlite-vec`.

Erwartete Größe auf einem privaten PC: einige 10.000 bis etwa 1 Mio. Chunks, ein Nutzer, kein Server.

## Entscheidung

Alles kommt in **eine SQLite-Datei**: Manifest, Chunks, `FTS5` für BM25 und `sqlite-vec` für die Vektoren.
Vektorzugriffe laufen ausschließlich über `index/vector_store.py`, damit das Backend austauschbar bleibt.

## Begründung

- **Reines BM25 reicht nicht:** Es findet keine Umschreibungen und keine anderen Sprachen. FTS5 hat außerdem
  kein deutsches Stemming. Genau diese Fälle sind der Kern des Projekts. Vektoren sind also nötig.
- **Eine separate Vektordatenbank ist bei dieser Größe nicht nötig:** Ein Brute-Force-Vergleich über einige
  100.000 Vektoren dauert Millisekunden, bei etwa 1 Mio. einige hundert Millisekunden auf der CPU.
  Mit Quantisierung (int8/binär) oder auf der GPU geht es schneller.
- **Konsistenz:** Ändert sich eine Datei, werden Text, BM25 und Vektoren in einer Transaktion aktualisiert.
  Zwei getrennte Speicher können dagegen auseinanderlaufen.
- **Einfachheit:** Kein Server, eine Datei, die sich leicht sichern, löschen oder neu aufbauen lässt.

## Konsequenzen

- (+) Weniger Abhängigkeiten, einfache Installation, leicht erklärbar
- (+) Hybrid-Suche (BM25 + Vektor + RRF) ohne Daten zwischen zwei Systemen zu verschieben
- (−) Kein ANN-Index (wie HNSW). Bei vielen Millionen Chunks wird die Suche langsamer
- (−) `sqlite-vec` ist noch vor Version 1.0

**Neu bewerten, wenn:** p95-Latenz der Vektorsuche > 300 ms oder mehr als ~5 Mio. Chunks.
Dann einen ANN-Index (LanceDB/Qdrant) hinter demselben Interface messen und vergleichen.
