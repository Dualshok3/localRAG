"""Textnormalisierung für die Volltextsuche.

SQLite FTS5 kennt nur englisches Stemming. Deshalb wird Text *vor* dem Einfügen in Python
tokenisiert, kleingeschrieben, mit dem deutschen Snowball-Stemmer auf den Wortstamm reduziert und
von Akzenten befreit. Dieselbe Funktion wird auf Suchanfragen angewendet, damit Index und Anfrage
zueinander passen: „Rechnungen“, „Rechnung“ und „rechnung“ ergeben alle ``rechnung``.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator
from functools import lru_cache

import snowballstemmer

# Buchstaben und Ziffern, ohne Unterstrich (FTS5 behandelt "_" als Trenner).
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

_stemmer = snowballstemmer.stemmer("german")

# Nur für Anfragen: sehr häufige Wörter, die das Ranking verwässern.
STOPWORDS: frozenset[str] = frozenset(
    """
    a ab alle als also am an auch auf aus bei bin bis da damit dann das dass dem den der des die
    doch du ein eine einem einen einer eines er es für gibt hab habe hat hatte ich ihr im in ist
    ja kann man mein meine meinem meinen meiner mich mir mit muss nach nicht noch nur ob oder sein
    sich sie sind so über um und uns von vor war was welche welcher welches wenn wer wie wieviel
    wieviele wird wo zu zum zur
    the an and are as at be by for from has have how i in is it my of on or that this to was what
    when where which who will with
    """.split()
)


def iter_tokens(text: str) -> Iterator[tuple[str, int, int]]:
    """Liefert (Token, Start, Ende) für jedes Wort im Originaltext."""
    for m in _TOKEN_RE.finditer(text):
        yield m.group(0), m.start(), m.end()


def _strip_diacritics(word: str) -> str:
    decomposed = unicodedata.normalize("NFKD", word)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


@lru_cache(maxsize=200_000)
def stem(token: str) -> str:
    lowered = token.casefold()
    return _strip_diacritics(_stemmer.stemWord(lowered))


def normalize(text: str) -> str:
    """Text → durch Leerzeichen getrennte Wortstämme (für die FTS5-Spalte)."""
    return " ".join(stem(tok) for tok, _, _ in iter_tokens(text))


def query_terms(query: str) -> list[str]:
    """Suchanfrage → eindeutige Wortstämme ohne Stoppwörter, Reihenfolge bleibt erhalten."""
    terms: list[str] = []
    for tok, _, _ in iter_tokens(query):
        if tok.casefold() in STOPWORDS:
            continue
        s = stem(tok)
        if s and s not in terms:
            terms.append(s)
    if not terms:  # Anfrage bestand nur aus Stoppwörtern → trotzdem etwas suchen
        terms = list(dict.fromkeys(stem(tok) for tok, _, _ in iter_tokens(query)))
    return terms
