from localrag.index.text_norm import normalize, query_terms, stem


def test_plural_and_singular_share_stem():
    assert stem("Rechnungen") == stem("Rechnung") == stem("rechnung")


def test_umlauts_are_folded():
    assert stem("Kündigung") == stem("Kundigung")
    assert "a" in stem("Äpfel")[0]


def test_normalize_drops_punctuation_and_underscores():
    assert normalize("Mietvertrag_2021.pdf!") == "mietvertrag 2021 pdf"


def test_query_terms_remove_stopwords_and_duplicates():
    terms = query_terms("Wie lange ist die Kündigungsfrist in meinem Mietvertrag? Kündigungsfrist!")
    assert terms == [stem("lange"), stem("Kündigungsfrist"), stem("Mietvertrag")]


def test_query_of_only_stopwords_still_searches():
    assert query_terms("die und das") != []
