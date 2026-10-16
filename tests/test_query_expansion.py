from src.retrieval.query_expansion import expand_query
from src.text_utils import tokenize


def test_slab_query_gets_statutory_phrasing():
    q = "What are the new tax regime slab rates?"
    out = expand_query(q)
    assert out.startswith(q)
    assert "rate of tax" in out and "section 202" in out


def test_abbreviations_expand_case_insensitively():
    assert "deduction of tax at source" in expand_query("TDS on rent")
    assert "house rent allowance" in expand_query("hra exemption")
    assert "fee for default in furnishing return" in expand_query("fee for late filing")


def test_no_rule_leaves_query_unchanged():
    assert expand_query("Section 80C limit") == "Section 80C limit"


def test_word_boundaries_respected():
    # "fy" must not fire inside other words, "salary" must not fire on "salaryman"-like tokens
    assert expand_query("notify") == "notify"
    assert "salaries" not in expand_query("salaryman")


def test_each_expansion_added_once():
    out = expand_query("salary and salaried employees")
    assert out.count("salaries") == 1


def test_bm25_search_applies_expansion(monkeypatch):
    from src import config
    from src.ingest.build_index import build_bm25_corpus
    from src.retrieval import bm25_retriever

    chunks = [
        {"id": "rates", "text": "Total income Rate of tax Upto ₹400000 Nil", "source_doc": "act.pdf"},
        {"id": "other", "text": "House property and municipal taxes", "source_doc": "act.pdf"},
    ]
    data = build_bm25_corpus(chunks)
    from rank_bm25 import BM25Okapi
    data["bm25"] = BM25Okapi(data["tokenized_corpus"])
    monkeypatch.setattr(bm25_retriever, "_bm25_data", data)

    monkeypatch.setattr(config, "BM25_QUERY_EXPANSION", False)
    assert all(r["score"] == 0 for r in bm25_retriever.search("slab", top_k=2))
    monkeypatch.setattr(config, "BM25_QUERY_EXPANSION", True)
    assert bm25_retriever.search("slab", top_k=1)[0]["chunk_id"] == "rates"


def test_tokenizer_drops_stopwords():
    assert tokenize("What are the rates of tax for a resident?") == ["rates", "tax", "resident"]


def test_llm_rewrite_returns_keywords_and_caches():
    from src.retrieval import llm_query
    calls = []

    def llm(messages, **kwargs):
        calls.append(messages)
        return {"content": "  rate of tax   total income\n section 202 "}

    llm_query._cache.clear()
    assert llm_query.rewrite_query("slab rates?", "k", "u", llm=llm) == "rate of tax total income section 202"
    assert llm_query.rewrite_query("slab rates?", "k", "u", llm=llm) == "rate of tax total income section 202"
    assert len(calls) == 1


def test_llm_rewrite_failure_returns_empty():
    from src.llm_client import LLMUnavailableError
    from src.retrieval import llm_query

    def down(*a, **k):
        raise LLMUnavailableError(["m"], ["timeout"])

    llm_query._cache.clear()
    assert llm_query.rewrite_query("anything", "k", "u", llm=down) == ""
