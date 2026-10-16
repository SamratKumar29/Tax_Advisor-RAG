"""Tests for RRF fusion and the BM25 retriever on a small in-memory corpus."""

import pytest

from src.ingest.build_index import build_bm25_corpus
from src.retrieval import bm25_retriever
from src.retrieval.hybrid import reciprocal_rank_fusion


def _r(cid, rank):
    return {"chunk_id": cid, "rank": rank, "text": cid, "metadata": {"source_doc": "d"}}


def test_rrf_rewards_agreement():
    dense = [_r("a", 1), _r("b", 2), _r("c", 3)]
    bm25 = [_r("b", 1), _r("d", 2), _r("a", 3)]
    fused = reciprocal_rank_fusion(dense, bm25, k=60, top_n=4, dense_weight=1, bm25_weight=1)
    ids = [f["chunk_id"] for f in fused]
    # b: 1/62 + 1/61 ; a: 1/61 + 1/63 → b first, then a; c and d only appear once
    assert ids[:2] == ["b", "a"]
    assert set(ids) == {"a", "b", "c", "d"}
    assert [f["rank"] for f in fused] == [1, 2, 3, 4]


def test_rrf_deduplicates_and_truncates():
    dense = [_r("a", 1), _r("b", 2)]
    bm25 = [_r("a", 1), _r("b", 2)]
    fused = reciprocal_rank_fusion(dense, bm25, top_n=1, dense_weight=1, bm25_weight=1)
    assert len(fused) == 1 and fused[0]["chunk_id"] == "a"
    assert fused[0]["rrf_score"] == pytest.approx(2 / 61)


def test_rrf_weights_favour_the_heavier_ranker():
    dense = [_r("a", 1), _r("b", 2)]
    bm25 = [_r("b", 1), _r("a", 2)]
    equal = reciprocal_rank_fusion(dense, bm25, top_n=2, dense_weight=1, bm25_weight=1)
    assert equal[0]["rrf_score"] == pytest.approx(equal[1]["rrf_score"])
    fused = reciprocal_rank_fusion(dense, bm25, top_n=2, dense_weight=1, bm25_weight=1.5)
    assert [f["chunk_id"] for f in fused] == ["b", "a"]
    assert fused[0]["rrf_score"] == pytest.approx(1 / 62 + 1.5 / 61)


def test_rrf_empty_inputs():
    assert reciprocal_rank_fusion([], [], top_n=5) == []


@pytest.fixture
def tiny_bm25(monkeypatch):
    chunks = [
        {"id": "c1", "text": "Deduction under section 123 limited to ₹1,50,000.",
         "source_doc": "act.pdf", "source_type": "primary", "topic": "deductions_80c",
         "tax_year_scope": "FY2026-27_AY2027-28", "section_ref": "Section 123"},
        {"id": "c2", "text": "House rent allowance exemption for metro city.",
         "source_doc": "act.pdf", "source_type": "primary", "topic": "hra",
         "tax_year_scope": "FY2026-27_AY2027-28", "section_ref": "Section 19"},
        {"id": "c3", "text": "Tax deducted at source on salary payments.",
         "source_doc": "act.pdf", "source_type": "primary", "topic": "tds",
         "tax_year_scope": "FY2026-27_AY2027-28", "section_ref": "Section 392"},
    ]
    data = build_bm25_corpus(chunks)
    from rank_bm25 import BM25Okapi
    data["bm25"] = BM25Okapi(data["tokenized_corpus"])
    monkeypatch.setattr(bm25_retriever, "_bm25_data", data)
    return data


def test_bm25_matches_amount_in_any_format(tiny_bm25):
    for q in ["limit of Rs 1,50,000", "150000 deduction", "₹1,50,000"]:
        assert bm25_retriever.search(q, top_k=1)[0]["chunk_id"] == "c1"


def test_bm25_result_shape_and_metadata(tiny_bm25):
    res = bm25_retriever.search("house rent allowance", top_k=2)
    assert res[0]["chunk_id"] == "c2" and res[0]["rank"] == 1
    assert res[0]["metadata"]["section_ref"] == "Section 19"
    assert set(res[0]) == {"chunk_id", "rank", "score", "text", "metadata"}


def test_bm25_top_k_larger_than_corpus(tiny_bm25):
    assert len(bm25_retriever.search("salary", top_k=50)) == 3
