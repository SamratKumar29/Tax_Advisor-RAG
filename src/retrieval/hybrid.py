"""
Hybrid retrieval with Reciprocal Rank Fusion (RRF).

Takes ranked results from both dense (ChromaDB) and sparse (BM25) retrievers,
fuses them using RRF, and returns a single deduplicated top-N list.

RRF formula: score(d) = sum( w_i / (k + rank_i(d)) ) for each ranker i
where k=60 is the standard constant from the original RRF paper
(Cormack, Clarke & Buettcher, 2009) and w_i is a per-ranker weight
(RRF_DENSE_WEIGHT, RRF_BM25_WEIGHT; 1.0 each is plain RRF).

This is ~20 lines of actual logic — simple enough to whiteboard and explain.
"""

from src.config import (
    BM25_TOP_K,
    DENSE_TOP_K,
    HYBRID_TOP_N,
    RRF_BM25_WEIGHT,
    RRF_DENSE_WEIGHT,
    RRF_K,
)
from src.retrieval import bm25_retriever, dense_retriever


def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    k: int = RRF_K,
    top_n: int = HYBRID_TOP_N,
    dense_weight: float = RRF_DENSE_WEIGHT,
    bm25_weight: float = RRF_BM25_WEIGHT,
) -> list[dict]:
    """
    Fuse two ranked lists using Reciprocal Rank Fusion.

    For each unique chunk, compute:
        rrf_score = sum(weight / (k + rank)) across whichever list(s) it appears in.

    Sort by rrf_score descending, return top-N.

    Args:
        dense_results: Ranked list from dense retriever.
        bm25_results: Ranked list from BM25 retriever.
        k: RRF constant (default 60, per the original paper).
        top_n: Number of final results to return.
        dense_weight, bm25_weight: Per-ranker weights.

    Returns:
        List of chunk dicts with rrf_score, text, and metadata.
    """
    # Accumulate RRF scores per chunk
    rrf_scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    for result in dense_results:
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + dense_weight / (k + result["rank"])
        chunk_data[cid] = result  # keep the full data

    for result in bm25_results:
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + bm25_weight / (k + result["rank"])
        if cid not in chunk_data:
            chunk_data[cid] = result

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    # Build final results
    fused = []
    for rank, cid in enumerate(sorted_ids[:top_n], start=1):
        data = chunk_data[cid]
        fused.append({
            "chunk_id": cid,
            "rank": rank,
            "rrf_score": rrf_scores[cid],
            "text": data["text"],
            "metadata": data["metadata"],
        })

    return fused


def search(query: str, top_n: int | None = None, bm25_extra: str = "") -> list[dict]:
    """Hybrid search results only; see search_with_confidence."""
    return search_with_confidence(query, top_n=top_n, bm25_extra=bm25_extra)[0]


def search_with_confidence(query: str, top_n: int | None = None,
                           bm25_extra: str = "") -> tuple[list[dict], float]:
    """
    Run hybrid search: dense + BM25 → RRF fusion → top-N results.

    This is the main entry point for retrieval in the generation chain.
    `bm25_extra` (e.g. LLM-rewritten statutory keywords) is appended to the
    BM25 query only; dense search always sees the user's question.
    """
    dense_results = dense_retriever.search(query, top_k=DENSE_TOP_K)
    bm25_query = f"{query} {bm25_extra}" if bm25_extra else query
    bm25_results = bm25_retriever.search(bm25_query, top_k=BM25_TOP_K)

    fused = reciprocal_rank_fusion(dense_results, bm25_results, top_n=top_n or HYBRID_TOP_N)
    # Best dense cosine similarity: the abstention signal (see config.ABSTAIN_DENSE_THRESHOLD).
    confidence = max((r["score"] for r in dense_results), default=0.0)
    return fused, confidence
