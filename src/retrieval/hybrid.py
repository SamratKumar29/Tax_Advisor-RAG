"""
Hybrid retrieval with Reciprocal Rank Fusion (RRF).

Takes ranked results from both dense (ChromaDB) and sparse (BM25) retrievers,
fuses them using RRF, and returns a single deduplicated top-N list.

RRF formula: score(d) = sum( 1 / (k + rank_i(d)) ) for each ranker i
where k=60 is the standard constant from the original RRF paper
(Cormack, Clarke & Buettcher, 2009).

This is ~20 lines of actual logic — simple enough to whiteboard and explain.
"""

from src.config import DENSE_TOP_K, BM25_TOP_K, HYBRID_TOP_N, RRF_K
from src.retrieval import dense_retriever, bm25_retriever


def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    k: int = RRF_K,
    top_n: int = HYBRID_TOP_N,
) -> list[dict]:
    """
    Fuse two ranked lists using Reciprocal Rank Fusion.

    For each unique chunk, compute:
        rrf_score = sum(1 / (k + rank)) across whichever list(s) it appears in.

    Sort by rrf_score descending, return top-N.

    Args:
        dense_results: Ranked list from dense retriever.
        bm25_results: Ranked list from BM25 retriever.
        k: RRF constant (default 60, per the original paper).
        top_n: Number of final results to return.

    Returns:
        List of chunk dicts with rrf_score, text, and metadata.
    """
    # Accumulate RRF scores per chunk
    rrf_scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    for result in dense_results:
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + result["rank"])
        chunk_data[cid] = result  # keep the full data

    for result in bm25_results:
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + result["rank"])
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


def search(query: str, top_n: int | None = None) -> list[dict]:
    """
    Run hybrid search: dense + BM25 → RRF fusion → top-N results.

    This is the main entry point for retrieval in the generation chain.
    """
    dense_results = dense_retriever.search(query, top_k=DENSE_TOP_K)
    bm25_results = bm25_retriever.search(query, top_k=BM25_TOP_K)

    return reciprocal_rank_fusion(
        dense_results, bm25_results, top_n=top_n or HYBRID_TOP_N
    )
