"""
BM25 sparse retriever: tokenizes a query and scores against the BM25 index.
"""

import pickle
import re
from pathlib import Path

from src.config import BM25_INDEX_PATH, BM25_TOP_K

# Module-level singleton
_bm25_data: dict | None = None


def _load_bm25():
    global _bm25_data
    if _bm25_data is None:
        with open(BM25_INDEX_PATH, "rb") as f:
            _bm25_data = pickle.load(f)
    return _bm25_data


def simple_tokenize(text: str) -> list[str]:
    """Same tokenizer used at index time — must match exactly."""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9₹]+(?:[a-z0-9₹,./\-]*[a-z0-9₹])?", text)
    return [t for t in tokens if len(t) > 1]


def search(query: str, top_k: int | None = None) -> list[dict]:
    """
    Tokenize the query and score against the BM25 index.

    Returns a list of dicts, each with:
        chunk_id, rank, score, text, metadata
    """
    k = top_k or BM25_TOP_K
    data = _load_bm25()

    bm25 = data["bm25"]
    chunk_ids = data["chunk_ids"]
    chunk_lookup = data["chunk_lookup"]

    query_tokens = simple_tokenize(query)
    scores = bm25.get_scores(query_tokens)

    # Get top-k indices by score (descending)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    ranked = []
    for rank, idx in enumerate(top_indices, start=1):
        cid = chunk_ids[idx]
        info = chunk_lookup[cid]
        ranked.append({
            "chunk_id": cid,
            "rank": rank,
            "score": float(scores[idx]),
            "text": info["text"],
            "metadata": {
                "source_doc": info["source_doc"],
                "source_type": info["source_type"],
                "topic": info["topic"],
                "tax_year_scope": info["tax_year_scope"],
                "section_ref": info["section_ref"],
            },
        })

    return ranked
