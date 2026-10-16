"""
BM25 sparse retriever: tokenizes a query and scores against the BM25 index.

The index file is plain JSON (tokens + chunk lookup); BM25Okapi is fitted from
it once per process. Loading never unpickles anything.
"""

import json

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import BM25_INDEX_PATH, BM25_TOP_K
from src.text_utils import tokenize

# Module-level singleton
_bm25_data: dict | None = None


def _load_bm25() -> dict:
    global _bm25_data
    if _bm25_data is None:
        with open(BM25_INDEX_PATH, encoding="utf-8") as f:
            data = json.load(f)
        data["bm25"] = BM25Okapi(data["tokenized_corpus"])
        _bm25_data = data
    return _bm25_data


def search(query: str, top_k: int | None = None) -> list[dict]:
    """
    Tokenize the query and score against the BM25 index.

    Returns a list of dicts, each with:
        chunk_id, rank, score, text, metadata
    """
    k = top_k or BM25_TOP_K
    data = _load_bm25()
    chunk_ids = data["chunk_ids"]

    scores = np.asarray(data["bm25"].get_scores(tokenize(query)))
    k = min(k, len(scores))
    if k == 0:
        return []
    # argpartition for the top-k, then a stable sort (ties broken by index)
    top = np.argpartition(-scores, k - 1)[:k]
    top = sorted(top, key=lambda i: (-scores[i], i))

    ranked = []
    for rank, idx in enumerate(top, start=1):
        cid = chunk_ids[idx]
        info = data["chunk_lookup"][cid]
        ranked.append({
            "chunk_id": cid,
            "rank": rank,
            "score": float(scores[idx]),
            "text": info["text"],
            "metadata": info["metadata"],
        })
    return ranked
