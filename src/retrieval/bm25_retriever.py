"""
BM25 sparse retriever: tokenizes a query and scores against the BM25 index.

The index file is plain JSON (tokens + chunk lookup); BM25Okapi is fitted from
it once per process. Loading never unpickles anything.
"""

import json

import numpy as np
from rank_bm25 import BM25Okapi

from src import config
from src.config import BM25_INDEX_PATH, BM25_TOP_K
from src.retrieval.query_expansion import expand_query
from src.text_utils import tokenize

# Module-level singleton
_bm25_data: dict | None = None


def _load_bm25() -> dict:
    global _bm25_data
    if _bm25_data is None:
        with open(BM25_INDEX_PATH, encoding="utf-8") as f:
            data = json.load(f)
        _merge_user_chunks(data)
        data["bm25"] = BM25Okapi(data["tokenized_corpus"])
        _bm25_data = data
    return _bm25_data


def _merge_user_chunks(data: dict) -> None:
    """Append user-uploaded chunks (src.ingest.user_docs) to the loaded base corpus."""
    from src.ingest.build_index import chunk_metadata
    from src.ingest.user_docs import load_user_chunks

    for c in load_user_chunks():
        if c["id"] in data["chunk_lookup"]:
            continue
        data["chunk_ids"].append(c["id"])
        data["tokenized_corpus"].append(tokenize(c["text"]))
        data["chunk_lookup"][c["id"]] = {"text": c["text"], "metadata": chunk_metadata(c)}


def search(query: str, top_k: int | None = None) -> list[dict]:
    """
    Tokenize the query and score against the BM25 index.

    Returns a list of dicts, each with:
        chunk_id, rank, score, text, metadata
    """
    k = top_k or BM25_TOP_K
    data = _load_bm25()
    chunk_ids = data["chunk_ids"]

    if config.BM25_QUERY_EXPANSION:
        query = expand_query(query)
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
