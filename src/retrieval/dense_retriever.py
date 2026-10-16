"""
Dense retriever: embeds a query and returns the top-k most similar chunks.

Two backends (DENSE_BACKEND):
- "exact" (default): cosine similarity against every stored vector, loaded once
  from the Chroma collection. Deterministic. At this corpus size (~2.7k x 384
  floats ≈ 4 MB) a full matrix-vector product is well under a millisecond.
- "hnsw": Chroma's approximate HNSW query. Measured on this corpus it returned
  different top-10s between process runs (7 of 42 eval questions) and differed
  from exact search on 37 of 42, so it is kept only for large corpora where
  exact search becomes too slow.
"""

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    DENSE_BACKEND,
    DENSE_QUERY_PREFIX,
    DENSE_TOP_K,
    EMBEDDING_MODEL,
)

# Module-level singletons (loaded once, reused across queries)
_model: SentenceTransformer | None = None
_collection = None
_matrix: dict | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        _collection = client.get_collection(CHROMA_COLLECTION_NAME)
    return _collection


def _get_matrix() -> dict:
    """All stored vectors (L2-normalised), ordered by chunk id for stable tie-breaking."""
    global _matrix
    if _matrix is None:
        data = _get_collection().get(include=["embeddings", "documents", "metadatas"])
        order = sorted(range(len(data["ids"])), key=lambda i: data["ids"][i])
        emb = np.asarray([data["embeddings"][i] for i in order], dtype=np.float64)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        _matrix = {
            "ids": [data["ids"][i] for i in order],
            "docs": [data["documents"][i] for i in order],
            "metas": [data["metadatas"][i] for i in order],
            "emb": emb,
        }
    return _matrix


def _embed_query(query: str) -> np.ndarray:
    # Optional bge query instruction; passages are embedded without it.
    return np.asarray(_get_model().encode(DENSE_QUERY_PREFIX + query), dtype=np.float64)


def exact_search(query: str, k: int) -> list[dict]:
    m = _get_matrix()
    q = _embed_query(query)
    sims = m["emb"] @ (q / np.linalg.norm(q))
    k = min(k, len(sims))
    if k == 0:
        return []
    top = np.argpartition(-sims, k - 1)[:k]
    top = sorted(top, key=lambda i: (-round(float(sims[i]), 6), m["ids"][i]))
    return [
        {"chunk_id": m["ids"][i], "rank": r, "score": float(sims[i]),
         "text": m["docs"][i], "metadata": m["metas"][i]}
        for r, i in enumerate(top, start=1)
    ]


def hnsw_search(query: str, k: int) -> list[dict]:
    results = _get_collection().query(
        query_embeddings=[_embed_query(query).tolist()],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "chunk_id": results["ids"][0][i],
            "rank": i + 1,
            "score": 1 - results["distances"][0][i],  # cosine distance → similarity
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
        }
        for i in range(len(results["ids"][0]))
    ]


def search(query: str, top_k: int | None = None, backend: str | None = None) -> list[dict]:
    """
    Embed the query and return the top-k most similar chunks.

    Returns a list of dicts, each with:
        chunk_id, rank, score, text, metadata
    """
    k = top_k or DENSE_TOP_K
    if (backend or DENSE_BACKEND) == "hnsw":
        return hnsw_search(query, k)
    return exact_search(query, k)
