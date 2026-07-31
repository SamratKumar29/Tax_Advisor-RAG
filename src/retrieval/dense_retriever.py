"""
Dense retriever: embeds a query and searches ChromaDB for top-k results.
"""

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    DENSE_TOP_K,
)

# Module-level singletons (loaded once, reused across queries)
_model: SentenceTransformer | None = None
_collection = None


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


def search(query: str, top_k: int | None = None) -> list[dict]:
    """
    Embed the query and search ChromaDB for the top-k most similar chunks.

    Returns a list of dicts, each with:
        chunk_id, rank, score, text, metadata
    """
    k = top_k or DENSE_TOP_K
    model = _get_model()
    collection = _get_collection()

    # Embed query
    query_embedding = model.encode(query).tolist()

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    # Package results
    ranked = []
    for i in range(len(results["ids"][0])):
        ranked.append({
            "chunk_id": results["ids"][0][i],
            "rank": i + 1,
            "score": 1 - results["distances"][0][i],  # cosine distance → similarity
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
        })

    return ranked
