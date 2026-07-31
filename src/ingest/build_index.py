"""
Build search indexes: ChromaDB (dense) + BM25 (sparse).

- Embeds all chunks with sentence-transformers (BAAI/bge-small-en-v1.5) locally.
- Writes vectors + metadata to a persisted ChromaDB collection.
- Builds a BM25Okapi index and pickles it alongside chunk ID mapping.
- Idempotent: safe to re-run whenever data/raw/ changes.
"""

import json
import pickle
import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from src.config import (
    CHUNKS_JSONL_PATH,
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    BM25_INDEX_PATH,
    CHROMA_COLLECTION_NAME,
)


# ── Load chunks ──────────────────────────────────────────────────────────

def load_chunks() -> list[dict]:
    """Load all chunks from chunks.jsonl."""
    chunks = []
    with open(CHUNKS_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"  Loaded {len(chunks)} chunks from {CHUNKS_JSONL_PATH}")
    return chunks


# ── ChromaDB (dense) index ───────────────────────────────────────────────

def build_chroma_index(chunks: list[dict]):
    """
    Embed all chunks and write to a persisted ChromaDB collection.

    Uses BAAI/bge-small-en-v1.5 locally — no API key needed for embeddings.
    """
    print(f"\n  Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Extract texts for embedding
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]

    print(f"  Embedding {len(texts)} chunks (this may take a few minutes)...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    # Prepare metadata (ChromaDB stores metadata as flat dicts)
    metadatas = []
    for c in chunks:
        metadatas.append({
            "source_doc": c["source_doc"],
            "source_type": c["source_type"],
            "topic": c["topic"],
            "tax_year_scope": c["tax_year_scope"],
            "section_ref": c["section_ref"],
        })

    # Create/recreate the ChromaDB collection
    persist_dir = str(CHROMA_PERSIST_DIR)
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=persist_dir)

    # Delete existing collection if it exists (idempotent)
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
        print(f"  Deleted existing collection '{CHROMA_COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # ChromaDB has a batch size limit; insert in batches
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        batch_end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:batch_end],
            embeddings=embeddings[i:batch_end].tolist(),
            documents=texts[i:batch_end],
            metadatas=metadatas[i:batch_end],
        )

    print(f"  ✅ ChromaDB: {collection.count()} vectors in '{CHROMA_COLLECTION_NAME}'")
    print(f"     Persisted to: {persist_dir}")


# ── BM25 (sparse) index ─────────────────────────────────────────────────

def simple_tokenize(text: str) -> list[str]:
    """
    Simple whitespace + punctuation tokenizer for BM25.
    Lowercases, strips non-alphanumeric, removes short tokens.
    """
    text = text.lower()
    tokens = re.findall(r"[a-z0-9₹]+(?:[a-z0-9₹,./\-]*[a-z0-9₹])?", text)
    return [t for t in tokens if len(t) > 1]


def build_bm25_index(chunks: list[dict]):
    """
    Build a BM25Okapi index over the tokenized chunk texts.

    Pickles the BM25 model + a parallel list of chunk IDs + chunk metadata
    (since BM25 doesn't store metadata itself).
    """
    print(f"\n  Tokenizing {len(chunks)} chunks for BM25...")

    tokenized_corpus = []
    chunk_ids = []
    chunk_lookup = {}  # id -> {text, metadata}

    for c in chunks:
        tokens = simple_tokenize(c["text"])
        tokenized_corpus.append(tokens)
        chunk_ids.append(c["id"])
        chunk_lookup[c["id"]] = {
            "text": c["text"],
            "source_doc": c["source_doc"],
            "source_type": c["source_type"],
            "topic": c["topic"],
            "tax_year_scope": c["tax_year_scope"],
            "section_ref": c["section_ref"],
        }

    bm25 = BM25Okapi(tokenized_corpus)

    # Save BM25 index + metadata
    bm25_path = Path(BM25_INDEX_PATH)
    bm25_path.parent.mkdir(parents=True, exist_ok=True)

    bm25_data = {
        "bm25": bm25,
        "chunk_ids": chunk_ids,
        "chunk_lookup": chunk_lookup,
        "tokenized_corpus": tokenized_corpus,
    }

    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_data, f)

    size_mb = bm25_path.stat().st_size / (1024 * 1024)
    print(f"  ✅ BM25: index built over {len(chunk_ids)} chunks")
    print(f"     Saved to: {bm25_path} ({size_mb:.1f} MB)")


# ── Main ─────────────────────────────────────────────────────────────────

def build_all_indexes():
    """Build both dense (ChromaDB) and sparse (BM25) indexes."""
    print("=" * 60)
    print("  Index Builder")
    print("=" * 60)

    chunks = load_chunks()
    build_chroma_index(chunks)
    build_bm25_index(chunks)

    print(f"\n{'=' * 60}")
    print("  All indexes built successfully!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    build_all_indexes()
