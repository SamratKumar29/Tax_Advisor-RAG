"""
Build search indexes: ChromaDB (dense) + BM25 (sparse).

- Embeds all chunks with sentence-transformers (BAAI/bge-small-en-v1.5) locally.
- Writes vectors + metadata to a persisted ChromaDB collection.
- Writes the BM25 corpus (tokens + chunk lookup) as JSON; the BM25Okapi model is
  rebuilt from it at load time, so no pickle is ever loaded.
- Writes indexes/manifest.json (counts, embedding model, chunks.jsonl hash) so the
  app and tests can check that both indexes were built from the same chunks.
- Idempotent: safe to re-run whenever chunks.jsonl changes.
"""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import (
    BM25_INDEX_PATH,
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    CHUNKS_JSONL_PATH,
    EMBEDDING_MODEL,
    INDEX_MANIFEST_PATH,
    SECTION_HEADER_IN,
)
from src.ingest.chunk import index_text
from src.text_utils import tokenize

METADATA_FIELDS = ("source_doc", "source_type", "topic", "tax_year_scope", "section_ref",
                   "section_title")


# ── Load chunks ──────────────────────────────────────────────────────────

def load_chunks(path: Path = CHUNKS_JSONL_PATH) -> list[dict]:
    """Load all chunks from chunks.jsonl."""
    with open(path, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    print(f"  Loaded {len(chunks)} chunks from {path}")
    return chunks


def chunk_metadata(chunk: dict) -> dict:
    """Flat metadata stored with each chunk in both indexes (Chroma rejects None)."""
    meta = {k: chunk.get(k, "") for k in METADATA_FIELDS}
    if isinstance(chunk.get("page"), int):
        meta["page"] = chunk["page"]
    return meta


# ── ChromaDB (dense) index ───────────────────────────────────────────────

def build_chroma_index(chunks: list[dict]) -> int:
    """Embed all chunks and write to a persisted ChromaDB collection."""
    print(f"\n  Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["text"] for c in chunks]
    embed_texts = [index_text(c) if "dense" in SECTION_HEADER_IN else c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [chunk_metadata(c) for c in chunks]

    print(f"  Embedding {len(texts)} chunks (this may take a few minutes)...")
    embeddings = model.encode(embed_texts, show_progress_bar=True, batch_size=64)

    # Rebuild the persist dir from scratch: delete_collection() alone leaves the
    # old HNSW segment directory and un-vacuumed sqlite pages behind (index bloat).
    persist_dir = str(CHROMA_PERSIST_DIR)
    if Path(persist_dir).exists():
        shutil.rmtree(persist_dir)
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)

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

    count = collection.count()
    print(f"  ✅ ChromaDB: {count} vectors in '{CHROMA_COLLECTION_NAME}'")
    print(f"     Persisted to: {persist_dir}")
    return count


# ── BM25 (sparse) index ─────────────────────────────────────────────────

def build_bm25_corpus(chunks: list[dict]) -> dict:
    """Tokenized corpus + chunk lookup; BM25Okapi is fitted from this at load time."""
    return {
        "chunk_ids": [c["id"] for c in chunks],
        "tokenized_corpus": [
            tokenize(index_text(c) if "bm25" in SECTION_HEADER_IN else c["text"]) for c in chunks],
        "chunk_lookup": {
            c["id"]: {"text": c["text"], "metadata": chunk_metadata(c)} for c in chunks
        },
    }


def build_bm25_index(chunks: list[dict]) -> int:
    print(f"\n  Tokenizing {len(chunks)} chunks for BM25...")
    data = build_bm25_corpus(chunks)

    bm25_path = Path(BM25_INDEX_PATH)
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bm25_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    size_mb = bm25_path.stat().st_size / (1024 * 1024)
    print(f"  ✅ BM25: corpus for {len(data['chunk_ids'])} chunks")
    print(f"     Saved to: {bm25_path} ({size_mb:.1f} MB)")
    return len(data["chunk_ids"])


# ── Manifest ─────────────────────────────────────────────────────────────

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest(n_chunks: int, n_dense: int, n_bm25: int) -> dict:
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding_model": EMBEDDING_MODEL,
        "chunks_jsonl_sha256": file_sha256(Path(CHUNKS_JSONL_PATH)),
        "n_chunks": n_chunks,
        "n_dense": n_dense,
        "n_bm25": n_bm25,
        "section_header_in": sorted(SECTION_HEADER_IN),
    }
    path = Path(INDEX_MANIFEST_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"  ✅ Manifest: {path}")
    return manifest


# ── Main ─────────────────────────────────────────────────────────────────

def build_all_indexes():
    """Build both dense (ChromaDB) and sparse (BM25) indexes."""
    print("=" * 60)
    print("  Index Builder")
    print("=" * 60)

    chunks = load_chunks()
    n_dense = build_chroma_index(chunks)
    n_bm25 = build_bm25_index(chunks)
    write_manifest(len(chunks), n_dense, n_bm25)

    print(f"\n{'=' * 60}")
    print("  All indexes built successfully!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    build_all_indexes()
