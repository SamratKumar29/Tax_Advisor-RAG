"""
Integrity checks on the committed index artifacts: both indexes must be built
from the current chunks.jsonl and contain exactly the same chunk ids.
Skipped automatically when the indexes have not been built.
"""

import json
from pathlib import Path

import pytest

from src.config import (
    BM25_INDEX_PATH,
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    CHUNKS_JSONL_PATH,
    INDEX_MANIFEST_PATH,
)
from src.ingest.build_index import file_sha256

pytestmark = pytest.mark.skipif(
    not Path(INDEX_MANIFEST_PATH).exists(), reason="indexes not built"
)


@pytest.fixture(scope="module")
def chunk_ids():
    with open(CHUNKS_JSONL_PATH, encoding="utf-8") as f:
        return [json.loads(line)["id"] for line in f if line.strip()]


def test_manifest_matches_chunks(chunk_ids):
    manifest = json.loads(Path(INDEX_MANIFEST_PATH).read_text(encoding="utf-8"))
    assert manifest["chunks_jsonl_sha256"] == file_sha256(Path(CHUNKS_JSONL_PATH))
    assert manifest["n_chunks"] == manifest["n_dense"] == manifest["n_bm25"] == len(chunk_ids)


def test_chunk_ids_unique(chunk_ids):
    assert len(chunk_ids) == len(set(chunk_ids))


def test_bm25_ids_match_chunks(chunk_ids):
    with open(BM25_INDEX_PATH, encoding="utf-8") as f:
        data = json.load(f)
    assert data["chunk_ids"] == chunk_ids
    assert set(data["chunk_lookup"]) == set(chunk_ids)


def test_chroma_ids_match_chunks(chunk_ids):
    chromadb = pytest.importorskip("chromadb")
    col = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR).get_collection(CHROMA_COLLECTION_NAME)
    assert set(col.get(include=[])["ids"]) == set(chunk_ids)


def test_dense_exact_search_is_deterministic_and_well_formed():
    pytest.importorskip("sentence_transformers")
    from src.retrieval import dense_retriever

    q = "deduction for life insurance premium"
    a = dense_retriever.search(q, top_k=10, backend="exact")
    b = dense_retriever.search(q, top_k=10, backend="exact")
    assert [r["chunk_id"] for r in a] == [r["chunk_id"] for r in b]
    assert [r["rank"] for r in a] == list(range(1, 11))
    scores = [r["score"] for r in a]
    assert scores == sorted(scores, reverse=True)
    assert all("section_ref" in r["metadata"] for r in a)
