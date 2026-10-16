"""Upload → retrievable → delete → gone, on a tiny in-memory base corpus (offline)."""

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from src import config
from src.ingest import user_docs
from src.ingest.build_index import build_bm25_corpus, chunk_metadata
from src.retrieval import bm25_retriever, dense_retriever
from tests.pdf_fixture import make_pdf

BOOK_LINES = [
    "Chapter 7: Agniveer Corpus Fund",
    "Contributions to the Agniveer Corpus Fund are fully deductible for the employee.",
    "The payout on completion of four years of service is exempt from tax.",
]
VOCAB = ["agniveer", "corpus", "house", "property", "deduction", "rent"]


def fake_embed(texts):
    """Bag-of-words vectors over a small vocabulary: deterministic, no model download."""
    return [[t.lower().count(w) + 0.01 for w in VOCAB] for t in texts]


@pytest.fixture
def overlay(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "USER_DOCS_DIR", str(tmp_path / "user"))
    base = [
        {"id": "b1", "text": "Income from house property is computed on annual value.",
         "source_doc": "act.pdf", "source_type": "primary", "topic": "house_property",
         "tax_year_scope": "", "section_ref": "Section 20"},
        {"id": "b2", "text": "Deduction for rent paid by an individual.",
         "source_doc": "act.pdf", "source_type": "primary", "topic": "deductions",
         "tax_year_scope": "", "section_ref": "Section 134"},
    ]
    corpus = build_bm25_corpus(base)

    def load_bm25():
        if bm25_retriever._bm25_data is None:
            data = {k: (list(v) if isinstance(v, list) else dict(v)) for k, v in corpus.items()}
            bm25_retriever._merge_user_chunks(data)
            data["bm25"] = BM25Okapi(data["tokenized_corpus"])
            bm25_retriever._bm25_data = data
        return bm25_retriever._bm25_data

    def get_matrix():
        if dense_retriever._matrix is None:
            data = {"ids": [c["id"] for c in base], "embeddings": fake_embed([c["text"] for c in base]),
                    "documents": [c["text"] for c in base], "metadatas": [chunk_metadata(c) for c in base]}
            dense_retriever._merge_user_chunks(data)
            emb = np.asarray(data["embeddings"], dtype=np.float64)
            emb /= np.linalg.norm(emb, axis=1, keepdims=True)
            dense_retriever._matrix = {"ids": data["ids"], "docs": data["documents"],
                                       "metas": data["metadatas"], "emb": emb}
        return dense_retriever._matrix

    monkeypatch.setattr(bm25_retriever, "_load_bm25", load_bm25)
    monkeypatch.setattr(dense_retriever, "_get_matrix", get_matrix)
    monkeypatch.setattr(dense_retriever, "_embed_query", lambda q: np.asarray(fake_embed([q])[0]))
    user_docs.reset_retriever_caches()
    yield
    user_docs.reset_retriever_caches()


def _ids(results):
    return [r["chunk_id"] for r in results]


def test_upload_is_retrieved_then_delete_removes_it_from_both_indexes(overlay):
    entry = user_docs.add_document("My Tax Book.pdf", make_pdf(BOOK_LINES), "CC-BY 4.0",
                                   embed=fake_embed)
    assert entry["source_doc"] == "user/My_Tax_Book.pdf" and entry["n_chunks"] >= 1
    ids = {c["id"] for c in user_docs.load_user_chunks()}

    bm25 = bm25_retriever.search("Agniveer corpus fund", top_k=1)
    dense = dense_retriever.search("Agniveer corpus fund", top_k=1, backend="exact")
    assert bm25[0]["chunk_id"] in ids and dense[0]["chunk_id"] in ids
    assert bm25[0]["metadata"]["source_type"] == "supplementary"
    assert "Agniveer" in bm25[0]["text"]

    assert user_docs.delete_document(entry["source_doc"]) == entry["n_chunks"]
    assert not ids & set(_ids(bm25_retriever.search("Agniveer corpus fund", top_k=10)))
    assert not ids & set(_ids(dense_retriever.search("Agniveer corpus fund", top_k=10, backend="exact")))
    assert user_docs.list_documents() == []
    assert not (user_docs._files_dir() / entry["filename"]).exists()


def test_markdown_upload_and_registry_records_licence(overlay):
    entry = user_docs.add_document("notes.md", "# Notes\n\nAgniveer corpus details here.".encode(),
                                   "Own notes", embed=fake_embed)
    assert user_docs.list_documents() == [entry]
    assert entry["licence"] == "Own notes" and len(entry["sha256"]) == 64


@pytest.mark.parametrize("name,data,licence,msg", [
    ("book.exe", b"x", "ok", "Unsupported"),
    ("book.pdf", b"", "ok", "empty"),
    ("book.pdf", b"not a pdf", "ok", "Could not read"),
    ("book.md", b"text", "  ", "licence"),
])
def test_invalid_uploads_rejected(overlay, name, data, licence, msg):
    with pytest.raises(ValueError, match=msg):
        user_docs.add_document(name, data, licence, embed=fake_embed)
    assert user_docs.list_documents() == []


def test_size_and_page_limits(overlay, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0)
    with pytest.raises(ValueError, match="larger than"):
        user_docs.add_document("a.md", b"some text", "ok", embed=fake_embed)
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 15)
    monkeypatch.setattr(config, "MAX_UPLOAD_PAGES", 0)
    with pytest.raises(ValueError, match="pages"):
        user_docs.add_document("a.pdf", make_pdf(BOOK_LINES), "ok", embed=fake_embed)


def test_duplicate_name_rejected(overlay):
    user_docs.add_document("a.md", b"Agniveer corpus fund text", "ok", embed=fake_embed)
    with pytest.raises(ValueError, match="already exists"):
        user_docs.add_document("a.md", b"other text entirely", "ok", embed=fake_embed)


def test_safe_filename_blocks_traversal():
    assert user_docs.safe_filename("../../etc/passwd.md") == "passwd.md"
    assert user_docs.safe_filename("my book (2026).PDF") == "my_book_2026.pdf"


def test_delete_unknown_document(overlay):
    with pytest.raises(ValueError):
        user_docs.delete_document("user/missing.pdf")


def test_documents_page_lists_and_deletes(overlay):
    from pathlib import Path

    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    user_docs.add_document("notes.md", b"Agniveer corpus fund notes", "Own notes", embed=fake_embed)
    page = str(Path(__file__).resolve().parent.parent / "app" / "pages" / "1_Documents.py")
    at = AppTest.from_file(page, default_timeout=30).run()
    assert not at.exception
    assert any("notes.md" in m.value for m in at.markdown)
    at.button(key="delete_user/notes.md").click().run()
    assert user_docs.list_documents() == []
    assert user_docs.load_user_chunks() == []
