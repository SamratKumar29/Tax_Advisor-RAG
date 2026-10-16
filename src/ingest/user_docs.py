"""
User-added documents (books, notes, explainers) as an overlay on the base corpus.

The committed base corpus (data/processed/chunks.jsonl + indexes/) is never
modified. Uploaded files, their chunks (with embeddings) and a registry with the
licence the user declared live under USER_DOCS_DIR (data/user/, git-ignored, so
an uploaded copyrighted book can never be committed by accident). Both
retrievers merge the overlay at load time; reset_retriever_caches() makes an
add/delete visible immediately.

Pipeline per upload: validate (type, size, pages, licence) → save file →
chunk_document() → tag "supplementary" → embed → append to the overlay →
reload retrievers. Delete removes the file, its chunks and its vectors.
Supported with the default exact dense backend; the hnsw backend only sees the
base corpus.
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from src import config
from src.ingest.chunk import chunk_document

ALLOWED_SUFFIXES = {".pdf", ".md", ".txt"}
USER_PREFIX = "user/"


def _dir() -> Path:
    return Path(config.USER_DOCS_DIR)


def _files_dir() -> Path:
    return _dir() / "files"


def _chunks_path() -> Path:
    return _dir() / "chunks.jsonl"


def _registry_path() -> Path:
    return _dir() / "registry.json"


# ── Reading the overlay ───────────────────────────────────────────────────

def load_user_chunks() -> list[dict]:
    """Overlay chunks (each with an "embedding" list); [] when nothing was uploaded."""
    path = _chunks_path()
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def list_documents() -> list[dict]:
    path = _registry_path()
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _write_chunks(chunks: list[dict]) -> None:
    _dir().mkdir(parents=True, exist_ok=True)
    with open(_chunks_path(), "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def _write_registry(entries: list[dict]) -> None:
    _dir().mkdir(parents=True, exist_ok=True)
    _registry_path().write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8")


def reset_retriever_caches() -> None:
    """Drop the in-process index caches so the next query sees the current overlay."""
    from src.retrieval import bm25_retriever, dense_retriever

    bm25_retriever._bm25_data = None
    dense_retriever._matrix = None


# ── Validation ────────────────────────────────────────────────────────────

def safe_filename(name: str) -> str:
    """Basename only, restricted characters (no path traversal, no shell surprises)."""
    base = Path(name or "").name
    stem, suffix = Path(base).stem, Path(base).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "document"
    return f"{stem[:80]}{suffix}"


def validate_upload(filename: str, data: bytes, licence: str) -> str:
    """Return the safe filename, or raise ValueError with a user-facing message."""
    name = safe_filename(filename)
    if Path(name).suffix not in ALLOWED_SUFFIXES:
        raise ValueError(f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}.")
    if not data:
        raise ValueError("The file is empty.")
    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    if len(data) > max_bytes:
        raise ValueError(f"File is larger than {config.MAX_UPLOAD_MB} MB.")
    if not (licence or "").strip():
        raise ValueError("Please state the licence / permission under which you may use this document.")
    if any(d["filename"] == name for d in list_documents()):
        raise ValueError(f"A document named {name} already exists. Delete it first to replace it.")
    if Path(name).suffix == ".pdf":
        from io import BytesIO

        from pypdf import PdfReader
        try:
            n_pages = len(PdfReader(BytesIO(data)).pages)
        except Exception as e:  # pypdf raises many types for malformed files
            raise ValueError(f"Could not read the PDF ({type(e).__name__}).") from e
        if n_pages > config.MAX_UPLOAD_PAGES:
            raise ValueError(f"PDF has {n_pages} pages; the limit is {config.MAX_UPLOAD_PAGES}.")
    return name


# ── Add / delete ──────────────────────────────────────────────────────────

def _default_embed(texts: list[str]) -> list[list[float]]:
    from src.retrieval.dense_retriever import _get_model

    return _get_model().encode(texts, batch_size=64).tolist()


def add_document(filename: str, data: bytes, licence: str, embed=None,
                 base_ids: set[str] | None = None) -> dict:
    """
    Validate, chunk, embed and add one document to the overlay.

    Returns the registry entry. Raises ValueError for invalid uploads.
    """
    name = validate_upload(filename, data, licence)
    _files_dir().mkdir(parents=True, exist_ok=True)
    path = _files_dir() / name
    path.write_bytes(data)

    try:
        chunks = chunk_document(path)
    except Exception as e:
        path.unlink(missing_ok=True)
        raise ValueError(f"Could not extract text from the document ({type(e).__name__}).") from e
    source_doc = USER_PREFIX + name
    existing = load_user_chunks()
    taken = set(base_ids or ()) | {c["id"] for c in existing}
    new = []
    for c in chunks:
        c["source_doc"] = source_doc
        c["source_type"] = "supplementary"
        c["id"] = "u" + hashlib.md5(f"{source_doc}:{c['id']}".encode()).hexdigest()[:11]
        if c["id"] not in taken:
            taken.add(c["id"])
            new.append(c)
    if not new:
        path.unlink(missing_ok=True)
        raise ValueError("No text could be extracted (scanned PDFs without a text layer are not supported).")

    vectors = (embed or _default_embed)([c["text"] for c in new])
    for c, v in zip(new, vectors):
        c["embedding"] = [float(x) for x in v]
    _write_chunks(existing + new)

    entry = {
        "source_doc": source_doc,
        "filename": name,
        "licence": licence.strip(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "n_chunks": len(new),
        "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _write_registry(list_documents() + [entry])
    reset_retriever_caches()
    return entry


def delete_document(source_doc: str) -> int:
    """Remove a document's file, chunks and vectors. Returns the number of chunks removed."""
    entries = list_documents()
    entry = next((d for d in entries if d["source_doc"] == source_doc), None)
    if entry is None:
        raise ValueError(f"No uploaded document {source_doc}.")
    chunks = load_user_chunks()
    kept = [c for c in chunks if c["source_doc"] != source_doc]
    _write_chunks(kept)
    _write_registry([d for d in entries if d["source_doc"] != source_doc])
    (_files_dir() / entry["filename"]).unlink(missing_ok=True)
    reset_retriever_caches()
    return len(chunks) - len(kept)
