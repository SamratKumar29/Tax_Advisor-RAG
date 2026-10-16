"""
Configuration module for Indian Tax Advisor RAG.

Design note: Embeddings run LOCALLY via sentence-transformers (no API key needed).
Only the generation LLM call goes through the configured OpenRouter/NIM endpoint.
This is a deliberate choice — keeps retrieval free, offline-capable, and not
dependent on which LLM provider is configured that day.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Project paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHUNKS_JSONL_PATH = DATA_PROCESSED_DIR / "chunks.jsonl"

# ── LLM (remote, via OpenRouter / NVIDIA NIM) ─────────────────────────────
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
# Comma-separated models tried in order after the primary model fails.
FALLBACK_MODELS = [m.strip() for m in os.getenv("FALLBACK_MODELS", "").split(",") if m.strip()]
MAX_QUERY_CHARS = int(os.getenv("MAX_QUERY_CHARS", "2000"))

# ── Embedding model (local, no API key) ────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# ── Index paths ────────────────────────────────────────────────────────────
def resolve_path(value: str | Path) -> Path:
    """Resolve a possibly-relative path against PROJECT_ROOT (not the cwd)."""
    p = Path(value).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


CHROMA_PERSIST_DIR = str(resolve_path(os.getenv("CHROMA_PERSIST_DIR", "indexes/chroma")))
# JSON (not pickle): the BM25 model is rebuilt from tokens at load time, so loading
# an index file never executes code.
_bm25_env = os.getenv("BM25_INDEX_PATH", "indexes/bm25.json")
if _bm25_env.endswith(".pkl"):  # legacy .env value from before the JSON index
    _bm25_env = _bm25_env[: -len(".pkl")] + ".json"
BM25_INDEX_PATH = str(resolve_path(_bm25_env))
INDEX_MANIFEST_PATH = str(resolve_path(os.getenv("INDEX_MANIFEST_PATH", "indexes/manifest.json")))


# User-uploaded documents overlay (git-ignored); see src.ingest.user_docs.
USER_DOCS_DIR = str(resolve_path(os.getenv("USER_DOCS_DIR", "data/user")))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "15"))
MAX_UPLOAD_PAGES = int(os.getenv("MAX_UPLOAD_PAGES", "500"))


class IndexesMissingError(RuntimeError):
    """Raised when the retrieval indexes have not been built."""


def check_indexes(chroma_dir: str | None = None, bm25_path: str | None = None) -> None:
    """Fail fast with an actionable message if the indexes are missing."""
    chroma = Path(chroma_dir or CHROMA_PERSIST_DIR)
    bm25 = Path(bm25_path or BM25_INDEX_PATH)
    missing = []
    if not chroma.is_dir() or not any(chroma.iterdir()):
        missing.append(f"Chroma index at {chroma}")
    if not bm25.is_file():
        missing.append(f"BM25 index at {bm25}")
    if missing:
        raise IndexesMissingError(
            "Missing " + " and ".join(missing)
            + ". Build them with: python -m src.ingest.build_index"
        )

# ── Retrieval parameters ──────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))       # characters (~375 tokens)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))  # characters
DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", "10"))       # ChromaDB retrieval
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "10"))         # BM25 retrieval
HYBRID_TOP_N = int(os.getenv("HYBRID_TOP_N", "5"))      # Final fused results
RRF_K = int(os.getenv("RRF_K", "60"))                   # RRF constant (paper default)
# Weighted RRF: BM25 (with query expansion) is the stronger ranker on the gold set
# (eval/results/n1_*); any weight in 1.5–3 gives the same recall@5.
RRF_DENSE_WEIGHT = float(os.getenv("RRF_DENSE_WEIGHT", "1.0"))
RRF_BM25_WEIGHT = float(os.getenv("RRF_BM25_WEIGHT", "1.5"))
# Instruction prepended to queries (not passages) before embedding. bge-v1.5
# recommends one for short-query retrieval; whether it helps here is measured
# in eval/ before changing the default.
DENSE_QUERY_PREFIX = os.getenv("DENSE_QUERY_PREFIX", "")
# "exact" (deterministic, default) or "hnsw" (approximate; for much larger corpora).
DENSE_BACKEND = os.getenv("DENSE_BACKEND", "exact")
# Which indexes see the "Section N — title" header in front of each chunk (see
# src.ingest.chunk.index_text). Comma-separated subset of "bm25,dense"; rebuild
# the indexes (make index) after changing it.
# Append statutory phrasing to BM25 queries ("slab" → "rate of tax"); see
# src.retrieval.query_expansion.
BM25_QUERY_EXPANSION = os.getenv("BM25_QUERY_EXPANSION", "1") not in ("0", "false", "False", "")
# Optional LLM rewrite of the question into statutory keywords for BM25 (one
# extra LLM call per question); see src.retrieval.llm_query.
LLM_QUERY_REWRITE = os.getenv("LLM_QUERY_REWRITE", "0") not in ("0", "false", "False", "")
# Abstain (no LLM call) when the best dense cosine similarity is below this.
# 0.68 abstains on 3/5 out-of-scope gold questions and 0/42 answerable ones
# (lowest answerable: 0.694 — a thin margin chosen on the same set). 0 disables.
ABSTAIN_DENSE_THRESHOLD = float(os.getenv("ABSTAIN_DENSE_THRESHOLD", "0.68"))
# Rewrite follow-up questions into standalone ones using the last turns (one extra
# LLM call per follow-up); see src.generation.memory.
CONVERSATION_MEMORY = os.getenv("CONVERSATION_MEMORY", "0") not in ("0", "false", "False", "")
MEMORY_TURNS = int(os.getenv("MEMORY_TURNS", "3"))
# Structured per-query log (JSONL; contains the user's questions — disable for shared deployments).
QUERY_LOG_PATH = os.getenv("QUERY_LOG_PATH", "logs/queries.jsonl")
SECTION_HEADER_IN = {p.strip() for p in os.getenv("SECTION_HEADER_IN", "").split(",") if p.strip()}

# ── ChromaDB collection name ──────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "tax_advisor_chunks"
