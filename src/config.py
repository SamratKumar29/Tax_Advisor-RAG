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
load_dotenv()

# ── Project paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHUNKS_JSONL_PATH = DATA_PROCESSED_DIR / "chunks.jsonl"

# ── LLM (remote, via OpenRouter / NVIDIA NIM) ─────────────────────────────
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "your-chosen-model")

# ── Embedding model (local, no API key) ────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# ── Index paths ────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "indexes" / "chroma"))
BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", str(PROJECT_ROOT / "indexes" / "bm25.pkl"))

# ── Retrieval parameters ──────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))        # tokens (approx)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))   # tokens (approx)
DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", "10"))       # ChromaDB retrieval
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "10"))         # BM25 retrieval
HYBRID_TOP_N = int(os.getenv("HYBRID_TOP_N", "5"))      # Final fused results
RRF_K = int(os.getenv("RRF_K", "60"))                   # RRF constant (paper default)

# ── ChromaDB collection name ──────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "tax_advisor_chunks"
