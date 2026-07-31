# Indian Tax Advisor RAG

A Retrieval-Augmented Generation (RAG) system for answering Indian personal income tax questions under the **Income-tax Act, 2025** (FY 2026-27 / AY 2027-28).

---

## Architecture

```
User Query
    │
    ▼
Hybrid Retrieval
├── Dense: ChromaDB + BAAI/bge-small-en-v1.5 (local, no API key)
├── Sparse: BM25Okapi (rank_bm25)
└── Fusion: Reciprocal Rank Fusion (RRF, k=60) → Top-5 chunks
    │
    ▼
Generation
└── OpenRouter / NVIDIA NIM (OpenAI-compatible API)
    System prompt: grounding rules + structured output format
    │
    ▼
Streamlit Chat UI
└── Answer + Retrieved Sources (ground truth) + Disclaimer
```

**Why hybrid search?** Tax text is full of exact terms like section numbers (80C, 123), ₹ amounts, and acronyms that BM25 handles better than embeddings. Dense retrieval handles semantic similarity. RRF fuses both without a black-box ensemble.

**Why local embeddings?** `BAAI/bge-small-en-v1.5` via `sentence-transformers` runs entirely on CPU — no API key, no cost, deterministic across runs. Only the generation call uses the configured API endpoint.

---

## Data Sources

| File                                      | Description                                   | Source Type |
| ----------------------------------------- | --------------------------------------------- | ----------- |
| `act_2025_full_text.pdf`                  | Income-tax Act, 2025 (as amended by FA 2026)  | Primary     |
| `income_tax_rules_2026.pdf`               | Income-tax Rules, 2026 (notified 20 Mar 2026) | Primary     |
| `faq_transition_1961_to_2025.pdf`         | Official FAQs on transition from 1961 Act     | Primary     |
| `press_release_act_2025_commencement.pdf` | Press release: Act in force from 1 Apr 2026   | Primary     |
| `circulars/*.pdf`                         | CBDT Circulars No. 5 & 6 of 2026              | Primary     |
| `section_mapping_1961_to_2025.md`         | Hand-written old→new section mapping table    | Primary     |

Total corpus: **2,761 chunks** indexed.

---

## Setup

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd TaxAdvisorRAG

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — fill in LLM_API_KEY and LLM_MODEL
```

`.env` values:

```
LLM_API_KEY=<your OpenRouter or NIM API key>
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=<e.g. google/gemini-flash-1.5>
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5   # runs locally
CHROMA_PERSIST_DIR=indexes/chroma
BM25_INDEX_PATH=indexes/bm25.pkl
```

---

## Build Indexes (first run only)

```bash
# Chunk all documents
python -m src.ingest.chunk

# Build ChromaDB + BM25 indexes
python -m src.ingest.build_index
```

---

## Run the App

```bash
streamlit run app/streamlit_app.py
```

---

## Project Structure

```
TaxAdvisorRAG/
├── data/
│   ├── raw/                         # Source PDFs + section mapping
│   │   ├── section_mapping_1961_to_2025.md
│   │   ├── SOURCES.md
│   │   └── circulars/
│   └── processed/
│       └── chunks.jsonl             # 2,761 tagged chunks
├── src/
│   ├── config.py                    # All settings from .env
│   ├── llm_client.py                # OpenAI-compatible LLM wrapper
│   ├── ingest/
│   │   ├── chunk.py                 # PDF/MD parsing + chunking
│   │   ├── tag_metadata.py          # Keyword-based topic tagging
│   │   ├── build_index.py           # Embeds + builds ChromaDB + BM25
│   │   └── download_sources.py      # Source verification helper
│   ├── retrieval/
│   │   ├── dense_retriever.py       # ChromaDB semantic search
│   │   ├── bm25_retriever.py        # BM25 sparse search
│   │   └── hybrid.py                # RRF fusion (~20 lines)
│   └── generation/
│       ├── prompt_templates.py      # System prompt + context formatter
│       └── chain.py                 # answer(query) → {answer, sources, disclaimer}
├── app/
│   └── streamlit_app.py             # Chat UI
├── indexes/
│   ├── chroma/                      # Persisted ChromaDB
│   └── bm25.pkl                     # Pickled BM25 index
└── tests/
    └── test_questions.md            # 6 test scenarios with quality criteria
```

---

## Design Decisions

| Decision                                    | Rationale                                                                                          |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Local embeddings (bge-small-en-v1.5)        | Free, offline-capable, deterministic, no API dependency                                            |
| BM25 + Dense hybrid                         | Tax text has many exact terms (section numbers, ₹ amounts) where BM25 outperforms embeddings alone |
| RRF fusion (hand-written)                   | 20-line formula, whiteboard-explainable, no black-box ensemble                                     |
| Keyword topic tagging                       | Transparent, free, inspectable — no ML classifier needed at this corpus size                       |
| Structured ANSWER/SOURCES/DISCLAIMER output | Mechanically parsable; UI shows actual retrieved chunks alongside model's self-reported sources    |
| No query rewriting / HyDE                   | Tax queries are usually direct; complexity not justified for this domain                           |

---

## Disclaimer

This system provides general tax information based on the documents in its corpus and is not professional tax advice. Always consult a qualified Chartered Accountant (CA) for advice specific to your situation.
