# Indian Tax Advisor RAG

A Retrieval-Augmented Generation (RAG) assistant for Indian personal income tax under the **Income-tax Act, 2025** (FY 2026-27 / AY 2027-28). It answers from official documents, cites the exact context blocks it used, and checks those citations deterministically before showing the answer.

> Status: portfolio project. Retrieval quality is measured on a 42-question proxy gold set (see [Results](#results)); answers are **not** legally verified. See [Limitations](#limitations).

---

## Architecture

```
User query (Streamlit chat, or one of the example prompts)
    │
    ▼
Hybrid retrieval                                    src/retrieval/
├── Dense: bge-small-en-v1.5 (local) + exact cosine over vectors stored in Chroma
├── Sparse: BM25Okapi over a shared tokenizer (₹1,50,000 ≡ Rs 1,50,000 ≡ 150000)
└── Reciprocal Rank Fusion (k=60) → top 5 chunks, each with source doc + Section/Rule/Schedule
    │
    ▼
Generation                                          src/generation/, src/llm_client.py
├── Grounded prompt: cite [n], never invent sections/₹ amounts, regime-neutral, old↔new numbers
├── OpenAI-compatible endpoint (OpenRouter / NVIDIA NIM), 60 s timeout, retries, FALLBACK_MODELS
└── Deterministic citation check: [n] must exist; cited section numbers / ₹ amounts must
    appear in the cited chunk(s) → warnings in the UI
    │
    ▼
Answer + model-reported sources + citation warnings + retrieved chunks (ground truth) + disclaimer
```

**Why hybrid?** Tax text is full of exact tokens (section numbers, ₹ amounts, acronyms) that BM25 matches well, while dense retrieval handles paraphrase. Measured on the gold set, weighted hybrid beats BM25 alone on recall@5 (0.952 vs 0.905) with equal recall@10 and practically equal MRR@10 (0.749 vs 0.750), and beats dense by a wide margin. See [Results](#results).

**Why exact dense search instead of HNSW?** On this corpus Chroma's HNSW returned different top-10s between process runs for 7 of 42 eval questions, and differed from exact search on 37 of 42. Exact cosine over about 2.7k vectors takes milliseconds, so the app uses it by default (`DENSE_BACKEND=exact`). HNSW remains available for much larger corpora.

---

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt     # runtime + pytest/ruff (requirements.txt = runtime only)
cp .env.example .env                    # fill in LLM_API_KEY and LLM_MODEL (or enter them in the sidebar)

make test                               # 73 offline tests, no API key needed
make app                                # streamlit run app/streamlit_app.py
```

The committed `indexes/` and `data/processed/chunks.jsonl` are enough to run the app; you only need the raw PDFs to re-chunk.

| Command | What it does |
|---|---|
| `make test` / `make lint` | pytest (113 tests) / ruff |
| `make eval` | Retrieval eval (dense, BM25, hybrid) → `eval/results/latest.{json,md}`; no API key |
| `make eval-gen` | 6 end-to-end answers with your `.env` model (6 LLM calls) |
| `make data` | Check `data/raw/` against `data/sources.json`; downloads files that have a direct URL |
| `make chunk` | Re-chunk the PDFs in `data/raw/` → `chunks.jsonl` (needs the PDFs) |
| `make upgrade-chunks` | Re-apply normalisation / section refs / tags to the existing `chunks.jsonl` (no PDFs needed) |
| `make index` | Rebuild Chroma + BM25 + `indexes/manifest.json` from `chunks.jsonl` |

**Adding your own books or notes:** open the **Documents** page in the app sidebar, upload a PDF / Markdown / text file and state its licence. It is chunked, embedded locally and searched alongside the Act as a *supplementary* source; **Delete** removes it from both indexes. Uploads are stored under `data/user/` (git-ignored, never committed). The committed base indexes are not modified.

**Docker:** `docker build -t tax-advisor .` then `docker run --rm -p 8501:8501 --env-file .env -v "$PWD/data/user:/app/data/user" tax-advisor` (query logging is off in the image).

### Configuration (`.env`)

| Variable | Default | Notes |
|---|---|---|
| `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` | –, OpenRouter, – | Can also be entered in the sidebar (kept in the session only) |
| `FALLBACK_MODELS` | empty | Comma-separated. At session start the app probes the primary model and every fallback with a tiny request (~5 s for 10 models), uses the first that works and moves failing ones to the end; each answer still falls back on rate limit / 5xx / timeout / empty reply |
| `LLM_TIMEOUT`, `LLM_MAX_RETRIES` | 60, 2 | Per call |
| `MAX_QUERY_CHARS` | 2000 | Longer questions are rejected |
| `DENSE_BACKEND` | `exact` | `hnsw` for large corpora |
| `DENSE_QUERY_PREFIX` | empty | bge query instruction; measured worse here (dense recall@5 0.738 → 0.714) |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | 1500, 150 | **Characters** |
| `CHROMA_PERSIST_DIR`, `BM25_INDEX_PATH` | `indexes/chroma`, `indexes/bm25.json` | Relative paths resolve against the project root, not the working directory |
| `BM25_QUERY_EXPANSION` | 1 | Statutory-wording expansion for BM25 (`src/retrieval/query_expansion.py`) |
| `RRF_DENSE_WEIGHT`, `RRF_BM25_WEIGHT` | 1.0, 1.5 | Weighted reciprocal rank fusion |
| `ABSTAIN_DENSE_THRESHOLD` | 0.68 | "Not found" without an LLM call below this best dense similarity; 0 disables |
| `LLM_QUERY_REWRITE` | 0 | LLM keyword rewrite for BM25 (+1 call; measured worse) |
| `CONVERSATION_MEMORY`, `MEMORY_TURNS` | 0, 3 | Rewrite follow-ups into standalone questions (+1 call) |
| `SECTION_HEADER_IN` | empty | Index chunks with their "Section N — title" header (`bm25`/`dense`); measured worse; needs `make index` |
| `QUERY_LOG_PATH` | `logs/queries.jsonl` | Per-query JSONL (latency per stage, model, tokens, chunk ids, citation result). Contains questions; empty disables |
| `USER_DOCS_DIR`, `MAX_UPLOAD_MB`, `MAX_UPLOAD_PAGES` | `data/user`, 15, 500 | Uploaded documents overlay |

---

## Data

| File | Type | Notes |
|---|---|---|
| `act_2025_full_text.pdf` | primary | Income-tax Act, 2025 as amended by FA 2026 |
| `income_tax_rules_2026.pdf` | primary | Income-tax Rules, 2026 |
| `faq_transition_1961_to_2025.pdf` | primary | Official transition FAQs |
| `press_release_act_2025_commencement.pdf` | primary | Commencement press release |
| `circulars/*.pdf` | primary | CBDT Circulars 5 and 6 of 2026 |
| `section_mapping_1961_to_2025.md` | **curated** | Hand-written old→new mapping; not independently verified |

The raw PDFs are not committed (see `.gitignore`). `data/sources.json` records the URL, licence, retrieval date and (once recorded) the sha256 of each file. `make data` reports missing files or hash mismatches and exits non-zero.

**Ingestion** (`src/ingest/`):
1. pypdf text is extracted per page.
2. The text is normalised (`src/text_utils.py`):
   - the backtick that pypdf emits for `₹` is turned back into `₹`;
   - amounts written in words get a numeric form ("fifteen lakh rupees (₹15,00,000)");
   - clause markers are tidied.
3. Recursive 1500-character chunks are made.
4. For the Act and Rules, every chunk is assigned its **Section / Rule / Schedule**. This works by tracking headings that increase monotonically, so numbered table rows and amendment footnotes are not mistaken for headings.
5. Keyword topic tags and duplicate removal follow.

Corpus: **2,748 chunks**, and 100% of the Act and Rules chunks carry a `section_ref` (for example, the slab table is in `Section 202`).

---

## Results

All numbers come from `eval/` runs committed under `eval/results/`.

**Retrieval** (42 answerable questions across 15 categories; `make eval`; `eval/results/n1_final.md`):

| config | recall@5 | recall@10 | MRR@10 |
|---|---|---|---|
| dense | 0.714 | 0.786 | 0.573 |
| BM25 (+ query expansion, stopwords) | 0.905 | 0.952 | 0.750 |
| **hybrid (app; weighted RRF)** | **0.952** | **0.952** | 0.749 |
| *earlier hybrid (`post_changes`)* | *0.809* | *0.881* | *0.686* |

These figures are a **proxy**. A retrieved chunk counts as relevant when it comes from an expected source and contains expected keywords, and none of the gold items are human-verified. Use them to compare configurations, not as absolute accuracy. Details are in [`eval/README.md`](eval/README.md).

**What the numbers say:**
- The gains come from three general changes: a **stopword** list in the shared tokenizer, a small documented **query-expansion** map from everyday words to statutory wording for BM25 ("slab" → "rate of tax", "TDS" → "deduction of tax at source", "late filing" → "fee for default in furnishing return"), and **weighted RRF** (BM25 1.5, dense 1.0; any BM25 weight from 1.5 to 3 gives the same recall@5). They were selected on the same 42 questions, so expect some optimism.
- Rejected after measuring: prefixing chunks with their section title (helped BM25 on one question, hurt hybrid overall), the bge query prefix, and an LLM keyword rewrite of the question (hybrid recall@5 0.929 vs 0.952, +1 call per question). The last two remain available as off-by-default flags.
- Still missed: `q1_slabs` (the Section 202 rate table is outranked by the mapping table and FAQ) and `tds_rent` (the Rent row of the Section 393 table is outranked by FAQ text about TDS).

**Generation smoke run** (`eval/results/gen_smoke_n1.json`, model `upstage/solar-mini4`, 6 calls):
- Q2–Q6 meet the criteria in `tests/test_questions.md`, including the out-of-scope refusal (FY 2024-25) and a neutral regime answer.
- Q1 still says the slab rates are not in the retrieved context (retrieval miss, not a hallucination); the citation check flags 7 section/₹ claims that its cited blocks do not contain.

**Abstention:** questions whose best dense similarity is below 0.68 get a "not found" answer without an LLM call. On the gold set this abstains on 3 of 5 out-of-scope questions (GST, stock tips, FY 2012-13) and 0 of 42 answerable ones; the margin is thin (lowest answerable 0.694). The other two (US tax, FY 2024-25) are on-topic enough to pass and are refused by the prompt instead.

---

## Project structure

```
app/streamlit_app.py           Chat UI (one answer path for example buttons + chat input)
src/config.py                  Settings from .env; root-anchored paths; index presence check
src/llm_client.py              OpenAI-compatible client: timeout, retries, fallback models
src/text_utils.py              Shared tokenizer + text normalisation (₹, amounts in words)
src/ingest/                    download_sources (registry check) → chunk → tag_metadata → build_index
src/retrieval/                 dense (exact / hnsw), bm25 (JSON index, no pickle), hybrid (RRF)
src/generation/                prompt_templates, chain.answer(), citations.verify_citations()
eval/                          gold.jsonl, run_eval.py (retrieval), run_generation.py, results/
tests/                         73 pytest tests (unit, Streamlit AppTest, index integrity)
indexes/                       chroma/, bm25.json, manifest.json (counts + chunks.jsonl hash)
data/sources.json              Source registry (URL, licence, sha256)
.github/workflows/ci.yml       ruff + pytest + retrieval eval with a regression gate
```

---

## Design decisions

| Decision | Rationale |
|---|---|
| Local embeddings (bge-small-en-v1.5) | Free, offline, deterministic; only generation needs an API key |
| BM25 + dense with RRF | Rank-based fusion with no score calibration; best MRR on the gold set |
| Exact dense search | HNSW was non-deterministic here; exact is cheap at this size |
| JSON BM25 index | Loading never unpickles; BM25Okapi is refitted at startup (< 1 s) |
| Monotonic heading tracking | Assigns sections without a layout parser; rejects table rows and footnotes |
| Deterministic citation check | Catches invented section numbers or ₹ amounts without a second LLM call |
| Keyword topic tags | Transparent but weak (65% of chunks are `general`); displayed only, not used for ranking |
| No query rewriting / HyDE (yet) | Not measured; the `q1_slabs` vocabulary gap is the main candidate for it |

---

## Limitations

- **Not legal advice; content unverified.** The gold set and the hand-written mapping table have not been checked by a tax professional.
- Rate tables are still hard to retrieve (`q1_slabs`, `tds_rent`); see Results.
- The citation check verifies numbers and section references only; it does not judge whether the prose is correct.
- Conversation memory exists but is off by default (`CONVERSATION_MEMORY=1`) and has not been measured.
- The abstention threshold was chosen on the same small gold set it is reported on.
- Uploaded documents are searched only with the default exact dense backend (`DENSE_BACKEND=hnsw` ignores them), and live on the machine running the app (`data/user/`, git-ignored).
- English only.

---

## Disclaimer

This system provides general tax information based on the documents in its corpus and is not professional tax advice. Always consult a qualified Chartered Accountant (CA) for advice specific to your situation.
