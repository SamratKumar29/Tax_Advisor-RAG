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

**Why hybrid?** Tax text is full of exact tokens (section numbers, ₹ amounts, acronyms) that BM25 matches well, while dense retrieval handles paraphrase. Measured on the gold set, hybrid has the best MRR@10 (0.686 vs 0.599 BM25 and 0.578 dense), but BM25 alone has slightly higher recall@5 (0.857 vs 0.809). See [Results](#results).

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
| `make test` / `make lint` | pytest (73 tests) / ruff |
| `make eval` | Retrieval eval (dense, BM25, hybrid) → `eval/results/latest.{json,md}`; no API key |
| `make eval-gen` | 6 end-to-end answers with your `.env` model (6 LLM calls) |
| `make data` | Check `data/raw/` against `data/sources.json`; downloads files that have a direct URL |
| `make chunk` | Re-chunk the PDFs in `data/raw/` → `chunks.jsonl` (needs the PDFs) |
| `make upgrade-chunks` | Re-apply normalisation / section refs / tags to the existing `chunks.jsonl` (no PDFs needed) |
| `make index` | Rebuild Chroma + BM25 + `indexes/manifest.json` from `chunks.jsonl` |

### Configuration (`.env`)

| Variable | Default | Notes |
|---|---|---|
| `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` | –, OpenRouter, – | Can also be entered in the sidebar (kept in the session only) |
| `FALLBACK_MODELS` | empty | Comma-separated; tried in order on rate limit / 5xx / timeout / empty reply |
| `LLM_TIMEOUT`, `LLM_MAX_RETRIES` | 60, 2 | Per call |
| `MAX_QUERY_CHARS` | 2000 | Longer questions are rejected |
| `DENSE_BACKEND` | `exact` | `hnsw` for large corpora |
| `DENSE_QUERY_PREFIX` | empty | bge query instruction; measured worse here (dense recall@5 0.738 → 0.714) |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | 1500, 150 | **Characters** |
| `CHROMA_PERSIST_DIR`, `BM25_INDEX_PATH` | `indexes/chroma`, `indexes/bm25.json` | Relative paths resolve against the project root, not the working directory |

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

**Retrieval** (42 answerable questions across 15 categories; `make eval`):

| config | recall@5 | recall@10 | MRR@10 | p50 latency |
|---|---|---|---|---|
| dense | 0.738 | 0.786 | 0.578 | ~20 ms |
| BM25 | 0.857 | 0.881 | 0.599 | ~6 ms |
| **hybrid (app)** | 0.809 | 0.881 | **0.686** | ~30 ms |

These figures are a **proxy**. A retrieved chunk counts as relevant when it comes from an expected source and contains expected keywords, and none of the gold items are human-verified. Use them to compare configurations, not as absolute accuracy. Details are in [`eval/README.md`](eval/README.md).

**What the numbers say:**
- The ingestion improvements (₹ normalisation, section refs, dedup) were **retrieval-neutral** on this set: hybrid recall@5 was 0.809 before and 0.809 after. Their value is in traceable citations and data hygiene.
- The bge query prefix made dense retrieval worse and stays off.
- Weakest categories: `tds`, `salary`, and `slabs`. `q1_slabs` ("new regime slab rates") still misses, because the rate table never uses the words "slab" or "new regime".

**Generation smoke run** (`eval/results/gen_smoke_post_changes.json`, model `upstage/solar-mini4`, 6 calls, 4–15 s each, about 2.3k–2.9k tokens):
- Q2–Q6 meet the criteria in `tests/test_questions.md`. That includes the out-of-scope refusal (FY 2024-25) and the regime question, where the answer stayed neutral.
- Q1 correctly says the slab rates are not in the retrieved context, which is a retrieval miss rather than a hallucination.
- The citation check flagged 3 section-number claims (in 2 sentences) that the model attributed to a context block not containing them.

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
- Retrieval misses questions whose wording differs from the statute's ("slab rates" vs the rate table in Section 202).
- The citation check verifies numbers and section references only; it does not judge whether the prose is correct.
- Single-turn: follow-up questions do not use the chat history.
- English only. A PDF upload UI for adding books is not implemented yet; for now, add files under `data/raw/`, register them in `data/sources.json`, then run `make chunk index`.
- Page numbers are only recorded when re-chunking from the PDFs. The committed chunks were upgraded without the PDFs, so they have section refs but no pages.

---

## Disclaimer

This system provides general tax information based on the documents in its corpus and is not professional tax advice. Always consult a qualified Chartered Accountant (CA) for advice specific to your situation.
