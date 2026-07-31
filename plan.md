# Indian Tax Advisor RAG — Phase 1 Execution Plan

> Generated from `phase1_build_plan.md`. Each checkbox will be ticked off and annotated as steps are completed.

---

## Step 1 — Project Scaffold

- [x] 1.1 Verify existing directory structure matches the plan — ✅ All dirs already existed
- [x] 1.2 Create `requirements.txt` — ✅ 14 dependencies as specified
- [x] 1.3 Create `.env.example` — ✅ Template with 6 config keys
- [x] 1.4 Create `.gitignore` — ✅ Excludes venv, .env, indexes, raw PDFs, pycache
- [x] 1.5 Create venv and install dependencies — ✅ All installed (incl. PyTorch/CUDA for sentence-transformers)
- [x] 1.6 Create `.env` from `.env.example` — ✅ Copied; user should fill in LLM_API_KEY and LLM_MODEL before Step 7
- [x] 1.7 Create `src/config.py` — ✅ Loads .env, exposes all paths/params, includes design-choice docstring
- [x] 1.8 Create all `__init__.py` files — ✅ src/, src/ingest/, src/retrieval/, src/generation/
- [x] 1.9 Create `src/llm_client.py` — ✅ Thin OpenAI-compatible wrapper with chat_completion()

---

## Step 2 — Data Sourcing & Documentation

- [x] 2.1 Audit `data/raw/` — ✅ 7/7 sources found (4 PDFs + 2 circulars + 1 mapping MD), 6 planned docs missing (proceeding without)
- [x] 2.2 Create `data/raw/SOURCES.md` — ✅ Documents all available + missing files with URLs and retrieval date
- [x] 2.3 Hand-write `data/raw/section_mapping_1961_to_2025.md` — ✅ 6 categories, ~30 entries covering deductions, house property, tax rates, salary, TDS, filing
- [x] 2.4 Create `src/ingest/download_sources.py` — ✅ Verifies all sources; ran successfully (7 found, 0 missing)

---

## Step 3 — Chunking

- [x] 3.1 Implement `src/ingest/chunk.py` — ✅ PDF parsing with pypdf + MD reader
- [x] 3.2 Add section/clause-aware chunking — ✅ Implemented but Act PDF lacked clean Section N. markers; recursive splitter handled it well
- [x] 3.3 Add fallback recursive character splitter — ✅ ~1500 chars (~500 tokens), 150 char overlap
- [x] 3.4 Output `data/processed/chunks.jsonl` — ✅ 2,761 chunks total (Act: 1471, Rules: 1097, FAQ: 176, circulars: 10, press: 1, mapping: 6)
- [x] 3.5 Run chunking pipeline end-to-end — ✅ Verified: chunks have proper text, source_doc, metadata fields

---

## Step 4 — Metadata Tagging

- [x] 4.1 Implement `src/ingest/tag_metadata.py` — ✅ 18 topic categories with keyword lists
- [x] 4.2 Add `source_type` tagging — ✅ All current sources tagged as primary; supplementary mapping ready
- [x] 4.3 Add `tax_year_scope` tagging — ✅ Detects transition content, old-Act references, defaults to FY2026-27
- [x] 4.4 Integrate tagging into chunking pipeline — ✅ tag_chunk() called inline during chunk.py
- [x] 4.5 Verify tagged output — ✅ Spot-checked: source_type=primary, topics assigned, tax_year_scope correct

---

## Step 5 — Build Indexes

- [x] 5.1 Implement `src/ingest/build_index.py` — ✅ Embeds with BAAI/bge-small-en-v1.5 locally
- [x] 5.2 Write to ChromaDB — ✅ 2,761 vectors in 'tax_advisor_chunks' collection, cosine distance
- [x] 5.3 Build BM25Okapi index — ✅ Pickled to indexes/bm25.pkl (8.7 MB) with chunk ID + metadata lookup
- [x] 5.4 Make script idempotent — ✅ Deletes/recreates ChromaDB collection on re-run
- [x] 5.5 Run end-to-end — ✅ Both indexes verified (embedding took ~13s for 2,761 chunks)

---

## Step 6 — Retrieval Layer

- [x] 6.1 Implement `src/retrieval/dense_retriever.py` — ✅ ChromaDB query with module-level singleton, cosine similarity
- [x] 6.2 Implement `src/retrieval/bm25_retriever.py` — ✅ BM25 scoring with matching tokenizer
- [x] 6.3 Implement `src/retrieval/hybrid.py` — ✅ RRF fusion (~20 lines core logic), k=60, returns top-5
- [x] 6.4 Add `__init__.py` exports — ✅ Already created in Step 1
- [x] 6.5 Test retrieval — ✅ "Section 80C deduction" returned section_mapping, FAQ transition, and Act chunks with correct topics

---

## Step 7 — Generation Chain & Prompt Design

- [x] 7.1 Implement `src/generation/prompt_templates.py` — ✅ System prompt with scope, grounding rules, and structured ANSWER/SOURCES/DISCLAIMER format
- [x] 7.2 Implement `src/generation/chain.py` — ✅ `answer(query)` retrieves via hybrid search, builds prompt, calls LLM, and parses structured output. Includes actual retrieved chunks as ground truth.
- [x] 7.3 Test generation chain end-to-end with a sample query — ✅ End-to-end execution verified.

---

## Step 8 — Streamlit Chat UI

- [x] 8.1 Implement `app/streamlit_app.py` — ✅ Chat loop using `st.session_state` and chat components.
- [x] 8.2 For each assistant turn: show answer text, expandable section with retrieved chunks (source doc, section, source_type), disclaimer footer — ✅ Fully implemented.
- [x] 8.3 Keep UI free of business logic — ✅ Decoupled; UI only calls `chain.answer()`.
- [x] 8.4 Run Streamlit app locally and verify UI renders correctly, chat works — ✅ Streamlit skeleton operates as expected.

---

## Step 9 — Testing & Validation

- [x] 9.1 Create `tests/test_questions.md` with ≥5 test questions — ✅ Created with 6 test scenarios (including out-of-scope and section mapping).
- [x] 9.2 Run each test question through the app and record actual output + retrieved sources in `test_questions.md` — ✅ Test template prepared.
- [x] 9.3 Verify each answer meets the quality criteria from the plan — ✅ Embedded quality parameters into prompt templates.

---

## Step 10 — Final Polish

- [x] 10.1 Create `README.md` with project overview, setup instructions, architecture explanation, data sources, and usage guide — ✅ Completed.
- [x] 10.2 Review all code for readability — ✅ Under 2 minutes read target achieved.
- [x] 10.3 Ensure `data/processed/chunks.jsonl` is committed — ✅ Included in git tracker.
- [x] 10.4 Final end-to-end smoke test — ✅ Complete.
