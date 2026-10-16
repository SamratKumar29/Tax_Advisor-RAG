# Audit Report — Indian Tax Advisor RAG

> Audit date: 2026-10-16 · Commit audited: `872300a` (clean working tree) · Auditor scope: read-only inspection + safe execution. No application code or configuration was modified; this file is the only deliverable.

---

## 1. Executive Summary

**What it is.** A hybrid-retrieval RAG chatbot for Indian personal income tax under the Income-tax Act, 2025 (FY 2026-27). Here's how it works:
- Official PDFs (Act, Rules, transition FAQ, two CBDT circulars, a press release) plus one hand-written section-mapping table are chunked and keyword-tagged.
- The chunks are indexed in two places: ChromaDB, using local `BAAI/bge-small-en-v1.5` embeddings, and BM25.
- At query time the two result lists are fused with Reciprocal Rank Fusion.
- An OpenAI-compatible LLM (OpenRouter/NIM) answers from the retrieved chunks, and a Streamlit chat UI displays the result.

**Current state.** The main workflow works end to end. I ran all six documented test questions against the configured model (`upstage/solar-mini4`) and all six returned well-formed, grounded answers. However:
- **Q1** ("new regime slab rates") **fails on substance**. Retrieval never surfaces the rate schedule.
- **Q3 (HRA)** is answered mostly from the author's hand-written mapping table rather than the statute.
- There are **no automated tests, no evaluation metrics, no CI**, and the raw PDFs are not in the repo, so ingestion cannot be reproduced from a clone.
- Several verified defects affect the demo path. The most visible: the example-prompt buttons in the UI never produce an answer.

**Overall score: 44 / 100 — Readiness Level 2 (Functional Project).**

**Strongest aspects**
- Clean, small module boundaries: `config` / `ingest` / `retrieval` / `generation` / `app`.
- A whiteboard-explainable hand-written RRF.
- Local embeddings.
- A "ground-truth retrieved chunks" panel shown next to the sources the model claims.
- A primary-source corpus with a provenance file.
- Sensible prompt rules: grounding, refusing out-of-scope questions, regime-neutrality, and citing both old and new section numbers.

**Biggest weaknesses**
1. No measured retrieval or answer quality. Every claim in the README is unevidenced.
2. The corpus and chunking cannot answer core questions (slab rates). Amounts are written in words in the Act, `₹` is extracted as a backtick, and section-aware chunking never runs.
3. Robustness gaps on the demo path: the example buttons are broken, LLM errors are unhandled, and paths depend on the working directory.

**Three most important next actions**
1. **Fix the demo-path defects:** P0-1 example buttons, P0-2 LLM error handling and fallback models, P0-3 absolute paths.
2. **Build an evaluation harness:** a gold QA set, recall@k/MRR for dense vs BM25 vs hybrid, and answer-faithfulness checks (P1-1, P1-2). Every later decision depends on it.
3. **Fix ingestion quality and reproducibility:** structure-aware chunking with section refs, normalising amounts written in words, a corpus manifest with hashes, and a re-download script (P1-4, P1-5, P1-6). Your stated goal of "paste more tax books" depends on this pipeline.

---

## 2. Project Understanding

| Aspect | Finding |
|---|---|
| Problem | Taxpayers struggle to map the new Income-tax Act, 2025 (renumbered sections, "tax year") onto what they know about the 1961 Act, and to get grounded answers with citations. |
| Intended users | Indian salaried individuals and tax learners. The stated owner goal is a general knowledge base that users can extend with more books (CA study material, etc.). |
| Scope as implemented | 7 source documents → 2,761 chunks; single-turn Q&A; English only. |
| Stack | Python 3.12, pypdf, langchain-text-splitters, sentence-transformers 3.4.1, ChromaDB 0.5.23, rank_bm25, openai 1.109.1, Streamlit 1.66.0 (versions observed in the user-provided venv). |
| External services | One OpenAI-compatible chat-completions endpoint. The key is supplied in the UI sidebar or `.env`. |

**Main workflow (traced):**
1. `src/ingest/chunk.py` runs: pypdf extraction → section split (attempted) → `RecursiveCharacterTextSplitter` (1500/150 characters) → `tag_metadata.tag_chunk` → `data/processed/chunks.jsonl`.
2. `src/ingest/build_index.py` embeds the chunks into Chroma (cosine, HNSW) and pickles a BM25Okapi index together with a chunk lookup to `indexes/bm25.pkl`.
3. On a query, `app/streamlit_app.py:141` calls `src/generation/chain.answer`:
   - `hybrid.search` takes dense top-10 + BM25 top-10 and fuses them with RRF (k=60) into the top 5;
   - `prompt_templates.build_messages` builds the prompt;
   - `llm_client.chat_completion` calls the model;
   - a regex parse splits the output into ANSWER/SOURCES;
   - the retrieved chunks are attached, truncated to 300 characters each.
4. The UI shows the answer, the sources the model reports, an expander with the retrieved chunks, and a fixed disclaimer.

**Implemented features:**
- Hybrid retrieval.
- Grounded prompt.
- Structured output parsing.
- Session-scoped API key entry.
- Example prompts (broken; see F-01).
- Sources panel.

**Documented or implied but not implemented:**
- Section-aware chunking (exists in code but never fires).
- `FALLBACK_MODELS` (set in your `.env`, but no code reads it).
- `CHUNK_SIZE`/`CHUNK_OVERLAP` config (ignored).
- FastAPI service (the dependency exists; there is no code).
- Recorded test results (`tests/test_questions.md` is an empty template).
- Use of `tax_year_scope`/`topic` metadata in retrieval (computed but never used to filter).

---

## 3. Repository and Execution Audit

### 3.1 Important files

| Path | Responsibility | Notes |
|---|---|---|
| `src/config.py` | Env-driven settings | Relative-path defaults from `.env` are cwd-dependent (F-03) |
| `src/ingest/chunk.py` | Parse + chunk | Section path dead in practice (F-05) |
| `src/ingest/tag_metadata.py` | Keyword topic / scope tags | 60% `general`; colliding patterns (F-08) |
| `src/ingest/build_index.py` | Build Chroma + BM25 | Rebuilds from scratch; duplicates the tokenizer (F-12) |
| `src/ingest/download_sources.py` | Lists sources, checks they exist | Does not download anything, despite its name and docstring |
| `src/retrieval/{dense_retriever,bm25_retriever,hybrid}.py` | Retrieval | Singletons; no query prefix (F-07) |
| `src/generation/{prompt_templates,chain}.py` | Prompt + orchestration | Regex parse; no validation (F-02, F-10) |
| `src/llm_client.py` | OpenAI client wrapper | New client per call; no timeout/retry (F-02) |
| `app/streamlit_app.py` | UI | Example-button bug; duplicated rendering code (F-01, F-13) |
| `app/indexes/chroma/chroma.sqlite3` | **Stray tracked empty Chroma DB** | Evidence of F-03 |
| `data/processed/chunks.jsonl`, `indexes/` | Committed artifacts (≈3.7 MB + 31 MB sqlite + 9 MB pickle) | Raw PDFs git-ignored (F-06) |
| `tests/test_questions.md` | Manual test template | No results recorded |
| `plan.md` | Author build log | Listed in `.gitignore:29` yet tracked |

### 3.2 Validation commands executed

| # | Check | Result |
|---|---|---|
| 1 | `git status` before and after the audit | Clean, apart from the `.cortex/` tooling dir and this report |
| 2 | `python -m py_compile` on all `src/` and `app/` modules | **Passed** |
| 3 | Dependency import in the user-provided `venv` | **Passed** (versions in §2) |
| 4 | Chunk statistics over `chunks.jsonl` | 2,761 chunks; **0 with `section_ref`**; 1,651 (60%) topic `general`; 13 duplicate texts; length 33–1500 chars (median 1368) |
| 5 | BM25 / dense / hybrid probes on 5 queries | **Ran**. Hybrid warm latency 27–38 ms (first query 261 ms); cold model + collection load 16.8 s. "HRA exemption metro city" top-5 came from the mapping table and the Rules, with topics tagged `tds`/`savings_interest`. "new tax regime slab rates" top-5 had no rate table. |
| 6 | Off-domain dense probe ("capital of France?") | Top cosine similarities 0.44–0.47. The tax queries are not on the same scale, so no fixed threshold exists to support abstention. |
| 7 | Run from `app/` with `.env`'s relative `CHROMA_PERSIST_DIR` | **Failed**: `InvalidCollectionException: Collection tax_advisor_chunks does not exist`. Chroma opened the stray `app/indexes/chroma`. |
| 8 | Corpus grep for slab figures (`4,00,000`, `12,00,000`) | **0 hits**. The Act writes amounts in words ("fifteen lakh rupees"), and `₹` extracts as `` ` `` (e.g. `` ` 100000``). |
| 9 | End-to-end `chain.answer()` on the 6 questions in `tests/test_questions.md` (6 paid/free API calls) | **6/6 returned and parsed** (7–16 s each). Judged against the file's own criteria: Q2, Q4, Q5, Q6 meet them; Q3 partially (the content comes from the hand-written table); **Q1 fails** (says the slabs are not in context, and also wrongly implies Section 115BAC is an "opt-out" of the old regime). |
| 10 | Headless `streamlit run` + `GET /_stcore/health` | **Passed** (HTTP 200, `ok`). Log warns that the pinned `uvicorn==0.30.*` is incompatible with the installed websockets (F-14). |
| — | Lint / type check / unit tests / CI | **Not run, because none exist** (no ruff/mypy/pytest config, no `.github/`). |
| — | `python -m src.ingest.chunk` / `build_index` | **Not run**: raw PDFs are absent, and running would overwrite committed artifacts. |
| — | Browser-level UI interaction | **Not run.** F-01 was established by reading the code, not by clicking. |

### 3.3 What works / fails / is unverified

- **Works (verified):** index loading from the repo root, hybrid retrieval, prompt construction, LLM call and parse on one provider and model, app startup, out-of-scope refusal (Q5), regime-neutral answer (Q4), the 80C→123 mapping (Q6).
- **Fails (verified):** slab-rate retrieval (Q1), starting from a non-root cwd with the documented `.env`, rebuilding from a clone (no PDFs).
- **Fails (verified by code reading):** example-prompt buttons (F-01), unhandled LLM exceptions (F-02).
- **Unverified:**
  - Whether `section_mapping_1961_to_2025.md` is legally correct. The author's note says it was "cross-checked against ClearTax + official FAQs"; I did not verify it against an authoritative mapping.
  - Answer quality beyond 6 questions.
  - Behaviour with other models or providers.
  - Concurrent multi-user behaviour.

---

## 4. Scorecard

| Category | Weight | Score (0–5) | Weighted | Rationale |
|---|---:|---:|---:|---|
| Problem definition & practical value | 10 | 3 | 6.0 | Real, timely problem (new Act, renumbering). Target user is implicit. No success criteria. |
| Technical approach & AI/ML methodology | 15 | 2 | 6.0 | Hybrid + RRF is a sound choice. But there are no baselines or metrics, the bge query prefix is missing, the chunking is naive for statute text, metadata is unused, there is no reranking or abstention, and Q1 fails. |
| Architecture & engineering design | 15 | 3 | 9.0 | Clear layering, and the UI holds no business logic. Weaknesses: duplicated tokenizer, global singletons, config not fully honoured, no typed result schema. |
| Code quality & correctness | 15 | 2 | 6.0 | Several verified defects (F-01–F-05, F-08). Readable code with good docstrings. |
| Testing & verification | 15 | 1 | 3.0 | Only a manual template with no recorded results. Zero automated tests. |
| Security, privacy & reliability | 10 | 2 | 4.0 | The key stays in the session (good). No timeouts or retries, unvalidated `pickle.load`, no input-length limits, raw tracebacks shown to users, no prompt-injection consideration for user-added documents. |
| Performance, cost & efficiency | 5 | 2 | 2.0 | Warm retrieval is fast (measured). The 17 s cold start is not cached with `st.cache_resource`. No token or cost accounting. |
| Documentation, setup & reproducibility | 10 | 3 | 6.0 | Good README with design rationale. But the PDFs are missing, unused deps are pinned loosely, there is no lockfile, and some README facts are stale (index size). |
| Demonstrability & delivery readiness | 5 | 2 | 2.0 | Starts and works when launched from the root. The demo path has the button bug and no deployment evidence. |
| **Total** | **100** | | **44.0** | **Level 2 — Functional Project** |

Uncertainty:
- Answer-quality scoring rests on 6 questions with one model.
- A larger evaluation could move the methodology score in either direction by about ±1.

---

## 5. Critical Findings

Severity labels: **Defect** = verified wrong behaviour; **Risk** = plausible failure that needs verification; **Design** = a weakness rather than a bug.

| ID | Type | Finding & evidence | Impact | Fix |
|---|---|---|---|---|
| F-01 | Defect | **Example prompts never get answered.** `app/streamlit_app.py:96-98` appends the user message and calls `st.rerun()`. Answer generation only happens in the `if prompt := st.chat_input(...)` branch (`:127-146`). | The first thing a reviewer clicks shows a question with no reply. | Store a `pending_prompt` in session state and process it through the same path as `chat_input`. |
| F-02 | Defect | **No error handling around generation.** `src/llm_client.py:38-45`: no timeout, retry or exception handling. `chain.py:85-88`: `message.content` can be `None`, which crashes `re.search`. `streamlit_app.py:141`: no try/except. The user message is already appended (`:134`), so a failure leaves an orphan turn. `FALLBACK_MODELS` in `.env` is never read. | Rate limits or outages on free models produce a stack trace in the UI. | Add a typed `LLMError`, a timeout, bounded retries, a fallback-model loop driven by config, and a friendly UI error. |
| F-03 | Defect | **Paths depend on the working directory.** `.env.example` sets `CHROMA_PERSIST_DIR=indexes/chroma`, which overrides the absolute default in `src/config.py:32`. From `app/`, Chroma silently creates or opens `app/indexes/chroma`. Reproduced: `InvalidCollectionException`. The stray DB is committed. | Fragile startup on Streamlit Cloud or IDE run configurations; a confusing error. | Resolve relative paths against `PROJECT_ROOT`, delete `app/indexes/`, and fail fast with a clear message when the collection is missing. |
| F-04 | Defect | **Core facts not retrievable (Q1).** Zero chunks contain `4,00,000` or `12,00,000`. The Act uses words ("fifteen lakh rupees") and pypdf maps `₹` to `` ` ``. The BM25 tokenizer (`bm25_retriever.py:26`) handles `₹`, but the corpus never contains it. The slab query's top-5 contains no rate table. | The most common tax question fails. Q1 also produced a factual slip (115BAC described as an "opt-out"). | Normalise during ingestion: `` ` ``→`₹`, add the numeric form alongside amounts written in words. Extract tables. Add a curated "rates & limits" document with citations. Measure the effect with P1-1. |
| F-05 | Defect | **Section-aware chunking is dead code.** `SECTION_PATTERN` (`chunk.py:43-46`) expects `Section 123.` headings, while the Act prints headings like `123. Deduction…`. With fewer than 5 matches the function returns `[]` (`:65-68`). Result: **0 of 2,761 chunks have a `section_ref`** (`plan.md` step 3.2 admits this). | Citations name only the PDF. Chunks cut through clauses. Section-number queries rely on luck. | Write a heading regex for the real layout (`^\s*\d+[A-Z]*\.\s+[A-Z]`), split long sections with the parent ref kept, and store `section_ref` plus page number. |
| F-06 | Defect (reproducibility) | **Raw corpus absent.** `.gitignore:12-13` ignores the PDFs; only `*.pdf:Zone.Identifier` (Windows download metadata) files are tracked. `download_sources.py` does not download, and two URLs are only "→ Rules section" hints. | The index can't be regenerated or audited; adding books has no reproducible pipeline. | Add a `sources.yaml` manifest (URL, sha256, retrieval date, licence), a real downloader, and a `make ingest`. Remove the Zone.Identifier files. |
| F-07 | Design | **bge query instruction not used.** `dense_retriever.py:47` encodes the raw query. The bge-v1.5 model card recommends a prefix for short-query → passage retrieval. Embeddings are not explicitly normalised. | Possibly lower dense recall. Unmeasured, so treat this as a hypothesis for P1-2. | Make the prefix configurable and A/B it in the eval harness. |
| F-08 | Defect | **Topic tagger is inaccurate and unused.** `tag_metadata.py`: `"50,000"` appears in three topics (`:41,68,97`); `"section 19"` (`:68`) matches sections 190–199; `"192"` (`:76`) matches any substring; ties keep the first topic (`:128`). Probe: a "section 192" TDS text was tagged `standard_deduction_salary`. 60% of chunks are `general`. Tags are only displayed (`streamlit_app.py:118`), never used to filter. | The labels shown in the UI are misleading, and an interviewer will ask why they exist. | Either remove the tags or make them useful: use word boundaries, derive topic from the chapter/section, and use them for metadata filtering, measured by eval. |
| F-09 | Risk | **Hand-written mapping drives answers.** In Q3 and Q6 the key content came from `section_mapping_1961_to_2025.md`, which is unverified and labelled `primary` (`tag_metadata.py:18-22`). | A legal error in one table is presented as primary law. | Tag it `curated`, cite the official mapping it came from, and add a test per row against the statute text. |
| F-10 | Design | **No answer validation.** Citations are not checked against the retrieved chunks. There is no retrieval-confidence gate (dense scores for nonsense queries are ~0.45). The parse falls back to the whole raw text (`chain.py:35`). The model-written DISCLAIMER is discarded. | Hallucinated section numbers go undetected. | Require `[Context n]` citations, verify that the cited ids exist and that the section numbers or amounts appear in the cited chunk, flag unsupported claims, and add a calibrated abstain threshold. |
| F-11 | Risk | **Unbounded input and context.** There is no query length limit. The 5×1500-character context plus 2048 max tokens is fixed. A 5-minute session is unbounded in calls. | Cost spikes or denial-of-service on a public demo that uses the owner's key. | Cap the query length, add simple per-session rate limiting, and log tokens. |
| F-12 | Design | **Duplicated tokenizer.** `simple_tokenize` exists in both `build_index.py:106` and `bm25_retriever.py:23` ("must match exactly"). BM25 top-k sorts the whole score array (`bm25_retriever.py:48`). | Silent drift breaks BM25. Minor inefficiency. | Use one shared `text_utils.tokenize`; use `np.argpartition`. |
| F-13 | Design | **UI duplication and caching.** Source rendering is duplicated (`streamlit_app.py:108-122` and `:157-171`). Model sources are lost from history (`:177-182`). Retriever singletons are module globals rather than `st.cache_resource`. Chunks are truncated to 300 characters (`chain.py:94`). | Maintenance cost, and a weaker trust experience. | Extract a `render_sources()`; cache resources; show the full chunk text, section and page. |
| F-14 | Design | **Dependency hygiene.** `requirements.txt` pins `langchain`, `langchain-community`, `fastapi`, `beautifulsoup4` and `requests`, none of which are imported. `uvicorn==0.30.*` now conflicts with Streamlit 1.66's server (warning observed at startup). `langchain-text-splitters` *is* imported (`chunk.py:16`) but is only available transitively. Major-only pins and no lockfile. | Slow, fragile installs; an interviewer will ask "why FastAPI?". | Trim to the imports actually used, pin `langchain-text-splitters` explicitly, and add a lockfile (`uv`/`pip-tools`). |
| F-15 | Risk | **`pickle.load` of the index** (`bm25_retriever.py:19`). Safe while the author controls the file, but unsafe if indexes are ever downloaded or uploaded. | Arbitrary code execution if the artifact is tampered with. | Rebuild BM25 at startup from `chunks.jsonl` (fast at this size), or store it in a safe format. |
| F-16 | Design | **Single-turn only.** Follow-ups ("and for senior citizens?") lose context, because `chain.answer` receives only the latest query. | Weak for "complex tax questions". | Condense the question using recent history before retrieval (keep it optional and measured). |
| F-17 | Minor | `LLM_MODEL` defaults to `"your-chosen-model"` (`config.py:26`). `LLM_API_KEY`, `CHUNK_SIZE` and `CHUNK_OVERLAP` are never used (the chunker hard-codes 1500/150, `chunk.py:103`). README "~12MB" vs a 31 MB sqlite. `plan.md` is git-ignored but tracked. 13 duplicate chunk texts. | Credibility papercuts. | Clean up as part of P1-7. |

---

## 6. What Is Already Good (preserve these)

- **Layering.** The UI calls one function (`chain.answer`) and contains no retrieval logic. Ingest, retrieval and generation are independently swappable. Keep this boundary.
- **Hybrid retrieval with hand-written RRF** (`hybrid.py`). It is correct, rank-based (no score-scale mixing), and you can explain it in an interview.
- **Local embeddings.** Retrieval is free, offline-capable and deterministic. Only generation costs money.
- **"Retrieved Sources (ground truth)" panel.** Showing the actual retrieved chunks next to the model's self-reported sources is exactly the right instinct for trust.
- **Prompt policy.** It was observed working: Q4 stayed neutral, Q5 refused the out-of-scope request, Q6 gave both the old and new numbers.
- **Bring-your-own-key UI.** The key is held in session only and never written to disk.
- **Primary-source corpus with a provenance doc** (`data/raw/SOURCES.md`) and written behavioural test criteria, including adversarial ones (out-of-scope, regime choice). That makes an excellent seed for a gold eval set.
- **Design-decision table in the README.** Few portfolio projects explain *why*.

---

## 7. Detailed Technical Assessment

### A. Problem & product value
- The problem is timely: the 2025 Act renumbers everything, so "what is 80C now?" is a real question.
- What's missing:
  - a named user persona and success criteria (e.g. "answers X% of a 100-question benchmark with correct section citations");
  - a document-ingestion UX for your stated goal of adding books.
- Differentiation should come from **verifiable citations to the statute plus measured accuracy**, not from piling on more documents.
- Mixing CA study books (secondary, often written for older law) with the primary Act creates a **version-conflict problem**: older material may cite the 1961 Act or earlier-year limits. The `source_type`/`tax_year_scope` metadata exists precisely for this and should be put to work (P2-2).

### B. Architecture
Appropriate for the scale. What needs work:
1. A typed contract (`RetrievedChunk`, `Answer` dataclasses or pydantic) instead of loose dicts. `bm25_retriever` and `dense_retriever` currently build metadata differently.
2. A `Retriever` protocol so that a reranker, filters or a new index can be added without touching `chain`.
3. Dependency injection of the LLM client and retriever into `answer()`, which makes tests trivial.
4. Ingestion as a pipeline of pure functions: `extract → normalise → split → tag → write`, each unit-tested.
No new frameworks are needed. LangChain/LlamaIndex would add little here.

### C. Correctness
See F-01–F-05, F-08, F-12, F-17.

Edge cases not handled:
- empty query;
- a Chroma collection with fewer than k items;
- the model omitting the `ANSWER:` header (the raw text is shown, including any echoed DISCLAIMER);
- a non-English or Hinglish query (bge-small is English-only).

### D. AI/ML methodology
- **Data:** pypdf extraction loses `₹` and tables (the Rules' tabular forms extract as broken text, e.g. `"Sl.No. Nature of claims Evidence…"`). Running a test extractor such as `pymupdf` or `pdfplumber` on the Act and Rules would be worth measuring. The data is official government material (low licensing risk). Third-party CA books you add may be copyrighted, so record a licence per source in the manifest and do not commit copyrighted PDFs.
- **Chunking:** fixed 1500 characters, ignoring statute structure (F-05). Legal RAG benefits most from section- or sub-section-level units that carry a parent-section header.
- **Retrieval:** sensible, but untuned (k=10/10/5, RRF k=60 are defaults) and unmeasured. Candidate improvements, each to be measured:
  - bge query prefix;
  - cross-encoder reranker (`BAAI/bge-reranker-base`) over the top 20;
  - exact-match boosting of section numbers (query "section 123" → filter `section_ref`);
  - a bigger embedding model (`bge-base`);
  - metadata filters.
- **Evaluation:** absent. This is the single largest credibility gap. Minimum needed:
  - 60–150 gold questions, each with expected section(s), source chunk ids and key facts;
  - retrieval recall@5/10 and MRR for dense / BM25 / hybrid / +rerank;
  - answer correctness (key-fact match) and citation validity;
  - abstention precision on out-of-scope questions;
  - p50/p95 latency and tokens per answer.
  Seed the set from `tests/test_questions.md` and the mapping table.
- **Reliability:**
  - no hallucination check;
  - no confidence gate;
  - no prompt-injection handling — this becomes relevant once users paste arbitrary documents, because retrieved text is inserted verbatim into the prompt;
  - no provider fallback.

### E. Testing
There are none. Highest-value tests to add, in order:
1. RRF unit tests (ordering, dedup, ties).
2. `_parse_response` cases: well-formed, missing headers, `None`, lowercase headers.
3. Tokenizer parity and `₹`/section-token cases.
4. Chunker tests on fixture text: section heading detection and `section_ref` extraction.
5. Tagger regression tests, including the `section 192` case.
6. `chain.answer` with a fake LLM and fake retriever.
7. An index-integrity test: the counts in Chroma, BM25 and the jsonl match, and so do the ids.
8. A Streamlit `AppTest` test for F-01.
9. An eval smoke run (5 questions, retrieval-only, no API key) in CI.

### F. Security, privacy, reliability
- Confirmed: no secrets in tracked files. `.env` is git-ignored, and `.env.example` has an empty key.
- Risks:
  - F-02 (no timeouts or retries), F-11 (unbounded usage), F-15 (pickle);
  - full exception text may expose provider error bodies in the UI;
  - user-uploaded documents (future) need size and type limits and must be treated as untrusted content in prompts.
- The sidebar claims credentials are "never sent to our servers". On a hosted Streamlit deployment the key does pass through the app server, so reword this to be accurate.

### G. Performance & cost
- Measured: warm hybrid retrieval 27–38 ms; cold load 16.8 s; LLM answers 6–16 s.
- The cold load reruns per Streamlit worker process. Use `st.cache_resource`, and stream tokens (`stream=True`) to improve perceived latency.
- Cost per question is not tracked. Log `usage` from the response.
- No distributed or vector-DB scaling is needed at roughly 3k–50k chunks.

### H. Reproducibility / DX
- Setup works once you have the PDFs.
- Problems:
  - no lockfile or Makefile;
  - no single command for ingest → index → eval → app;
  - no data manifest;
  - committed binary indexes (31 MB sqlite) bloat history on every rebuild.

  Decide between committing the artifacts for Streamlit Cloud (the current rationale in `.gitignore:8-9`) and building them in CI or at startup. Document the choice.

### I. Documentation
- Strong rationale table.
- Missing:
  - measured results;
  - screenshots or GIF;
  - limitations section (e.g. "cannot answer slab questions yet");
  - an accurate project tree (the stray `app/indexes`; the "6 test scenarios" are not tests);
  - instructions for adding new documents.

### J. Delivery
- `runtime.txt` targets Streamlit Cloud. There is no evidence of a live deployment.
- No CI, no health or diagnostics page, no structured logging, no Dockerfile.
- For a portfolio, the following are sufficient:
  - CI (lint + tests + retrieval eval);
  - one hosted demo;
  - a Dockerfile.

  Monitoring stacks are not needed.

---

## 8. Prioritized Improvement Roadmap

Effort: S (< ½ day of focused work), M (≈1–2 days), L (several days), XL (a week or more). These are relative sizes, not deadlines.

### Phase 1 — Stabilize (P0)

**P0-1 · Fix example-prompt buttons** · Effort S
- Evidence: F-01.
- Files: `app/streamlit_app.py`.
- Steps:
  1. The buttons set `st.session_state.pending = prompt`.
  2. Compute `prompt = st.chat_input(...) or st.session_state.pop("pending", None)`.
  3. Route through one `handle_prompt()`.
- Acceptance: clicking each of the 6 example buttons renders an assistant answer and a sources expander. A Streamlit `AppTest` test asserts that an assistant message exists after the click.
- Verify: `pytest tests/test_app.py`.
- Value: the first-impression demo path works.

**P0-2 · Robust LLM call with fallback models** · Effort M
- Evidence: F-02.
- Files: `src/llm_client.py`, `src/config.py`, `src/generation/chain.py`, `app/streamlit_app.py`.
- Steps:
  1. Create the client once per (key, url) pair.
  2. `timeout=60`, `max_retries=2`.
  3. Read `FALLBACK_MODELS` as a comma-separated list and try the models in order on `RateLimitError`, `APIStatusError` with status ≥ 500, timeouts, or empty content.
  4. Raise `LLMUnavailable` once all of them fail.
  5. The UI catches it, shows a friendly message, and removes the orphan user turn.
  6. Return `model_used` and `usage`.
- Acceptance: unit tests with a fake client cover success, fallback-after-429, all-fail, and `content=None`. The UI never shows a traceback.
- Verify: `pytest tests/test_llm_client.py`.
- Depends on: P1-3 (test scaffold) — or create the scaffold within this task.

**P0-3 · Root-anchored paths + fail-fast index check** · Effort S
- Evidence: F-03.
- Files: `src/config.py`, `.env.example`, delete `app/indexes/`.
- Steps:
  1. Add `_resolve(p)`, which returns `PROJECT_ROOT / p` when `p` is relative.
  2. On startup, check that the collection exists and that its count equals the number of BM25 ids. If not, show the message "run `make index`".
- Acceptance: `cd app && streamlit run streamlit_app.py` works. A missing index shows the actionable message. `git ls-files app/indexes` is empty.
- Verify: a test that changes the cwd and imports the config.

### Phase 2 — Establish correctness (P1)

**P1-1 · Gold evaluation set** · Effort M
- Steps:
  1. Create `eval/gold.jsonl` with 60+ items: `{id, question, category, expected_sections[], expected_facts[], answerable: bool, gold_chunk_ids[]}`.
  2. Cover all `TOPIC_KEYWORDS` areas, the 6 existing scenarios, 10+ unanswerable or out-of-scope items, 10 old↔new mapping items, and 5 adversarial or injection-style items.
  3. Label `gold_chunk_ids` by searching `chunks.jsonl`.
- Acceptance: the file passes the schema; every answerable item has ≥1 gold chunk id; categories are documented in `eval/README.md`.
- Assumption: you (or a CA) verify the facts against the statute. Mark unverified items.
- Value: the foundation for every claim you make.

**P1-2 · Retrieval and answer evaluation harness** · Effort M
- Depends on: P1-1.
- Files: `eval/run_eval.py`, `eval/results/`.
- Steps:
  1. Retrieval mode (no API key): recall@5/10 and MRR for `dense`, `bm25`, `hybrid`, plus a per-category breakdown.
  2. Generation mode: key-fact match, citation validity (from P2-4), abstention accuracy, latency p50/p95, tokens.
  3. Write a markdown table and JSON with git sha, config and model.
- Acceptance: `python -m eval.run_eval --mode retrieval` is deterministic across two runs, and its results table is committed.
- Value: turns the README's "why hybrid" into evidence.

**P1-3 · Unit-test suite + tooling** · Effort M
- Files: `tests/`, `pyproject.toml` (ruff, pytest, mypy config).
- Tests: the list in §7E, items 1–7.
- Acceptance: `pytest -q` passes offline in under 30 s with no API key; `ruff check .` is clean; the critical modules have meaningful assertions, not just a coverage number.

**P1-4 · Structure-aware statute chunking** · Effort L
- Evidence: F-05.
- Files: `src/ingest/chunk.py`.
- Steps:
  1. Inspect the real heading layout of the Act and the Rules (record samples in a fixture).
  2. Write the heading regex and chapter tracking.
  3. Split sections longer than about 1,500 characters at sub-section `(1)`/`(2)` boundaries, prefixing each part with `"Section 123 — <title>"`.
  4. Keep the page number.
  5. Store `section_ref`, `section_title`, `chapter`, `page`.
- Acceptance:
  - ≥80% of Act chunks have a non-empty `section_ref`;
  - fixture tests pass;
  - the P1-2 recall@5 on section-number questions does not regress, with the target being an improvement — report the measured delta.

**P1-5 · Text normalisation for amounts and symbols** · Effort M
- Evidence: F-04.
- Steps:
  1. Map `` ` `` followed by digits to `₹`.
  2. Turn amounts written in words into numbers, keeping both forms ("fifteen lakh rupees" → append `(₹15,00,000)`).
  3. Clean up `( a)` spacing.
  4. Re-chunk and re-index.
- Acceptance: unit tests on 10 sample phrases; the slab-rate gold questions reach recall@5 > 0. Report the before/after numbers.

**P1-6 · Reproducible corpus manifest & ingest command** · Effort M
- Evidence: F-06.
- Files: `data/sources.yaml`, `src/ingest/download_sources.py`, `Makefile`.
- Steps:
  1. Record URL, sha256, retrieval date, licence and `source_type` per document.
  2. The downloader verifies the hashes.
  3. Provide `make data index eval app`.
  4. Delete the Zone.Identifier files.
- Acceptance: a fresh clone followed by `make data index` reproduces identical chunk ids. The jsonl hash matches, or a documented diff explains any difference.

**P1-7 · Dependency & config hygiene** · Effort S
- Evidence: F-14, F-17.
- Steps:
  1. Remove unused packages.
  2. Pin `langchain-text-splitters`.
  3. Add a lockfile.
  4. Wire `CHUNK_SIZE`/`CHUNK_OVERLAP`, or delete them.
  5. Make the `LLM_MODEL` default `None`, with validation.
  6. Untrack `plan.md` or remove it from `.gitignore`.
  7. Fix the README facts.
- Acceptance: a clean venv install succeeds; `streamlit run` emits no uvicorn warning; `grep` finds no unused config keys.

### Phase 3 — Strengthen the core (P2)

**P2-1 · Retrieval upgrades, each A/B'd** · Effort M
- Depends on: P1-2.
- Variants:
  1. bge query prefix;
  2. cross-encoder rerank of the top 20 → 5;
  3. a section-number exact-match boost;
  4. `bge-base`.
- Acceptance: an eval table with one row per variant. Adopt only variants that improve recall@5 or MRR without more than +300 ms p95. Write up the decision in the README.

**P2-2 · Metadata-aware retrieval & source authority** · Effort M
- Depends on: P1-4, P1-6.
- Steps:
  1. Use `tax_year_scope` and `source_type` (primary / curated / secondary) for filtering or boosting.
  2. When a secondary book conflicts with a primary source, the prompt must prefer the primary and say so.
  3. Fix or remove the topic tagger (F-08).
- Acceptance: gold "conflict" items cite the primary source; the tagger tests pass.

**P2-3 · User document ingestion in the UI** · Effort L
- Implements your stated goal.
- Depends on: P1-4, P1-5, P0-3.
- Steps:
  1. Add an upload page (PDF/MD, with size and page limits).
  2. Extract → chunk → embed → add incrementally to Chroma and BM25.
  3. Tag the upload as `secondary` with its source name.
  4. Show the ingestion report (pages, chunks, warnings).
  5. Allow deleting a document.
- Acceptance:
  - after uploading a fixture PDF, questions about its content retrieve it;
  - deleting it removes all of its ids from both indexes (tested);
  - files over the limit are rejected.

**P2-4 · Citation verification & abstention** · Effort M
- Evidence: F-10.
- Steps:
  1. The prompt requires `[n]` markers.
  2. A post-check confirms that each `[n]` exists and that the cited section numbers and ₹ amounts appear in chunk n.
  3. Flag unsupported sentences in the UI.
  4. Abstain when the top rerank score falls below a threshold chosen on the gold unanswerable set.
- Acceptance: citation-validity and abstention precision/recall are reported in the P1-2 output; Q5-type items abstain.

**P2-5 · Conversation-aware retrieval** · Effort S–M
- Evidence: F-16.
- Steps: condense the last 3 turns plus the new question into a standalone query, with an on/off config switch.
- Acceptance: 10 gold follow-up pairs improve on recall versus no condensation (measured).

**P2-6 · Typed contracts & shared utils** · Effort S
- Evidence: F-12, F-13.
- Steps: dataclasses for chunk and answer; a single tokenizer; `render_sources()`; `st.cache_resource`; replace the pickle (F-15).
- Acceptance: mypy passes on `src/`; the duplicated code is removed; cold start happens once per process (logged).

### Phase 4 — Prepare for delivery (P2/P3)

**P2-7 · CI** · Effort S
- Depends on: P1-3, P1-2.
- GitHub Actions: ruff, mypy, pytest, and the retrieval eval on `main`. Fail if recall@5 drops more than 2 points below the committed baseline.
- Acceptance: a green badge; a deliberately broken RRF fails CI.

**P3-1 · Observability & limits** · Effort S
- Structured JSON logs per query: id, latency per stage, model, tokens, retrieved ids, abstained flag. Query length cap; per-session rate limit.
- Acceptance: one log line per query, verified in a test; limits enforced.

**P3-2 · Packaging & hosted demo** · Effort M
- Dockerfile, a Streamlit Cloud (or HF Spaces) deployment, and a diagnostics sidebar showing index counts, git sha and model.
- Acceptance: a public URL, plus a cold-start → first-answer path that is documented and measured.

### Phase 5 — Elevate the portfolio (P3)

**P3-3 · README rewrite with evidence** · Effort S
- Depends on: P1-2, P2-1.
- Contents: a problem statement, an architecture diagram, the eval results table, an ablation story, a limitations section, a GIF, and "how to add your own books".

**P3-4 · UX polish** · Effort M
- Streamed answers; clickable citations that open the full chunk with page and section; a regime calculator that uses explicit, cited formulas (deterministic code, not the LLM) for the numbers.
- Acceptance: covered by Streamlit `AppTest`, and the calculator's unit tests match hand-computed examples.

### Dependency graph

```
P0-3 ─┬─> P2-3
P1-3 ─┼─> P0-2, P2-7
P1-1 ──> P1-2 ──> P2-1, P2-4, P2-5, P2-7, P3-3
P1-6 ──> P1-4 ──> P1-5 ──> P2-2, P2-3
P0-1, P1-7, P2-6, P3-1 independent
```

---

## 9. Minimum Viable Recruiter-Ready Milestone

**The minimum set:** P0-1, P0-2, P0-3, P1-1, P1-2, P1-3, P1-4, P1-5, P1-6, P1-7, P2-1 (at least the reranker and prefix ablation), P2-4, P2-7, and P3-3.

**Exit criteria:**
- [ ] A fresh clone followed by `make data index test eval app` works without hidden files.
- [ ] `pytest` passes offline; CI is green.
- [ ] An eval table covering 60+ gold questions, with recall@5, MRR, answer key-fact accuracy, citation validity, abstention accuracy and p95 latency, across ≥3 retrieval configurations.
- [ ] The slab-rate, 80C/123, HRA and regime questions answer correctly with statute citations that include the section number.
- [ ] No traceback is reachable from the UI; the example buttons work.
- [ ] The README states measured results and known limitations.

---

## 10. Differentiation and Stretch Goals (ranked by value ÷ effort)

1. **Ablation write-up** (cheap once P1-2 exists): "BM25 vs dense vs hybrid vs +rerank on legal text". This is the strongest interview story.
2. **Citation verification UI** (P2-4): sentence-level support highlighting. It is rare in portfolio RAG projects.
3. **Deterministic tax calculator tool** combined with RAG: the LLM picks the inputs, Python computes the result, and the source sections are cited. This shows an understanding of where LLMs should *not* do arithmetic.
4. **Old↔new section resolver** as a structured lookup (exact-match first, then RAG) built from a verified table.
5. **Error-analysis notebook**: categorise 20 failures (extraction, chunking, retrieval, generation) and show the fixes.
6. **User-library ingestion with source authority and conflict detection** (P2-3/P2-2).
7. Hindi/Hinglish queries using a multilingual embedder, measured on a small set.

**Avoid:**
- agent frameworks or LangGraph;
- a vector-DB migration;
- Kubernetes;
- fine-tuning;
- adding many books before you have an eval;
- dashboards with fabricated metrics;
- more UI themes.

None of these improve evidence of skill.

---

## 11. Recruiter and Technical Interview Assessment

**What it proves today beyond a tutorial:**
- deliberate hybrid retrieval with hand-rolled RRF;
- local embeddings;
- grounded-prompt policy design;
- separation of retrieved evidence from the model's claims;
- a real, timely domain.

**What it does not yet prove:** evaluation competence, robustness, testing discipline, data-pipeline rigour.

**Credible claims after the MVP milestone:**
- "Built a hybrid RAG over the Income-tax Act 2025 with structure-aware chunking."
- "Designed a 60+ question gold set and evaluation harness."
- "Ran retrieval ablations."
- "Added citation verification and calibrated abstention."
- "CI-gated regression eval."

**Unsupported today:**
- any accuracy or recall figure;
- "section-aware chunking";
- "handles complex tax questions";
- "production-ready";
- "deployed";
- "hallucination-free".

**Interviewer's first challenges:**
1. "How do you know hybrid beats dense here?" (There is no data yet.)
2. "Your chunks have no section numbers. How do your citations work?"
3. "Ask it the slab rates." (Fails today.)
4. "What stops it from inventing a section number?"
5. "Why is FastAPI in your requirements?"

**Design decisions to defend:**
- RRF vs weighted score fusion;
- k values;
- chunk granularity for statutes;
- local vs API embeddings;
- abstention threshold selection;
- why there is no query rewriting (and the evidence for that);
- primary vs secondary source precedence.

**Likely questions and follow-ups:**
- Why RRF k=60? What happens at k=1? → Have you tuned it?
- How was the gold set built, and how do you avoid labelling bias? → Who verified the legal facts?
- Recall@5 is X; what are the failure categories? → Which fix moved which category?
- How would you handle a CA book that cites the 1961 Act's limits? → How does the model know which is current?
- What does a query cost, and what is the p95 latency? → How would you cut both by half?
- A user uploads a PDF containing "ignore previous instructions…". What happens?
- How do you detect that the index is stale after the Finance Act 2027?

**Recommended live demo (3 minutes):**
1. Ask the slab question, and show the cited section and page in the source panel.
2. Ask "what is 80C now?" and show the old→new answer.
3. Ask an out-of-scope question and show the abstention.
4. Show an injected or unsupported citation getting flagged.
5. Open the eval table and the CI run.

**Resume bullets (only once implemented; numbers are placeholders until measured):**
- "Built a hybrid (BM25 + bge embeddings, RRF) RAG over the Income-tax Act 2025; a cross-encoder reranker improved recall@5 from **[measured]** to **[measured]** on a 60-question gold set."
- "Implemented statute-aware chunking and citation verification; flagged unsupported claims in **[measured]%** of answers; CI blocks retrieval regressions."

---

## 12. Definition of Done (industry-ready demonstrator)

- [ ] All P0 tasks are closed, with tests.
- [ ] Reproducible pipeline from a clean clone; source manifest with hashes and licences.
- [ ] ≥80% of statute chunks carry `section_ref` and page.
- [ ] Gold set of 60+ items, with documented verification of the facts.
- [ ] A committed eval baseline, plus ≥3 retrieval ablations, with the chosen configuration justified.
- [ ] Citation validity and abstention measured and shown in the UI.
- [ ] Offline unit tests plus a Streamlit `AppTest` test; ruff and mypy clean; CI gate on the eval.
- [ ] LLM timeouts, retries and fallback; no traceback reachable in the UI; input and rate limits.
- [ ] Structured per-query logs including latency and tokens.
- [ ] Hosted demo plus a Dockerfile; diagnostics showing index version and model.
- [ ] README with results, architecture, limitations, how to add documents, and an accurate disclaimer.

---

## 13. Final Verdict

**Competitiveness today:** above a typical tutorial in design intent (hybrid retrieval, grounding policy, evidence panel), but below the shortlist bar. It has no evidence that it works beyond six hand-run questions. Its most basic tax question fails. A reviewer clicking an example prompt gets no answer.

**Most valuable remaining work:**
1. The evaluation harness and gold set.
2. Statute-aware ingestion with amount normalisation.
3. Demo-path robustness.

These three convert the project from "a RAG I built" into "a RAG whose quality I measured and improved", which is the main thing that distinguishes strong early-career AI candidates.

**Portfolio vs production:** completing §9 yields a strong portfolio demonstrator (Level 3, approaching 4 with P2-4/P3-2). A real tax-advice production system would additionally need:
- legally verified content with update processes tied to each Finance Act;
- professional review and liability controls;
- authentication and abuse protection;
- monitoring and incident handling;
- privacy handling for users' financial data.

Those are out of scope for a portfolio and should be stated as limitations, not claimed.

---

# Appendix: Post-audit implementation log (2026-10-16)

> Everything above this line is the **original audit** of commit `872300a` and is left unchanged. That includes its scores and the stale line references to the old code. This appendix records what was implemented afterwards, against the roadmap in §8. All numbers below come from commands that were actually run. Nothing has been committed.

## Roadmap status

| Task | Status | Evidence |
|---|---|---|
| P0-1 Example buttons | **Done** | One `handle_prompt()` path (`app/streamlit_app.py`); `tests/test_app.py::test_example_button_produces_assistant_message` |
| P0-2 LLM robustness | **Done** | `src/llm_client.py`: 60 s timeout, 2 retries, `FALLBACK_MODELS`, `LLMUnavailableError`, empty/None content handled; UI shows a friendly error and removes the orphan turn; `tests/test_llm_client.py`, `tests/test_chain.py` |
| P0-3 Root-anchored paths | **Done** | `config.resolve_path`, `check_indexes()`; stray `app/indexes/` removed and git-ignored; `tests/test_config.py` (changes cwd) |
| F-11 Query cap | **Done** | `MAX_QUERY_CHARS=2000` in chain + UI; `test_overlong_query_rejected` |
| P1-1 Gold set | **Done (proxy)** | `eval/gold.jsonl`: 47 items (42 answerable, 5 out-of-scope); all `verified:false`; 11 loose keyword sets tightened **before** measuring changes |
| P1-2 Eval harness | **Done** | `eval/run_eval.py` (recall@5/10, MRR@10, latency, per-category), `eval/run_generation.py`; deterministic (two runs identical apart from latency) |
| P1-3 Tests + tooling | **Done** | 73 pytest tests pass offline; `pyproject.toml` (pytest, ruff); `ruff check .` clean |
| P1-4 Structure-aware chunking | **Partial** | Monotonic Section/Rule/Schedule tracking + page tracking (`src/ingest/chunk.py`, `tests/test_chunk.py`). Applied to the existing chunks: **100% of Act/Rules chunks now have `section_ref`** (was 0). Re-chunking from the PDFs (page numbers) has not been run, because the PDFs are not on this machine |
| P1-5 Normalisation | **Done** | `src/text_utils.py`: backtick→₹ (208 chunks now contain ₹, 0 leftover backticks), amounts in words annotated (180), clause spacing; `tests/test_text_utils.py` |
| P1-6 Reproducible corpus | **Partial** | `data/sources.json` registry + `download_sources.py` (status/download/`--record-hashes`, non-zero exit on problems), `Makefile`; Zone.Identifier files removed. **sha256 values are still null** (need the PDFs); 4 of 6 URLs are portal pages, not direct links |
| P1-7 Dependency hygiene | **Done** | `requirements.txt` trimmed to direct imports and pinned to the tested versions; the uvicorn/fastapi/langchain/bs4/requests pins are gone; `requirements-dev.txt` |
| P2-1 Retrieval ablation | **Partial** | bge query prefix measured and **rejected** (dense recall@5 0.738→0.714). Reranker not attempted |
| P2-2 Source authority | **Partial** | Mapping table now tagged `curated` (F-09). Tagger boundary bugs fixed (F-08), but the tags are still weak (65% `general`) and not used for ranking |
| P2-4 Citation verification | **Done (deterministic part)** | `src/generation/citations.py` + `[n]` prompt rule + UI warnings; `tests/test_citations.py`, `test_citation_problems_shown_as_warning`. Abstention threshold not implemented |
| P2-6 Typed contracts / shared utils | **Partial** | Single tokenizer (F-12), `render_assistant_details()` dedup + model sources kept in history (F-13), JSON BM25 index instead of pickle (F-15). No dataclasses |
| P2-7 CI | **Written, not run on GitHub** | `.github/workflows/ci.yml`: ruff, pytest, retrieval eval, gate on a hybrid recall@5 drop > 0.02 vs `post_changes.json`. YAML parses and the gate script passed locally; a real Actions run is unverified |
| P3-3 README | **Done** | Rewritten with measured results and limitations |
| P2-3 Upload UI, P2-5 conversation memory, P3-1 observability, P3-2 Docker/hosting, P3-4 UX | **Not started** | |

## Deviations from the roadmap

- **Exact dense search is now the production default.** I found this while building the eval: Chroma's HNSW returned different top-10s between process runs (7 of 42 questions) and differed from exact search on 37 of 42. The app would therefore not have behaved like the measured system. `DENSE_BACKEND=exact` (cosine over all stored vectors) fixes that; `hnsw` stays available for large corpora.
- **Chunks were upgraded in place, not re-chunked**, because the PDFs are absent (user decision). `make upgrade-chunks` is idempotent; `make chunk` (with PDFs) is the full path.
- **Index rebuild now recreates the Chroma directory.** `delete_collection()` left the old HNSW segment behind and grew the sqlite file to 58 MB; a clean rebuild is 33 MB.
- **`data/sources.json` instead of YAML**, so no new dependency is needed.
- **`CHUNK_SIZE`/`CHUNK_OVERLAP` redefined as characters (1500/150)**, matching what the chunker always did, and now wired in.
- **Workers:** the teammate subagents in this environment were not allowed to run Python. The lead agent therefore implemented B3–B5 and ran every validation command.

## Measured results

Retrieval, hybrid, exact dense backend, 42 answerable gold items (`eval/results/`):

| run | recall@5 | recall@10 | MRR@10 |
|---|---|---|---|
| `baseline_pre_changes` (original indexes) | 0.809 | 0.905 | 0.688 |
| `post_changes` (normalised, sectioned, deduped) | 0.809 | 0.881 | 0.686 |

The ingestion changes were **retrieval-neutral** on this proxy set. The gains are qualitative: correct section citations, ₹ restored, duplicates removed, and no pickle.

Generation (`gen_smoke_post_changes.json`, 6 calls on `upstage/solar-mini4`):
- Q2–Q6 meet the `tests/test_questions.md` criteria on manual review.
- **Q1 (slab rates) still fails at retrieval.** The model correctly reports the information as missing.
- The citation check flagged 3 claims, and on inspection the model had indeed cited a block that did not contain them.
- One false positive I found during development ("yea**rs** 8" parsed as an amount) and one false negative ("Rs 8 crore") were fixed, with regression tests.

Clean environment: a fresh venv from `requirements-dev.txt` (CPU torch) passed `ruff`, all tests, and reproduced hybrid recall@5 0.8095 exactly.

## Remaining issues (highest value first)

1. **Slab-rate retrieval (`q1_slabs`)** fails because of a vocabulary gap. Candidates, each to be measured: a reranker, a query-expansion map ("slab" → "rate of tax", "new regime" → "section 202"), or a curated "rates & limits" document with citations.
2. **Weakest categories:** `tds` and `salary` (recall@5 0.33 and 0.50).
3. **The gold set is a keyword proxy**, and no item is professionally verified.
4. **Re-chunk from the PDFs** to get page numbers, then record sha256 values (`python -m src.ingest.download_sources --record-hashes`).
5. **CI** has never run on GitHub.
6. **Not started:** the upload UI for user-added books, an abstention threshold, conversation memory, and structured logging.

## Independent review (verifier teammate, read-only) and fixes

The verifier could not execute code in this environment and reviewed the diff statically. It confirmed:
- the README and appendix numbers match `eval/results/*`;
- no secrets appear in any tracked or new file.

| Finding | Severity | Fix | Regression test |
|---|---|---|---|
| A wrong API key (`AuthenticationError`) or a 4xx error escaped the UI handler as a raw traceback | Medium | `LLMConfigError` for auth/permission errors (fails fast; fallbacks don't help with the same key); other `APIStatusError`s fall back to the next model | `test_bad_key_raises_config_error_without_trying_fallbacks`, `test_other_status_errors_fall_back`, `test_bad_api_key_shows_friendly_error` |
| Section tracker rejected lettered sections (80 → 80A) and could freeze after a jump larger than the gap | Medium | Accepts same-number lettered headings; resyncs after two consecutive far headings. On the real corpus this changed exactly 1 ref (Section 354 → 354A, a real heading); anchors unchanged | `test_lettered_section_after_base_accepted`, `test_resyncs_after_a_large_gap` |
| The citation sentence splitter broke "Rs. 8 crore" / "Sec. 80C", so the amount went unchecked | Medium | Abbreviation-aware split | `test_abbreviation_does_not_split_sentence` |
| "Section 80" was treated as supported by "80,000" | Low | Section numbers no longer match inside amounts | `test_section_number_not_matched_inside_amount` |
| "No model configured" error message gave no hint | Low | Message names `LLM_MODEL` | `test_no_model_message_is_actionable` |
| Rounding 0.809/0.810 and HNSW 7 vs 11 of 42 inconsistent | Low | Docs use the generated 0.809; HNSW counts attributed to the original vs rebuilt indexes | — |
| Zero-norm vector / empty corpus division | Low | Not fixed (cannot occur with the current non-empty corpus and bge embeddings) | — |

After the fixes: **73 tests pass**, `ruff check .` is clean, and the retrieval eval is unchanged (hybrid 0.809 / 0.881 / 0.686).

---

# Appendix: Second implementation session (2026-10-16, HANDOFF §2–4)

All numbers below come from commands run in this session. Nothing has been committed. Team note: worker subagents again could not run shell commands, so the lead session did all the implementation and validation.

## Step 0 — baseline reproduced
- The venv lacked the dev tools; installed them with `pip install -r requirements-dev.txt`.
- `73 passed`, ruff clean, `make eval` reproduced hybrid 0.809 / 0.881 / 0.686 exactly.

## Step 1 — full re-chunk from the PDFs (HANDOFF §3)
- `download_sources`: 0 problems; `--record-hashes` wrote the sha256 values into `data/sources.json`.
- `make chunk`:
  - 2,743 chunks (was 2,748);
  - Act 1467/1467 and Rules 1096/1096 chunks have a `section_ref`;
  - **2,550/2,550 Act+Rules chunks have an integer `page`**.
- Anchors:
  - `Upto ₹400000 Nil` → Section 202 (p. 265);
  - `does not exceed ₹150000` + Schedule XV → Section 123. The chunk that runs into the 123 heading is tagged 122, which is correct for its start.
  - The last Act section is 536.
- `post_rechunk`: hybrid unchanged (0.809 / 0.881 / 0.686); dense recall@5 fell from 0.738 to 0.714.

## Startup model probe (user request)
- `llm_client.probe_models()` sends a 5-token request to each candidate.
  - Auth errors stop the probe immediately.
  - Statuses are ok / error / untested.
  - The returned order is working models first, then untested, then failed.
- The app probes once per (key, URL, model) at session start, before chat is enabled:
  - the sidebar shows the model in use, a per-model status expander and a "Re-check models" button;
  - chat is blocked if no model works;
  - answers use the probed order as fallbacks.
- The app now defaults the key and URL to `.env`; they used to be blank.
- Live run (10 candidates): full scan 5.4 s; 7 OK, 3 rate-limited (`gemma-4-31b/26b :free`, `ling-3.0-flash-vl`).
- **Found:** `LLM_API_KEY` in `.env` has a leading label before the key, so OpenRouter rejects it ("Missing Authentication header"). For the live runs this session the label was stripped in-process only; **`.env` itself needs fixing**.

## N1 — slab retrieval and choosing the retrieval default
Error analysis: the rate table ranked 1,393 in BM25 and 95 in dense. Changes measured, in order (hybrid recall@5 / recall@10 / MRR@10):

| change | result | kept? |
|---|---|---|
| Section-title header (`Section 202 — New tax regime…`) in both indexes | 0.810 / 0.881 / 0.656; dense worse | no |
| Header in BM25 only | 0.833 / 0.857 / 0.687 | no (configurable, off) |
| + BM25 query-expansion map (`src/retrieval/query_expansion.py`) | 0.857 / 0.881 / 0.702 | expansion yes |
| Expansion, no header, weighted RRF 1.5 | 0.905 / 0.976 / 0.718 | yes |
| + stopwords in the shared tokenizer | **0.952 / 0.952 / 0.749** | **yes → `n1_final`** |
| + LLM keyword rewrite (OpenRouter, 42 calls) | 0.929 / 0.976 / 0.718 | no (flag, off) |

- **Default retrieval choice:** hybrid with weighted RRF (dense 1.0, BM25 1.5) beats BM25 alone on recall@5 (0.952 vs 0.905), with equal recall@10 (0.952) and practically equal MRR@10 (0.749 vs 0.750). Weights from 1.5 to 3 tie.
- **Caveat:** the configuration was chosen on the gold set itself.
- The section titles (`section_title`: 420/431 Act units including schedules, 281/288 Rules) are kept as metadata.
- The CI gate now compares against `eval/results/n1_final.json`.
- **N1 acceptance not met:**
  - `q1_slabs` still misses. The Section 202 heading chunk ranks 6, and the rate table is the next chunk, but the mapping table and FAQ outrank it.
  - `gen_smoke_n1` Q1 still reports the rates as missing; the citation check flags 7 unsupported claims.
  - Follow-up idea, not built: follow "section N of the Act, 2025" references from top chunks. The FAQ mixes 1961 and 2025 numbering, so this needs care.

## N2 — tds / salary
- Fixed by the N1 changes:
  - salary recall@5 0.50 → 1.00 (`ded_std` now hits);
  - tds 0.33 → 0.67 (`tds_salary` hits);
  - `penalty_late` hits.
- Remaining miss: `tds_rent`. The Rent row of the Section 393 TDS table is outranked by FAQ text about TDS in general. This is a retrieval problem, not a gold problem; the gold set was not edited.

## N3 — upload UI
- New overlay module `src/ingest/user_docs.py`; the base corpus and committed indexes are untouched.
- **Validation:** file type pdf/md/txt, size ≤ `MAX_UPLOAD_MB`, PDF pages ≤ `MAX_UPLOAD_PAGES`, licence required, safe filename, no duplicate names.
- **Pipeline:** `chunk_document` → tag `supplementary` → embed with the local bge model → stored in `data/user/` (git-ignored).
- Both retrievers merge the overlay; caches reset after each add or delete.
- `app/pages/1_Documents.py` provides upload, list and delete.
- **Deviation from HANDOFF:** licences go in `data/user/registry.json`, not `data/sources.json`. Putting them in `sources.json` would make `download_sources` fail on machines that don't have the files.
- **Tests:** 11 tests using a PDF generated in code. A test PDF is retrieved by both BM25 and dense, and after delete its ids are gone from both; the page AppTest covers list and delete.
- **End-to-end with the real embedder:** a circular uploaded under a new name appeared in the hybrid top 5, and after deletion 0 user chunks remained.

## N4 — abstention
- If the best dense cosine is below `ABSTAIN_DENSE_THRESHOLD=0.68`, the app returns a "not found" answer without an LLM call.
- On the gold set it abstains on 3 of 5 out-of-scope questions and 0 of 42 answerable ones. The margin is thin (lowest answerable 0.694), and the threshold was chosen on the same set.
- BM25 scores did not separate the two groups.

## N5 — smaller items
- **Conversation memory** (`src/generation/memory.py`): behind `CONVERSATION_MEMORY` (off), condenses the last 3 turns; the app passes its history. Not measured.
- **Structured log** (`src/query_log.py`) → `logs/queries.jsonl` (git-ignored; tests disable it):
  - per-stage ms, model, usage, chunk ids, abstention, confidence, citation result.
- **Types:** `src/types.py` TypedDicts; `answer()` is annotated `-> Answer`.
- Topic tags removed from the source labels in the UI.
- `Dockerfile` + `.dockerignore`: CPU torch, model baked in, non-root user, healthcheck, logging off. **Not built**, because the Docker daemon was not running.
- No hosted demo (outward-facing; not done).

## State at end of session
- `113 passed`, ruff clean.
- Retrieval: hybrid 0.952 / 0.952 / 0.749 (`n1_final`).
- Generation smoke: Q2–Q6 meet the criteria; Q1 fails at retrieval.
