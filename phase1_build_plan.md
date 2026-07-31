# Indian Tax Advisor RAG — Phase 1 Build Plan (Ground-Up, In-Depth)

## 1. Verified data sources to download into `data/raw/`

Download these **as files** (PDF/HTML saved locally) rather than relying
on live scraping at query time — RAG should run offline against a fixed,
versioned snapshot so answers are reproducible and you can explain
exactly what's in the index during an interview.

### Primary (official) — tag `source_type: primary`

| #   | Document                                                                                                                                                   | URL                                                                                                                                                                            | Save as                                            |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------- |
| 1   | Income-tax Act, 2025 (as amended by Finance Act 2026) — full text                                                                                          | `https://www.incometaxindia.gov.in/documents/d/guest/income_tax_act_2025_as_amended_by_fa_act_2026-pdf`                                                                        | `data/raw/act_2025_full_text.pdf`                  |
| 2   | Press release: "Income-tax Act, 2025 comes into force from 1 April 2026" (plain-language summary of what changed, Tax Year concept, Rules 2026)            | `https://www.incometaxindia.gov.in/documents/d/guest/press-release-income-tax-act-2025-comes-into-force-from-01-04-2026-pdf`                                                   | `data/raw/press_release_act_2025_commencement.pdf` |
| 3   | Income-tax Rules, 2026 (operationalizes the Act — notified 20 Mar 2026)                                                                                    | via `https://www.incometaxindia.gov.in/cbdt/` → Rules → Income-tax Rules, 2026 (get the direct PDF link when you visit; the landing page rotates document IDs)                 | `data/raw/income_tax_rules_2026.pdf`               |
| 4   | Official FAQs as per Income-tax Act, 2025                                                                                                                  | via `https://www.incometax.gov.in/` → footer/help section "FAQs as per Income-tax Act, 2025"                                                                                   | `data/raw/faq_act_2025.pdf` (or HTML)              |
| 5   | Official FAQs on Interplay/Transition from 1961 Act to 2025 Act — **this is the single best source for the old/new-numbering + transition-year confusion** | same portal, "FAQs on Interplay and Transition from the Income Tax Act, 1961 to the Income Tax Act, 2025"                                                                      | `data/raw/faq_transition_1961_to_2025.pdf`         |
| 6   | General FAQs, FAQs for Senior Citizens, FAQs on house property, FAQs on filing returns, FAQs on TDS (all official, same portal)                            | same portal footer section                                                                                                                                                     | `data/raw/faq_*.pdf` (one per topic)               |
| 7   | CBDT Circulars & Notifications relevant to FY 2026-27 (filter for the ones dated 2026, on rebate/slab/rules clarifications)                                | `https://www.incometaxindia.gov.in/circulars` and `.../notifications` — pick the specific circulars relevant to income tax rates/deductions, not every administrative circular | `data/raw/circulars/*.pdf`                         |

### Section-mapping reference (small, structured, high-value)

Build `data/raw/section_mapping_1961_to_2025.md` yourself as a short
structured table (the one in §0, expanded with every section a personal-
tax question is likely to touch: 80C→123, 80CCD(1B)→124, 80D→126,
24(b) home loan interest, HRA provisions, 80TTA/80TTB→153, 87A→202,
115BAC→202). Cross-check entries against at least two independent trackers
before finalizing (e.g. ClearTax's mapping guide and a second independent
one) so a single source's error doesn't propagate. This file becomes one
of the most-retrieved documents in the whole corpus — spend real care on
it.

### Supplementary (secondary/explainer) — tag `source_type: supplementary`

Use 2-3 of these, clearly tagged, for plain-language context ChromaDB can
retrieve alongside the primary legal text (a user asking "how does HRA
work" benefits from both the rule and a worked example):

- ClearTax's Income Tax Act 2025 explainer + section-mapping guide
- Kotak Life or similar deduction-list guide (80C–80U) for FY 2026-27
- One HRA-specific explainer with a worked calculation example

Every supplementary chunk must retain its `source_type: supplementary`
metadata and, ideally, a line in the generation prompt that treats primary
sources as authoritative when the two disagree.

---

## 2. Hybrid search design (BM25 + dense, kept simple and explainable)

**Goal**: get the benefits of "hybrid search" and "latest RAG techniques"
without ending up with retrieval logic you can't explain line-by-line in
an interview. Concretely:

- **Dense retrieval**: ChromaDB with a **local, open-source sentence
  embedding model** (`BAAI/bge-small-en-v1.5` via `sentence-transformers`
  — small, fast on CPU, no API key/cost, and deterministic, which matters
  for a demo you'll re-run live). This is a deliberate choice over calling
  an embeddings API through OpenRouter: keeps retrieval free, offline-
  capable, and not dependent on which LLM provider is configured that day.
- **Sparse retrieval**: `rank_bm25`'s `BM25Okapi` over the same chunked
  corpus. Tax text is full of exact terms (section numbers, ₹ amounts,
  "80C", "HRA") that BM25 is _better_ at than embeddings — this is the
  actual reason to use hybrid here, not just because it's trendy, and
  that's a good sentence to have ready for an interview.
- **Fusion**: **Reciprocal Rank Fusion (RRF)**, implemented by hand in
  ~20 lines — no black-box `EnsembleRetriever` magic. For each query, get
  ranked results from both retrievers, then score each unique chunk as
  `sum(1 / (k + rank))` across whichever list(s) it appears in (k=60 is
  the standard constant from the original RRF paper). Sort by that score,
  take top-N. This is simple enough to whiteboard and explain, and it's
  what "hybrid search" concretely means in production RAG systems.
- **Metadata filtering** as a first-class retrieval step: every chunk
  carries `source_type` (primary/supplementary), `topic` (slab_rates,
  80C, 80D, HRA, regime_comparison, etc.), and `tax_year_scope`. Simple
  keyword-based topic tagging at ingestion time (not a separate ML
  classifier) is enough — keep it inspectable.
- **Optional stretch** (only if time allows, keep as a clearly separated,
  toggleable step so the "simple core" isn't disturbed): a cross-encoder
  reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) on the fused top-20 to
  produce the final top-5. This is worth mentioning as "the next thing I'd
  add" even if not built, since re-ranking is the other standard piece of
  a modern hybrid pipeline.

Explicitly **not** doing (and why, so you have the answer ready if asked):
no query rewriting/expansion agent, no multi-hop retrieval, no HyDE — all
add real complexity for a domain (India personal tax) where queries are
usually direct and don't need query transformation. Naming what you left
out on purpose is more impressive in an interview than silently not
knowing about it.

---

## 3. Directory structure

```
indian-tax-advisor/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   ├── raw/                      # downloaded source docs (PDF/HTML)
│   │   └── circulars/
│   └── processed/
│       └── chunks.jsonl          # chunked + metadata-tagged, pre-embedding
├── src/
│   ├── ingest/
│   │   ├── download_sources.py   # fetches/organizes data/raw
│   │   ├── chunk.py              # section/clause-aware chunking
│   │   ├── tag_metadata.py       # source_type, topic, tax_year_scope
│   │   └── build_index.py        # embeds + writes to ChromaDB, builds BM25 index
│   ├── retrieval/
│   │   ├── bm25_retriever.py     # thin wrapper over rank_bm25
│   │   ├── dense_retriever.py    # thin wrapper over ChromaDB query
│   │   └── hybrid.py             # RRF fusion — the ~20-line function
│   ├── generation/
│   │   ├── prompt_templates.py   # system prompt, disclaimer, citation format
│   │   └── chain.py              # retrieve -> build context -> call LLM
│   ├── config.py                 # loads .env, model names, chunk size, k values
│   └── llm_client.py             # thin wrapper over OpenRouter/NIM (existing pattern)
├── app/
│   └── streamlit_app.py          # Phase 1 chat UI
├── indexes/
│   ├── chroma/                   # persisted ChromaDB
│   └── bm25.pkl                  # persisted BM25 index (pickle of tokenized corpus)
├── tests/
│   └── test_questions.md         # the 5+ test questions + expected behavior notes
└── BLOCKED.md                    # only if something can't be finished
```

Keep `src/` as plain, readable Python — no clever abstractions, no
inheritance hierarchies. Every file should be short enough to read top to
bottom in under two minutes; that's the actual design constraint given
"tweakable, quick to modify, explainable in an interview."

---

## 4. Environment & dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

`requirements.txt`:

```
langchain==0.3.*
langchain-community==0.3.*
chromadb==0.5.*
sentence-transformers==3.*
rank_bm25==0.2.*
pypdf==5.*
beautifulsoup4==4.*
requests==2.*
streamlit==1.*
fastapi==0.115.*
uvicorn==0.30.*
python-dotenv==1.*
openai==1.*            # OpenAI-compatible client works for OpenRouter/NVIDIA NIM
tiktoken==0.7.*
```

`.env.example`:

```
LLM_API_KEY=
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=your-chosen-model
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
CHROMA_PERSIST_DIR=indexes/chroma
BM25_INDEX_PATH=indexes/bm25.pkl
```

Note: embeddings run **locally** via `sentence-transformers` (no API
key needed for embeddings), only the generation call goes through the
configured OpenRouter/NIM endpoint. This is worth a one-line comment in
`config.py` since it's a deliberate architectural choice, not an oversight.

---

## 5. Build steps (in order)

### Step 1 — Project scaffold

Create the directory structure above, venv, `requirements.txt`, `.env`
from `.env.example`, `.gitignore` (include `venv/`, `.env`, `indexes/`,
`data/raw/*.pdf` if you don't want large binaries in git — but do keep
`data/processed/chunks.jsonl` since that's small and worth version-
controlling for reproducibility).

### Step 2 — Source and download data

Implement `src/ingest/download_sources.py`:

- Navigate from the two stable entry points (`incometaxindia.gov.in/cbdt/`
  for Act/Rules, `incometax.gov.in` footer for FAQs) rather than
  hardcoding today's PDF URLs.
- Save every file into `data/raw/` with clear, descriptive filenames
  (matching the table in §1).
- Hand-write `data/raw/section_mapping_1961_to_2025.md` per §1, cross-
  checked against two sources.
- Log exactly what was downloaded, from where, and when into
  `data/raw/SOURCES.md` (URL + retrieval date per file) — this becomes
  part of the README's "sources used" section later, and matters if a
  question comes up about how you verified currency of the data.

### Step 3 — Chunking

Implement `src/ingest/chunk.py`:

- Parse PDFs with `pypdf`; parse any HTML with `BeautifulSoup`.
- Chunk on section/clause boundaries where the source has clear numbering
  (e.g. split the Act text on `Section N.` boundaries, not fixed
  token windows) — this directly serves the "don't chunk mid-clause"
  requirement and is easy to demo (show a chunk, show it's one complete
  section).
- Where a document doesn't have clean structural markers (e.g. an FAQ
  page), fall back to a recursive character splitter (~500 tokens,
  ~50 token overlap) via `langchain_text_splitters.RecursiveCharacterTextSplitter`.
- Output `data/processed/chunks.jsonl`, one JSON object per chunk:
  `{"id", "text", "source_doc", "source_type", "topic", "tax_year_scope", "section_ref"}`.

### Step 4 — Metadata tagging

Implement `src/ingest/tag_metadata.py`:

- Simple keyword-based topic tagging (dict of topic → keyword list is
  fine — e.g. `"80C" / "123" / "Schedule XV"` → topic `deductions_80c`).
  Resist the urge to use an LLM call to tag every chunk; keyword tagging
  is transparent, free, and good enough at this corpus size, and you can
  say exactly why in an interview.
- `tax_year_scope` is mostly `"FY2026-27_AY2027-28"` (new Act) except the
  transition FAQ and any explicitly old-Act content, which gets
  `"FY2025-26_AY2026-27_or_earlier"`.

### Step 5 — Build the indexes

Implement `src/ingest/build_index.py`:

- Embed all chunks with `sentence-transformers` (`BAAI/bge-small-en-v1.5`),
  write to a persisted ChromaDB collection (`indexes/chroma/`), storing
  the full metadata dict alongside each vector.
- Tokenize all chunk texts and build a `BM25Okapi` index; pickle it to
  `indexes/bm25.pkl` alongside a parallel list of chunk IDs (BM25 doesn't
  store metadata itself, so keep the ID-to-metadata mapping in the same
  pickle or a sidecar JSON).
- Make this script idempotent/rerunnable — you'll re-run it whenever
  `data/raw/` changes.

### Step 6 — Retrieval layer

- `src/retrieval/dense_retriever.py`: given a query string, embed it,
  query ChromaDB for top-k (k≈10), return `[(chunk_id, rank, score, metadata)]`.
- `src/retrieval/bm25_retriever.py`: given a query string, tokenize,
  score against the BM25 index, return top-k similarly ranked.
- `src/retrieval/hybrid.py`: the RRF function from §2 — takes both ranked
  lists, returns a single fused, deduplicated top-N (N≈5) list of full
  chunk objects (text + metadata) ready to go into the prompt. Write this
  function with a docstring that states the RRF formula explicitly —
  this is the piece most worth being able to explain from memory.

### Step 7 — Generation chain + prompt design

`src/generation/prompt_templates.py`:

- System prompt states: scope is FY 2026-27 / AY 2027-28 under the
  Income-tax Act, 2025; always ground numeric/legal claims in the
  provided context, never invent a section number or ₹ figure; if the
  retrieved context doesn't cover the question, say so rather than
  guessing; never give a single "you should do X" recommendation for
  regime choice or tax-saving — lay out the relevant numbers/options and
  let the user's own figures decide; every answer must end with a
  "Sources:" section listing the retrieved documents actually used, and
  the standing disclaimer (not a CA, general guidance only).
- Structure the output format explicitly in the prompt (e.g. `ANSWER:`
  then `SOURCES:` then the disclaimer line) so it's mechanically parsable
  by the UI layer — but **also** independently show the actual retrieved
  chunks/metadata in the Streamlit UI regardless of what the model claims
  it cited. Don't rely solely on the LLM to self-report which sources it
  used — surface the ground truth (what was actually retrieved and fed to
  it) alongside the model's own "Sources:" line. This is a meaningfully
  more trustworthy design than trusting free-text citations, and it's a
  good thing to point out unprompted in an interview.

`src/generation/chain.py`:

- `answer(query) -> {"answer": str, "sources": list, "disclaimer": str}`.
- Retrieve via `hybrid.py` → build a context block (chunk text + its
  metadata, e.g. `[Source: Income-tax Act 2025, Section 123]`) → format
  the full prompt → call the LLM via `llm_client.py` → parse the
  structured response → return.

### Step 8 — Streamlit chat UI

`app/streamlit_app.py`:

- Standard chat loop: message history in `st.session_state`, `st.chat_input`,
  `st.chat_message` for turns.
- For each assistant turn: show the answer text, then an expander showing
  the actual retrieved chunks (source doc, section, source_type) used for
  grounding, then the disclaimer footer.
- Keep this file free of business logic — it only calls
  `src.generation.chain.answer()` and renders the result.

### Step 9 — Test against required questions

Create `tests/test_questions.md` with at least these 5, run manually
(and note actual output/behavior next to each, including retrieved
sources):

1. What are the new tax regime slab rates for FY 2026-27?
2. How much can I deduct under Section 123 (old 80C) and what qualifies?
3. How does HRA exemption work?
4. Should I pick the old or new tax regime? (must confirm it lays out
   numbers/options, doesn't give a single confident recommendation)
5. A deliberately ambiguous/out-of-scope question (e.g. something about
   FY2025-26/old Act, or something the corpus genuinely doesn't cover) —
   confirm the system says so rather than confidently guessing.
6. (Bonus, given the new numbering) "What is Section 80C called now?" —
   good specific test of the section-mapping document actually being
   retrieved and used correctly.
