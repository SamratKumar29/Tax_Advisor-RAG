# HANDOFF — Indian Tax Advisor RAG

Continuation notes for a new coding session, for example on the laptop that has the raw PDFs. Read this first, then `report.md`: the audit is §1–13, and what has been done since is in **"Appendix: Post-audit implementation log"**. `README.md` describes the current system and `eval/README.md` describes the evaluation.

Written 2026-10-16. The work is in commit `90210a5` (on top of the audited `872300a`).

---

## 1. State at handoff

| Item | Status (all verified by commands actually run) |
|---|---|
| Tests | `73 passed` (`python -m pytest -q -o addopts=""`), offline, no API key needed |
| Lint | `ruff check .` clean |
| Clean install | Fresh venv + CPU torch + `requirements-dev.txt` → ruff, tests and eval all pass and reproduce the numbers below |
| Retrieval eval (hybrid, app mode) | recall@5 **0.809**, recall@10 **0.881**, MRR@10 **0.686** (`eval/results/post_changes.*`, also the CI baseline) |
| Generation smoke run | 6 questions on `upstage/solar-mini4`: Q2–Q6 meet the criteria; **Q1 (slab rates) fails at retrieval** (`eval/results/gen_smoke_post_changes.json`) |
| App | Starts headless, `/_stcore/health` = ok; a real answer works with the working directory set to `app/` |
| CI | `.github/workflows/ci.yml` is written; YAML parses and the gate passes locally. **It has never run on GitHub.** |
| Corpus | 2,748 chunks; 100% of Act/Rules chunks have a `section_ref`; **no page numbers** (the chunks were upgraded without the PDFs) |

---

## 2. Step 0: get the code onto the laptop

The implementation is committed as **`90210a5` ("tweaks")**. Only this `HANDOFF.md` was left uncommitted at handoff. Commit and push it, then on the laptop pull. Check that `.env`, `venv/` and `.cortex/` are not in the repo (all three are git-ignored).

Then, on the laptop:

```bash
git pull
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt          # if torch is large: pip install torch --index-url https://download.pytorch.org/whl/cpu  first
cp .env.example .env                         # fill LLM_API_KEY, LLM_MODEL, optional FALLBACK_MODELS
make test lint                               # expect: 73 passed, All checks passed!
make eval                                    # expect hybrid 0.809 / 0.881 / 0.686 (eval/results/latest.md)
```

If those numbers don't reproduce, stop and investigate before changing anything else.

---

## 3. Step 1 (needs the PDFs): full re-chunk with page numbers

Put the files in place exactly as listed in `data/sources.json`:

```
data/raw/act_2025_full_text.pdf
data/raw/income_tax_rules_2026.pdf
data/raw/faq_transition_1961_to_2025.pdf
data/raw/press_release_act_2025_commencement.pdf
data/raw/circulars/Circular_No_5_2026.pdf
data/raw/circulars/Circular-No-06-2026.pdf
data/raw/section_mapping_1961_to_2025.md     (already in git)
```

Then:

```bash
python -m src.ingest.download_sources                    # expect: 0 missing
python -m src.ingest.download_sources --record-hashes    # writes sha256 into data/sources.json — commit it
cp data/processed/chunks.jsonl /tmp/chunks_before.jsonl  # keep for comparison
make chunk                                               # re-chunks from PDFs (pages + section refs)
make index
make test
make eval
```

**Acceptance checks:**
- [ ] `make chunk` prints `(N with section_ref)` close to N for both the Act and the Rules.
- [ ] Every Act/Rules chunk has an integer `page`. Check with:
  ```bash
  python -c "import json;r=[json.loads(l) for l in open('data/processed/chunks.jsonl')];print(sum('page' in x for x in r),len(r))"
  ```
- [ ] Anchor 1: the chunk containing `Upto ₹400000 Nil` has `section_ref == "Section 202"`.
- [ ] Anchor 2: the chunk containing `does not exceed ₹150000` near `Schedule XV` is in Section 123.
- [ ] Anchor 3: the last Act section is 536.
- [ ] All tests pass. `test_index_integrity.py` checks the manifest hash and that the ids match across both indexes.
- [ ] `make eval`: hybrid recall@5 is not below **0.789** (the CI gate is a drop of more than 0.02).

**Expected differences:**
- **Chunk ids will change.** The ids hash the chunk index and text, and both change when pypdf output is normalised before splitting. The gold set does not use chunk ids, so the eval still works.
- **Counts may differ slightly** from 2,748.
- **If recall moves**, record it as a new result (`--name post_rechunk`) and update `eval/results/post_changes.*` only if you intend to make it the new CI baseline.

**If the section tracker misbehaves on fresh pypdf output:** the logic is `assign_section_refs` in `src/ingest/chunk.py`. It works by tracking monotonically increasing headings, accepting lettered sections such as 80A, and resyncing after two consecutive headings that jump beyond `MAX_SECTION_GAP=8`. Its tests are in `tests/test_chunk.py`.

---

## 4. Prioritised next work

Each item states the evidence and how to accept it. Measure before and after with `make eval`, and never edit `eval/gold.jsonl` to make a change look better. If gold items are wrong, fix them in a separate step and regenerate **both** the baseline and the new result.

### N1. Fix slab-rate retrieval (`q1_slabs`) — highest value
- **Evidence:** `q1_slabs` misses at 10 for every configuration. The rate table (Section 202, chunk text `TABLE Sl. No. Total income Rate of tax … Upto ₹400000 Nil …`) never contains the words "slab" or "new regime".
- **Options, cheapest first:**
  1. A small, documented query-expansion map in `src/retrieval/` (for example "slab" → "rate of tax total income", "new regime" → "section 202"), applied to the BM25 query only.
  2. A cross-encoder reranker over the top 20, such as `BAAI/bge-reranker-base`. This is a new dependency and a model download, so justify it with numbers.
  3. A curated `rates_and_limits.md` in `data/raw/` that cites the sections, tagged `curated` in `src/ingest/tag_metadata.py::CURATED_SOURCES`.
- **Accept when:**
  - `q1_slabs` is a hit at 5;
  - overall hybrid recall@5 has not regressed;
  - `make eval-gen` Q1 answer lists the rates with citation `[n]` and `citation_check.ok`.

### N2. Weak categories: `tds` (recall@5 0.33) and `salary` (0.50)
- Look at `per_item` → `top_ids` in `eval/results/post_changes.json` for `tds_salary`, `tds_rent`, `ded_std` and `penalty_late`.
- Do error analysis first: is it the gold keywords, chunking, or retrieval?

### N3. Upload UI for adding books (the project's stated goal)
- Add a Streamlit page that takes a PDF or Markdown file and runs it through the existing pipeline:
  - `chunk_document()`;
  - tag the result `supplementary`;
  - append to `chunks.jsonl`;
  - rebuild BM25;
  - add the vectors to Chroma;
  - update the manifest.
- Include size and page limits and a delete-document action.
- **Copyright:** do not commit copyrighted CA books. Record a licence for each upload in `data/sources.json`.
- **Accept when:** after uploading a fixture PDF, a question about its contents retrieves it, and deleting it removes its ids from both indexes (tested).

### N4. Abstain on low retrieval confidence
- A rerank score threshold, chosen on the 5 `answerable: false` gold items plus some answerable ones, is a better signal than raw cosine similarity. Measured off-domain dense scores were about 0.45, which overlaps with in-domain scores.

### N5. Smaller items
- **Conversation memory:** condense the last 3 turns into a standalone query, behind a flag and measured.
- **Structured per-query log:** latency per stage, model, tokens, retrieved ids, and citation result.
- **Dockerfile** and a hosted demo.
- **Typed dataclasses** for chunk and answer.
- **Topic tagger:** 65% of chunks are `general`. Either use the tags for filtering or stop displaying them.

---

## 5. Gotchas

- **Use `-o addopts=""` to see the pass count.** `pyproject.toml` sets `addopts = "-q"`, so plain `pytest -q` prints dots only.
- **Old `.env` files** may say `BM25_INDEX_PATH=indexes/bm25.pkl`. Config maps `.pkl` to `.json` automatically, but update the line anyway.
- **Upgrade uvicorn in an old venv.** If the venv still has `uvicorn 0.30.x`, Streamlit warns about websockets. A clean install from `requirements.txt` resolves to 0.54 and the warning goes away; otherwise run `pip install -U uvicorn`.
- **`DENSE_BACKEND=exact` is deliberate.** Chroma's HNSW gave different top-10s between runs (7–11 of 42 questions). Don't switch back unless the corpus grows to roughly 100k chunks.
- **Rebuild with `make index` only.** It deletes and recreates `indexes/chroma/`. The earlier `delete_collection` approach left stale segments and bloat.
- **`make upgrade-chunks` is for when the PDFs are absent.** It is idempotent and keeps chunk boundaries and ids. Prefer `make chunk` once the PDFs exist.
- **Generation eval costs money.** `make eval-gen` makes 6 LLM calls. Results are for manual review against `tests/test_questions.md` and are not auto-scored.
- **The gold set is a keyword proxy.** Every item is `verified: false`. Don't quote the numbers as accuracy, only as comparisons between configurations.
- **Citation check limits:** it checks section numbers and ₹ amounts against the cited chunks, not the meaning of the prose. Known unfixed edge cases: a zero-norm vector or an empty corpus would divide by zero, but neither can occur with the current data.
- **Team-agent note from the previous session:** subagents could not run Python in that environment, so all validation was done by the lead session.

---

## 6. Where things are

| Path | Purpose |
|---|---|
| `src/config.py` | All settings; root-anchored paths; `check_indexes()` |
| `src/llm_client.py` | Timeout and retries, `FALLBACK_MODELS`, `LLMUnavailableError` / `LLMConfigError` |
| `src/text_utils.py` | Shared `tokenize()`, `normalize_text()` (₹, amounts in words) |
| `src/ingest/chunk.py` | Extract → normalise → split → `assign_section_refs` → tag → dedupe; `--upgrade-existing` |
| `src/ingest/build_index.py` | Chroma (clean rebuild) + `bm25.json` + `manifest.json` |
| `src/ingest/download_sources.py` | Registry check, download and record hashes against `data/sources.json` |
| `src/retrieval/` | `dense_retriever` (exact/hnsw), `bm25_retriever` (JSON), `hybrid` (RRF k=60) |
| `src/generation/` | `prompt_templates` (cite `[n]`), `chain.answer()`, `citations.verify_citations()` |
| `app/streamlit_app.py` | One `handle_prompt()` path; `render_assistant_details()`; citation warnings |
| `eval/` | `gold.jsonl`, `run_eval.py`, `run_generation.py`, `results/` |
| `tests/` | 73 tests, including Streamlit `AppTest` and index integrity |

---

## 7. Prompt to start the next session

> Read `HANDOFF.md`, then `report.md` (audit plus the "Post-audit implementation log" appendix), `README.md` and `eval/README.md`. Check `git status`. Reproduce the baseline with `make test lint eval`; the expected numbers are in HANDOFF §1. If the raw PDFs are present, run HANDOFF §3 and verify its acceptance checks. Then work on HANDOFF §4 item N1 and continue down the list. Work in small batches: inspect first, add tests with each change, run `make test lint eval` after each batch and report the actual results, never edit the gold set to improve scores, and append what you did to the implementation log in `report.md`.
