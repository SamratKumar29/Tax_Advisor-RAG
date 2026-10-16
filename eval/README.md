# Retrieval evaluation

```
HF_HUB_OFFLINE=1 venv/bin/python -m eval.run_eval --mode retrieval --configs dense,bm25,hybrid --name <name>
```

Writes `eval/results/<name>.json` (full per-item detail, top-10 chunk ids) and `<name>.md` (summary).

## Gold set (`gold.jsonl`)
47 items: 42 answerable across 15 categories (including the 6 scenarios from `tests/test_questions.md`, ids `q1_*`..`q6_*`) and 5 `answerable: false` out-of-scope items (`q5_fy2425`, GST, US tax, stock picks, FY 2012-13). Out-of-scope items are **not** scored by retrieval mode; they exist for generation/refusal evals.

Schema: `{id, question, category, answerable, relevant: {source_docs[], must_contain_any[], must_contain_all[]?}, verified}`.

## Relevance is a proxy
A retrieved chunk counts as relevant if its `source_doc` is in `source_docs`, its text contains **every** `must_contain_all` keyword (optional), **and** it contains any `must_contain_any` keyword (case-insensitive substring). This is not human judgement: it can reward a chunk that only mentions a keyword, and can miss a correct chunk that paraphrases. All items are `verified: false`. Compare configs against each other on the same gold set; don't read the absolute numbers as true accuracy.

The first version of the gold set used very generic keywords for 11 items (e.g. `"24"`, `"202"`, `"slab"`, `"option"`), which made almost any chunk "relevant" and overstated recall (hybrid recall@5 0.833 vs 0.809 after tightening). They were tightened **before** any retrieval changes were measured, and the baseline was regenerated, so before/after comparisons use the same gold set. For example `q1_slabs` now requires the rate table itself (`rate of tax` + `nil` + `30%`).

## Metrics
- recall@k = fraction of items with at least one relevant chunk in the top k (hit rate)
- MRR@10 = mean reciprocal rank of the first relevant chunk
- latency p50/max per query in ms (warm-up query excluded; machine-dependent; excluded from determinism comparisons)

## Determinism
Chroma's HNSW query is approximate. It returned different dense top-10s between process runs for 11 of 42 questions on the original indexes and 7 of 42 on the rebuilt ones and differed from exact search on 37 of 42. Because of that, the **app itself** now uses exact cosine search by default (`DENSE_BACKEND=exact` in `src/retrieval/dense_retriever.py`; about 2.7k × 384 vectors, ~ms per query). The eval measures the same code path the app runs. `--dense-backend hnsw` measures the ANN path. Two consecutive runs give identical results apart from latency.

## Results
| file | what |
|---|---|
| `results/baseline_pre_changes.*` | Original indexes (before normalisation / section refs / dedup), tightened gold set |
| `results/post_changes.*` | Rebuilt indexes after the B3 changes (previous CI baseline) |
| `results/post_rechunk.*` | Full re-chunk from the PDFs (page numbers added); retrieval unchanged except dense |
| `results/n1_titles*.*` | Section-title header in front of chunks (both indexes / BM25 only) — rejected |
| `results/n1_llm_rewrite.*` | `hybrid` vs `hybrid_llm` (LLM keyword rewrite for BM25; 42 LLM calls) — rejected |
| `results/n1_final.*` | Stopwords + BM25 query expansion + weighted RRF (1.0 / 1.5) — **the CI regression baseline** |
| `results/gen_smoke_n1.json` | 6 end-to-end answers on the n1_final configuration |
| `results/post_changes_query_prefix.*` | Same, with `DENSE_QUERY_PREFIX` set to the bge query instruction (rejected: dense recall@5 0.738 → 0.714) |
| `results/gen_smoke_post_changes.json` | 6 end-to-end answers (`eval/run_generation.py`) with model, latency, tokens and citation check — for manual review, not auto-scored |

Summary (hybrid, exact dense backend):

| | recall@5 | recall@10 | MRR@10 |
|---|---|---|---|
| baseline_pre_changes | 0.809 | 0.905 | 0.688 |
| post_changes | 0.809 | 0.881 | 0.686 |
| post_rechunk | 0.809 | 0.881 | 0.686 |
| n1_final | **0.952** | **0.952** | **0.749** |

`n1_final` was selected on these same 42 questions (a small grid over the BM25 RRF weight 1–3 and candidate depth 10/20/30, with and without the section-title header); weights 1.5–3 tie on recall@5, so the choice is not knife-edge, but the numbers are optimistic. Still missed: `q1_slabs`, `tds_rent`.

The eval always measures the committed base corpus: `run_eval` points `USER_DOCS_DIR` at an empty location so uploaded documents never change the numbers.

The B3 changes were retrieval-neutral on this gold set (5 items moved by about one rank, one fell out of the top 10). Their value is in data quality (section refs on every Act/Rules chunk, ₹ restored, duplicates removed, no pickle), not in measured recall. At that point `q1_slabs` missed because the rate table (Section 202) uses none of the words a user asks with ("slab", "new regime"); with expansion it is still outranked (see `n1_final`).

## Generation smoke eval
```
venv/bin/python -m eval.run_generation --name <name>        # 6 LLM calls, needs .env
```
Records answer, model used, latency, token usage and the deterministic citation check (`src/generation/citations.py`) for each of the six `tests/test_questions.md` scenarios.
