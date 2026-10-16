# Retrieval eval: n1_llm_rewrite

Items scored: 42 (answerable only). Relevance = proxy (source + keyword). Dense backend: exact.

| config | recall@5 | recall@10 | MRR@10 | p50 ms |
|---|---|---|---|---|
| hybrid | 0.952 | 0.952 | 0.749 | 39.2 |
| hybrid_llm | 0.929 | 0.976 | 0.718 | 706.3 |

## Recall@5 by category

| category | n | hybrid | hybrid_llm |
|---|---|---|---|
| capital_gains | 4 | 1.00 | 1.00 |
| circulars | 2 | 1.00 | 1.00 |
| compliance | 4 | 1.00 | 1.00 |
| deductions | 5 | 1.00 | 1.00 |
| exemptions | 4 | 1.00 | 1.00 |
| house_property | 2 | 1.00 | 1.00 |
| mapping | 8 | 1.00 | 1.00 |
| regime | 1 | 1.00 | 1.00 |
| residence | 1 | 1.00 | 1.00 |
| salary | 2 | 1.00 | 1.00 |
| slabs | 3 | 0.67 | 0.67 |
| tds | 3 | 0.67 | 0.33 |
| transition | 3 | 1.00 | 1.00 |

## Misses at 10 (any config)

- **hybrid**: q1_slabs, tds_rent
- **hybrid_llm**: q1_slabs
