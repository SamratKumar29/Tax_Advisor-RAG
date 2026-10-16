# Retrieval eval: n1_titles

Items scored: 42 (answerable only). Relevance = proxy (source + keyword). Dense backend: exact.

| config | recall@5 | recall@10 | MRR@10 | p50 ms |
|---|---|---|---|---|
| dense | 0.691 | 0.738 | 0.575 | 21.9 |
| bm25 | 0.809 | 0.881 | 0.609 | 5.4 |
| hybrid | 0.809 | 0.881 | 0.656 | 28.0 |

## Recall@5 by category

| category | n | dense | bm25 | hybrid |
|---|---|---|---|---|
| capital_gains | 4 | 0.75 | 1.00 | 1.00 |
| circulars | 2 | 1.00 | 0.50 | 1.00 |
| compliance | 4 | 0.75 | 0.75 | 0.50 |
| deductions | 5 | 1.00 | 1.00 | 1.00 |
| exemptions | 4 | 0.50 | 0.75 | 0.75 |
| house_property | 2 | 1.00 | 1.00 | 1.00 |
| mapping | 8 | 0.50 | 1.00 | 0.88 |
| regime | 1 | 1.00 | 1.00 | 1.00 |
| residence | 1 | 0.00 | 1.00 | 1.00 |
| salary | 2 | 0.50 | 0.50 | 0.50 |
| slabs | 3 | 0.67 | 0.33 | 0.67 |
| tds | 3 | 0.33 | 0.33 | 0.33 |
| transition | 3 | 1.00 | 1.00 | 1.00 |

## Misses at 10 (any config)

- **dense**: q3_hra, map_24b, map_87a, map_115bac, map_194a, ded_std, tds_salary, tds_rent, surcharge, agri, residence
- **bm25**: q1_slabs, ded_std, tds_salary, tds_rent, penalty_late
- **hybrid**: ded_std, tds_salary, tds_rent, surcharge, agri
