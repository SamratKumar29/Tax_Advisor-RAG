# Retrieval eval: n1_final

Items scored: 42 (answerable only). Relevance = proxy (source + keyword). Dense backend: exact.

| config | recall@5 | recall@10 | MRR@10 | p50 ms |
|---|---|---|---|---|
| dense | 0.714 | 0.786 | 0.573 | 22.9 |
| bm25 | 0.905 | 0.952 | 0.750 | 3.1 |
| hybrid | 0.952 | 0.952 | 0.749 | 28.2 |

## Recall@5 by category

| category | n | dense | bm25 | hybrid |
|---|---|---|---|---|
| capital_gains | 4 | 0.75 | 1.00 | 1.00 |
| circulars | 2 | 1.00 | 1.00 | 1.00 |
| compliance | 4 | 0.75 | 1.00 | 1.00 |
| deductions | 5 | 1.00 | 1.00 | 1.00 |
| exemptions | 4 | 0.75 | 0.75 | 1.00 |
| house_property | 2 | 1.00 | 1.00 | 1.00 |
| mapping | 8 | 0.50 | 1.00 | 1.00 |
| regime | 1 | 1.00 | 1.00 | 1.00 |
| residence | 1 | 1.00 | 1.00 | 1.00 |
| salary | 2 | 0.50 | 1.00 | 1.00 |
| slabs | 3 | 0.33 | 0.67 | 0.67 |
| tds | 3 | 0.33 | 0.33 | 0.67 |
| transition | 3 | 1.00 | 1.00 | 1.00 |

## Misses at 10 (any config)

- **dense**: q1_slabs, q3_hra, map_24b, map_87a, map_194a, ded_std, tds_salary, tds_rent, surcharge
- **bm25**: q1_slabs, tds_rent
- **hybrid**: q1_slabs, tds_rent
