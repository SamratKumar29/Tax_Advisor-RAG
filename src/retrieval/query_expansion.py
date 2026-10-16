"""
Query expansion for the BM25 retriever: everyday tax vocabulary → statutory wording.

Users ask about "slabs", "TDS" or "late filing"; the Income-tax Act 2025 says
"rate of tax", "deduction of tax at source" and "fee for default in furnishing
return". BM25 only matches exact tokens, so these gaps cause misses that no
amount of ranking can fix. Each rule appends the statutory phrase to the query
(the original words are kept). Dense retrieval already handles paraphrase and
gets the original query.

Rules are general terminology, not answers to particular questions; keep them
that way. Matching is case-insensitive on word boundaries.
"""

import re

# (pattern, statutory phrasing appended to the BM25 query)
EXPANSIONS: list[tuple[str, str]] = [
    (r"\bslabs?\b", "rate of tax total income"),
    (r"\bnew (?:tax )?regime\b", "new tax regime section 202"),
    (r"\bold (?:tax )?regime\b", "option section 202"),
    (r"\btds\b", "deduction of tax at source deducted"),
    (r"\btcs\b", "collection of tax at source collected"),
    (r"\bsalaried\b|\bsalary\b", "salaries"),
    (r"\blate (?:filing|fee)\b|\bbelated\b", "fee for default in furnishing return of income"),
    (r"\bhra\b", "house rent allowance"),
    (r"\bltcg\b", "long-term capital gains"),
    (r"\bstcg\b", "short-term capital gains"),
    (r"\bitr\b", "return of income"),
    (r"\bnri\b", "non-resident"),
    (r"\bhuf\b", "hindu undivided family"),
    (r"\bfy\b", "tax year"),
    (r"\bassessment year\b|\bay\b", "tax year"),
]

_COMPILED = [(re.compile(p, re.IGNORECASE), extra) for p, extra in EXPANSIONS]


def expand_query(query: str) -> str:
    """Return the query with the statutory phrasing of every matching rule appended."""
    extras = []
    for pattern, extra in _COMPILED:
        if pattern.search(query) and extra not in extras:
            extras.append(extra)
    return " ".join([query, *extras]) if extras else query
