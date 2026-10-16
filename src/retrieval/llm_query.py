"""
Optional LLM query rewriting (HyDE-style keywords) for the BM25 retriever.

The deterministic expansion map (query_expansion.py) covers common vocabulary.
For everything else, a small LLM call turns the user's question into the
wording the Income-tax Act 2025 / Rules 2026 would use ("slab rates" →
"rate of tax total income new tax regime section 202"). The keywords are
appended to the BM25 query only; dense retrieval keeps the original question.

Off by default (LLM_QUERY_REWRITE=0): it adds one LLM call (~1-3 s) per
question. Failures never block answering; the original query is used.
Results are cached in-process, keyed by question.
"""

from src.llm_client import LLMUnavailableError, chat_completion

REWRITE_PROMPT = """You turn a user's Indian income-tax question into search keywords for a \
keyword index over the Income-tax Act, 2025 and the Income-tax Rules, 2026.

Write ONE line of 5-20 keywords/short phrases using the statutory wording the Act would use \
(e.g. "rate of tax", "total income", "deduction of tax at source", "fee for default", \
"chargeable under the head salaries"), plus the section number of the Act 2025 if you know it. \
Do not answer the question. Output only the keywords line."""

_cache: dict[str, str] = {}


def rewrite_query(query: str, api_key: str, base_url: str, model: str | None = None,
                  fallbacks: list[str] | None = None, llm=chat_completion) -> str:
    """Return statutory keywords for `query`, or "" if the LLM is unavailable."""
    if query in _cache:
        return _cache[query]
    messages = [
        {"role": "system", "content": REWRITE_PROMPT},
        {"role": "user", "content": query},
    ]
    kwargs = {"fallbacks": fallbacks} if fallbacks is not None else {}
    try:
        result = llm(messages, api_key=api_key, base_url=base_url, model=model,
                     temperature=0, max_tokens=80, **kwargs)
    except LLMUnavailableError:
        return ""
    keywords = " ".join((result.get("content") or "").split())[:400]
    _cache[query] = keywords
    return keywords
