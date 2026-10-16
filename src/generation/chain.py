"""
Generation chain: retrieve → build context → call LLM → parse response.

This is the single entry point for answering a user question.
"""

import re
import time
from typing import Callable

from src import config
from src.config import MAX_QUERY_CHARS
from src.generation.citations import verify_citations
from src.generation.memory import condense_question
from src.generation.prompt_templates import build_messages
from src.llm_client import chat_completion
from src.query_log import log_query
from src.retrieval.llm_query import rewrite_query
from src.types import Answer


def hybrid_search(query: str, bm25_extra: str = "") -> tuple[list[dict], float]:
    """Lazy import so the chain can be used/tested without loading indexes.
    Returns (chunks, confidence) — see src.retrieval.hybrid.search_with_confidence."""
    from src.retrieval.hybrid import search_with_confidence
    return search_with_confidence(query, bm25_extra=bm25_extra)


ABSTAIN_MESSAGE = (
    "I could not find anything in the Income-tax Act, 2025, the Income-tax Rules, 2026 "
    "or the other loaded documents that matches this question closely enough to answer "
    "it reliably. This assistant covers Indian income tax for FY 2026-27 / AY 2027-28. "
    "Try rephrasing with the specific provision, or consult a Chartered Accountant."
)


DISCLAIMER = (
    "This is general tax information based on available documents and is not "
    "professional tax advice. Please consult a qualified Chartered Accountant (CA) "
    "for advice specific to your situation. Tax laws are subject to amendments "
    "and notifications."
)


def _parse_response(raw: str | None) -> dict:
    """
    Parse the structured LLM response into answer, sources, disclaimer.

    Expected format:
        ANSWER:
        ...
        SOURCES:
        ...
        DISCLAIMER:
        ...

    Falls back gracefully if the model doesn't follow format exactly.
    """
    raw = raw or ""
    answer = raw
    sources_text = ""
    disclaimer = DISCLAIMER

    # Try to extract ANSWER section
    answer_match = re.search(r"ANSWER:\s*\n(.*?)(?=\nSOURCES:|\nDISCLAIMER:|\Z)",
                             raw, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer = answer_match.group(1).strip()

    # Try to extract SOURCES section
    sources_match = re.search(r"SOURCES:\s*\n(.*?)(?=\nDISCLAIMER:|\Z)",
                              raw, re.DOTALL | re.IGNORECASE)
    if sources_match:
        sources_text = sources_match.group(1).strip()

    return {
        "answer": answer,
        "sources_text": sources_text,
        "disclaimer": disclaimer,
        "raw_response": raw,
    }


def answer(query: str, api_key: str, base_url: str, model: str | None = None,
           fallbacks: list[str] | None = None,
           history: list[dict] | None = None,
           retriever: Callable[[str], list[dict]] | None = None,
           llm: Callable[..., dict] | None = None) -> Answer:
    """
    Answer a user's tax question using hybrid retrieval + LLM generation.

    Args:
        query: The user's question.
        api_key: LLM API key.
        base_url: LLM API base URL.
        model: Optional model override.
        fallbacks: Optional fallback order (e.g. from llm_client.probe_models);
            defaults to FALLBACK_MODELS.
        history: Earlier chat turns [{"role", "content"}]; used only when
            CONVERSATION_MEMORY is on, to rewrite a follow-up into a standalone question.
        retriever: Optional retrieval function (defaults to hybrid search).
        llm: Optional chat-completion function (defaults to llm_client).

    Returns:
        {
            "answer": str,          # The main answer text
            "sources_text": str,    # LLM-reported sources
            "disclaimer": str,      # Standing disclaimer
            "retrieved_chunks": list,  # Actual retrieved chunks (ground truth)
            "raw_response": str,    # Full raw LLM response
            "model_used": str,      # Model that actually answered
            "usage": dict | None,   # Token usage if reported
            "citation_check": dict, # See src.generation.citations.verify_citations
            "abstained": bool,      # True when retrieval confidence was too low (no LLM call)
            "confidence": float | None,  # best dense cosine similarity (None with a custom retriever)
            "search_query": str,    # the query actually searched (standalone if memory is on)
        }

    Raises:
        ValueError: if the query is empty or longer than MAX_QUERY_CHARS.
        LLMUnavailableError: if every candidate model failed.
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("Query is empty.")
    if len(query) > MAX_QUERY_CHARS:
        raise ValueError(f"Query is too long ({len(query)} > {MAX_QUERY_CHARS} characters).")

    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    # 0. Optional conversation memory: rewrite a follow-up into a standalone question
    search_query = query
    if history and config.CONVERSATION_MEMORY:
        search_query = condense_question(query, history, api_key=api_key, base_url=base_url,
                                         model=model, fallbacks=fallbacks,
                                         llm=llm or chat_completion)
        timings["condense_ms"] = _ms(t0)

    # 1. Retrieve relevant chunks via hybrid search (dense + BM25 → RRF)
    t = time.perf_counter()
    confidence = None
    if retriever is not None:
        chunks = retriever(search_query)
    else:
        extra = ""
        if config.LLM_QUERY_REWRITE:
            extra = rewrite_query(search_query, api_key=api_key, base_url=base_url, model=model,
                                  fallbacks=fallbacks, llm=llm or chat_completion)
        chunks, confidence = hybrid_search(search_query, bm25_extra=extra)
    timings["retrieval_ms"] = _ms(t)

    # 1b. Abstain without calling the LLM when nothing relevant was found
    if confidence is not None and confidence < config.ABSTAIN_DENSE_THRESHOLD:
        out = {
            "answer": ABSTAIN_MESSAGE, "sources_text": "", "disclaimer": DISCLAIMER,
            "raw_response": "", "model_used": "", "usage": None, "abstained": True,
            "confidence": confidence, "search_query": search_query, "retrieved_chunks": [],
            "citation_check": verify_citations(ABSTAIN_MESSAGE, []),
        }
        timings["total_ms"] = _ms(t0)
        log_query(query, out, timings)
        return out

    # 2. Build the prompt messages
    messages = build_messages(search_query, chunks)

    # 3. Call the LLM
    t = time.perf_counter()
    llm_kwargs = {"fallbacks": fallbacks} if fallbacks is not None else {}
    result = (llm or chat_completion)(messages, api_key=api_key, base_url=base_url, model=model,
                                      **llm_kwargs)
    timings["llm_ms"] = _ms(t)

    # 4. Parse the structured response
    parsed = _parse_response(result.get("content"))
    parsed["model_used"] = result.get("model_used", "")
    parsed["usage"] = result.get("usage")

    # 5. Deterministically check the [n] citations against the provided chunks
    parsed["citation_check"] = verify_citations(parsed["answer"], chunks)

    # 6. Attach the actual retrieved chunks — don't rely solely on LLM's
    #    self-reported sources. Surface the ground truth alongside.
    parsed["retrieved_chunks"] = [
        {
            "text": c["text"][:600] + ("..." if len(c["text"]) > 600 else ""),
            "source_doc": c["metadata"]["source_doc"],
            "source_type": c["metadata"]["source_type"],
            "topic": c["metadata"]["topic"],
            "section_ref": c["metadata"].get("section_ref", ""),
            "page": c["metadata"].get("page"),
        }
        for c in chunks
    ]
    parsed["abstained"] = False
    parsed["confidence"] = confidence
    parsed["search_query"] = search_query

    timings["total_ms"] = _ms(t0)
    log_query(query, parsed, timings, chunk_ids=[c.get("chunk_id", "") for c in chunks])
    return parsed


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
