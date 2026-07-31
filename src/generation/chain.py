"""
Generation chain: retrieve → build context → call LLM → parse response.

This is the single entry point for answering a user question.
"""

import re
from src.retrieval.hybrid import search as hybrid_search
from src.generation.prompt_templates import build_messages, format_context
from src.llm_client import chat_completion


DISCLAIMER = (
    "This is general tax information based on available documents and is not "
    "professional tax advice. Please consult a qualified Chartered Accountant (CA) "
    "for advice specific to your situation. Tax laws are subject to amendments "
    "and notifications."
)


def _parse_response(raw: str) -> dict:
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


def answer(query: str) -> dict:
    """
    Answer a user's tax question using hybrid retrieval + LLM generation.

    Returns:
        {
            "answer": str,          # The main answer text
            "sources_text": str,    # LLM-reported sources
            "disclaimer": str,      # Standing disclaimer
            "retrieved_chunks": list,  # Actual retrieved chunks (ground truth)
            "raw_response": str,    # Full raw LLM response
        }
    """
    # 1. Retrieve relevant chunks via hybrid search (dense + BM25 → RRF)
    chunks = hybrid_search(query)

    # 2. Build the prompt messages
    messages = build_messages(query, chunks)

    # 3. Call the LLM
    raw_response = chat_completion(messages)

    # 4. Parse the structured response
    parsed = _parse_response(raw_response)

    # 5. Attach the actual retrieved chunks — don't rely solely on LLM's
    #    self-reported sources. Surface the ground truth alongside.
    parsed["retrieved_chunks"] = [
        {
            "text": c["text"][:300] + ("..." if len(c["text"]) > 300 else ""),
            "source_doc": c["metadata"]["source_doc"],
            "source_type": c["metadata"]["source_type"],
            "topic": c["metadata"]["topic"],
            "section_ref": c["metadata"].get("section_ref", ""),
        }
        for c in chunks
    ]

    return parsed
