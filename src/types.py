"""
Typed contracts for the dicts passed between ingestion, retrieval and generation.

TypedDicts (not dataclasses) so the existing JSON/dict code paths and tests stay
unchanged while editors and type checkers can see the shapes.
"""

from typing import Any, NotRequired, TypedDict


class Chunk(TypedDict):
    """One line of data/processed/chunks.jsonl (or of the user-docs overlay)."""
    id: str
    text: str
    source_doc: str
    section_ref: str
    section_title: NotRequired[str]
    source_type: str  # primary | curated | supplementary
    topic: str
    tax_year_scope: str
    page: NotRequired[int]
    embedding: NotRequired[list[float]]  # user-docs overlay only


class RetrievedChunk(TypedDict):
    """A ranked search result from dense_retriever / bm25_retriever / hybrid."""
    chunk_id: str
    rank: int
    text: str
    metadata: dict[str, Any]
    score: NotRequired[float]
    rrf_score: NotRequired[float]


class Answer(TypedDict):
    """Return value of src.generation.chain.answer()."""
    answer: str
    sources_text: str
    disclaimer: str
    raw_response: str
    model_used: str
    usage: dict | None
    citation_check: dict
    retrieved_chunks: list[dict]
    abstained: bool
    confidence: float | None
    search_query: str
