"""
Conversation memory: rewrite a follow-up ("and for senior citizens?") into a
standalone question using the last few turns, so retrieval sees the full topic.

Off by default (CONVERSATION_MEMORY=0); costs one small LLM call per follow-up.
If the LLM is unavailable, the original question is used unchanged.
"""

from src import config
from src.llm_client import LLMUnavailableError, chat_completion

CONDENSE_PROMPT = """Rewrite the user's latest question as ONE standalone question about Indian \
income tax that can be understood without the conversation. Keep all specifics (sections, \
amounts, years). If it is already standalone, return it unchanged. Output only the question."""


def condense_question(query: str, history: list[dict], api_key: str, base_url: str,
                      model: str | None = None, fallbacks: list[str] | None = None,
                      llm=chat_completion, turns: int | None = None) -> str:
    """Standalone version of `query` given earlier turns [{"role", "content"}]."""
    recent = [m for m in history if m.get("role") in ("user", "assistant")]
    recent = recent[-2 * (turns or config.MEMORY_TURNS):]
    if not recent:
        return query
    transcript = "\n".join(f"{m['role']}: {m['content'][:800]}" for m in recent)
    messages = [
        {"role": "system", "content": CONDENSE_PROMPT},
        {"role": "user", "content": f"Conversation:\n{transcript}\n\nLatest question: {query}"},
    ]
    kwargs = {"fallbacks": fallbacks} if fallbacks is not None else {}
    try:
        result = llm(messages, api_key=api_key, base_url=base_url, model=model,
                     temperature=0, max_tokens=120, **kwargs)
    except LLMUnavailableError:
        return query
    rewritten = " ".join((result.get("content") or "").split())
    return rewritten[: config.MAX_QUERY_CHARS] or query
