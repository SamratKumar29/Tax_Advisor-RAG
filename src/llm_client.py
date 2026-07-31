"""
Thin wrapper over an OpenAI-compatible client for LLM generation.

Works with OpenRouter, NVIDIA NIM, or any endpoint that speaks the
OpenAI chat-completions API. Embeddings are handled separately via
sentence-transformers (local, no API key).
"""

from openai import OpenAI
from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


def get_llm_client() -> OpenAI:
    """Return a configured OpenAI-compatible client."""
    return OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
    )


def chat_completion(messages: list[dict], model: str | None = None,
                    temperature: float = 0.2, max_tokens: int = 2048) -> str:
    """
    Send a chat-completion request and return the assistant's response text.

    Args:
        messages: List of {"role": ..., "content": ...} dicts.
        model: Override the default model from config.
        temperature: Sampling temperature (low = more deterministic).
        max_tokens: Maximum tokens in the response.

    Returns:
        The assistant message content as a string.
    """
    client = get_llm_client()
    response = client.chat.completions.create(
        model=model or LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
