"""
Thin wrapper over an OpenAI-compatible client for LLM generation.

Works with OpenRouter, NVIDIA NIM, or any endpoint that speaks the
OpenAI chat-completions API. Embeddings are handled separately via
sentence-transformers (local, no API key).
"""

from openai import OpenAI
from src.config import LLM_BASE_URL as DEFAULT_LLM_BASE_URL, LLM_MODEL as DEFAULT_LLM_MODEL


def get_llm_client(api_key: str, base_url: str) -> OpenAI:
    """Return a configured OpenAI-compatible client."""
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


def chat_completion(messages: list[dict], api_key: str, base_url: str,
                    model: str | None = None, temperature: float = 0.2,
                    max_tokens: int = 2048) -> str:
    """
    Send a chat-completion request and return the assistant's response text.

    Args:
        messages: List of {"role": ..., "content": ...} dicts.
        api_key: API key for the LLM provider.
        base_url: Base URL for the API endpoint.
        model: Model identifier (defaults to config default).
        temperature: Sampling temperature (low = more deterministic).
        max_tokens: Maximum tokens in the response.

    Returns:
        The assistant message content as a string.
    """
    client = get_llm_client(api_key, base_url)
    response = client.chat.completions.create(
        model=model or DEFAULT_LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
