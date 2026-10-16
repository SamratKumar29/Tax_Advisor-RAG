"""
Thin wrapper over an OpenAI-compatible client for LLM generation.

Works with OpenRouter, NVIDIA NIM, or any endpoint that speaks the
OpenAI chat-completions API. Embeddings are handled separately via
sentence-transformers (local, no API key).

Clients are cached per (api_key, base_url) and calls fall back through
FALLBACK_MODELS when the primary model fails.
"""

from functools import lru_cache

import openai
from openai import OpenAI

from src.config import (
    FALLBACK_MODELS,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT,
)
from src.config import (
    LLM_MODEL as DEFAULT_LLM_MODEL,
)

# Errors that justify trying the next model rather than failing outright.
RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.InternalServerError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.NotFoundError,
)


class LLMUnavailableError(RuntimeError):
    """Raised when every candidate model failed."""

    def __init__(self, models_tried: list[str], errors: list[str]):
        self.models_tried = models_tried
        self.errors = errors
        super().__init__(
            "All LLM models failed: "
            + "; ".join(f"{m} ({e})" for m, e in zip(models_tried, errors))
        )


class LLMConfigError(LLMUnavailableError):
    """Raised for problems no fallback model can fix: no model set, bad key, no access."""

    def __init__(self, message: str, models_tried: list[str] | None = None):
        self.models_tried = models_tried or []
        self.errors = [message]
        RuntimeError.__init__(self, message)


# Errors where the same credentials will fail for every model.
CONFIG_ERRORS = (openai.AuthenticationError, openai.PermissionDeniedError)


@lru_cache(maxsize=8)
def get_llm_client(api_key: str, base_url: str) -> OpenAI:
    """Return a cached, configured OpenAI-compatible client."""
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=LLM_TIMEOUT,
        max_retries=LLM_MAX_RETRIES,
    )


def candidate_models(model: str | None, fallbacks: list[str] | None = None) -> list[str]:
    """Primary model followed by fallbacks, deduped, order preserved."""
    fallbacks = FALLBACK_MODELS if fallbacks is None else fallbacks
    seen: list[str] = []
    for m in [model or DEFAULT_LLM_MODEL, *fallbacks]:
        if m and m not in seen:
            seen.append(m)
    return seen


def chat_completion(messages: list[dict], api_key: str, base_url: str,
                    model: str | None = None, temperature: float = 0.2,
                    max_tokens: int = 2048, fallbacks: list[str] | None = None,
                    client: OpenAI | None = None) -> dict:
    """
    Send a chat-completion request, falling back across models on failure.

    Returns:
        {"content": str, "model_used": str, "usage": dict | None}

    Raises:
        LLMUnavailableError: if every candidate model failed.
    """
    models = candidate_models(model, fallbacks)
    if not models:
        raise LLMConfigError(
            "No LLM model configured. Set LLM_MODEL in .env or enter a model in the sidebar.")
    client = client or get_llm_client(api_key, base_url)

    tried, errors = [], []
    for m in models:
        tried.append(m)
        try:
            response = client.chat.completions.create(
                model=m,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except CONFIG_ERRORS as e:
            raise LLMConfigError(
                f"The LLM provider rejected the credentials ({type(e).__name__}). "
                "Check the API key and base URL.", tried) from e
        except RETRYABLE_ERRORS as e:
            errors.append(type(e).__name__)
            continue
        except openai.APIStatusError as e:  # e.g. 400/402/422: model-specific, try the next one
            errors.append(f"{type(e).__name__} {e.status_code}")
            continue
        content = response.choices[0].message.content if response.choices else None
        if not content or not content.strip():
            errors.append("empty content")
            continue
        usage = getattr(response, "usage", None)
        if usage is not None and hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        return {"content": content, "model_used": m, "usage": usage}

    raise LLMUnavailableError(tried, errors)
