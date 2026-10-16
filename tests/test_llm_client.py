import httpx
import openai
import pytest

from src.llm_client import LLMUnavailableError, candidate_models, chat_completion

MSGS = [{"role": "user", "content": "hi"}]


def _rate_limit():
    req = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    return openai.RateLimitError("rate limited", response=httpx.Response(429, request=req), body=None)


def _timeout():
    return openai.APITimeoutError(request=httpx.Request("POST", "https://example.invalid"))


def test_candidate_models_dedupes_in_order():
    assert candidate_models("a", ["b", "a", "c", "b"]) == ["a", "b", "c"]


def test_primary_success_returns_model_and_usage(fake_client):
    client, comp = fake_client({"a": "hello"})
    out = chat_completion(MSGS, "k", "u", model="a", fallbacks=["b"], client=client)
    assert out == {"content": "hello", "model_used": "a", "usage": {"total_tokens": 42}}
    assert comp.calls == ["a"]


def test_falls_back_on_rate_limit_timeout_and_empty(fake_client):
    client, comp = fake_client({"a": _rate_limit(), "b": _timeout(), "c": None, "d": "ok"})
    out = chat_completion(MSGS, "k", "u", model="a", fallbacks=["b", "c", "d"], client=client)
    assert out["model_used"] == "d"
    assert comp.calls == ["a", "b", "c", "d"]


def test_all_fail_raises_with_models_tried(fake_client):
    client, _ = fake_client({"a": _rate_limit(), "b": "   "})
    with pytest.raises(LLMUnavailableError) as exc:
        chat_completion(MSGS, "k", "u", model="a", fallbacks=["b"], client=client)
    assert exc.value.models_tried == ["a", "b"]
    assert "a" in str(exc.value) and "b" in str(exc.value)


def test_no_model_configured_raises(fake_client, monkeypatch):
    # Isolate from any LLM_MODEL set in the developer's .env
    monkeypatch.setattr("src.llm_client.DEFAULT_LLM_MODEL", "")
    client, _ = fake_client({})
    with pytest.raises(LLMUnavailableError):
        chat_completion(MSGS, "k", "u", model="", fallbacks=[], client=client)


def _status(cls, code):
    req = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    return cls("err", response=httpx.Response(code, request=req), body=None)


def test_bad_key_raises_config_error_without_trying_fallbacks(fake_client):
    from src.llm_client import LLMConfigError
    client, comp = fake_client({"a": _status(openai.AuthenticationError, 401), "b": "unused"})
    with pytest.raises(LLMConfigError) as exc:
        chat_completion(MSGS, "k", "u", model="a", fallbacks=["b"], client=client)
    assert "API key" in str(exc.value)
    assert comp.calls == ["a"]  # same key would fail for every model


def test_other_status_errors_fall_back(fake_client):
    client, comp = fake_client({"a": _status(openai.BadRequestError, 400), "b": "ok"})
    out = chat_completion(MSGS, "k", "u", model="a", fallbacks=["b"], client=client)
    assert out["model_used"] == "b"


def test_no_model_message_is_actionable(fake_client, monkeypatch):
    from src.llm_client import LLMConfigError
    monkeypatch.setattr("src.llm_client.DEFAULT_LLM_MODEL", "")
    client, _ = fake_client({})
    with pytest.raises(LLMConfigError, match="LLM_MODEL"):
        chat_completion(MSGS, "k", "u", model="", fallbacks=[], client=client)
