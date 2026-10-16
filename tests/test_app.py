from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest

import src.config
import src.generation.chain

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


def fake_answer(query, api_key, base_url, model=None):
    return {
        "answer": f"Fake answer to: {query}",
        "sources_text": "",
        "disclaimer": "disclaimer",
        "retrieved_chunks": [],
        "model_used": model,
        "usage": None,
    }


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(src.generation.chain, "answer", fake_answer)
    monkeypatch.setattr(src.config, "check_indexes", lambda *a, **k: None)
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["llm_api_key"] = "test-key"
    at.session_state["llm_model"] = "test-model"
    return at.run()


def test_example_button_produces_assistant_message(app):
    app.button(key="example_0").click().run()
    assert not app.exception
    msgs = app.session_state["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["content"].startswith("Fake answer to: What is the Section 80C")


def test_chat_input_uses_same_path(app):
    app.chat_input[0].set_value("Standard deduction?").run()
    assert [m["role"] for m in app.session_state["messages"]] == ["user", "assistant"]


def test_overlong_query_rejected(app):
    app.chat_input[0].set_value("x" * 3000).run()
    assert app.session_state["messages"] == []
    assert app.error


def test_citation_problems_shown_as_warning(monkeypatch):
    def answer_with_bad_citation(query, api_key, base_url, model=None):
        r = fake_answer(query, api_key, base_url, model)
        r["citation_check"] = {
            "invalid_markers": [7], "uncited_figures": [], "markers_used": [], "ok": False,
            "unsupported": [{"sentence": "s", "claim": "Section 999", "cited": [1]}],
        }
        return r

    monkeypatch.setattr(src.generation.chain, "answer", answer_with_bad_citation)
    monkeypatch.setattr(src.config, "check_indexes", lambda *a, **k: None)
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["llm_api_key"] = "k"
    at.session_state["llm_model"] = "m"
    at.run()
    at.chat_input[0].set_value("q").run()
    texts = " ".join(w.value for w in at.warning)
    assert "[7]" in texts and "Section 999" in texts


def test_bad_api_key_shows_friendly_error(monkeypatch):
    from src.llm_client import LLMConfigError

    def bad_key(*a, **k):
        raise LLMConfigError("The LLM provider rejected the credentials. Check the API key and base URL.")

    monkeypatch.setattr(src.generation.chain, "answer", bad_key)
    monkeypatch.setattr(src.config, "check_indexes", lambda *a, **k: None)
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["llm_api_key"] = "wrong"
    at.session_state["llm_model"] = "m"
    at.run()
    at.chat_input[0].set_value("q").run()
    assert not at.exception
    assert at.session_state["messages"] == []  # orphan user turn removed
    assert "API key" in at.error[0].value
