import pytest

from src.config import MAX_QUERY_CHARS
from src.generation.chain import _parse_response, answer
from tests.conftest import make_chunk


def test_answer_with_injected_retriever_and_llm():
    seen = {}

    def llm(messages, **kwargs):
        seen["messages"] = messages
        return {"content": "ANSWER:\nUp to ₹1,50,000 [1].\nSOURCES:\nact.pdf\n",
                "model_used": "m1", "usage": {"total_tokens": 7}}

    out = answer("80C limit?", api_key="k", base_url="u", model="m1",
                 retriever=lambda q: [make_chunk()], llm=llm)
    assert out["answer"] == "Up to ₹1,50,000 [1]."
    assert out["sources_text"] == "act.pdf"
    assert out["model_used"] == "m1"
    assert out["usage"] == {"total_tokens": 7}
    assert out["retrieved_chunks"][0]["section_ref"] == "80C"
    assert any("80C" in m["content"] for m in seen["messages"])


def test_parse_response_handles_none():
    out = _parse_response(None)
    assert out["answer"] == "" and out["sources_text"] == ""


@pytest.mark.parametrize("query", ["", "   ", "x" * (MAX_QUERY_CHARS + 1)])
def test_rejects_empty_or_overlong_query(query):
    with pytest.raises(ValueError):
        answer(query, api_key="k", base_url="u",
               retriever=lambda q: pytest.fail("retriever called"),
               llm=lambda *a, **k: pytest.fail("llm called"))


# ── Abstention, memory, query log ─────────────────────────────────────────
import json  # noqa: E402

import src.config  # noqa: E402
import src.generation.chain as chain  # noqa: E402

GOOD_LLM_OUT = {"content": "ANSWER:\nUp to ₹1,50,000 [1].\nSOURCES:\nact.pdf\n", "model_used": "m1"}


def test_low_confidence_abstains_without_llm_call(monkeypatch):
    monkeypatch.setattr(chain, "hybrid_search", lambda q, bm25_extra="": ([make_chunk()], 0.40))
    out = answer("GST on restaurants?", api_key="k", base_url="u",
                 llm=lambda *a, **k: pytest.fail("llm must not be called"))
    assert out["abstained"] is True and out["answer"] == chain.ABSTAIN_MESSAGE
    assert out["retrieved_chunks"] == [] and out["confidence"] == 0.40


def test_confident_retrieval_answers(monkeypatch):
    monkeypatch.setattr(chain, "hybrid_search", lambda q, bm25_extra="": ([make_chunk()], 0.85))
    out = answer("80C limit?", api_key="k", base_url="u", llm=lambda *a, **k: GOOD_LLM_OUT)
    assert out["abstained"] is False and out["confidence"] == 0.85


def test_threshold_zero_disables_abstention(monkeypatch):
    monkeypatch.setattr(src.config, "ABSTAIN_DENSE_THRESHOLD", 0.0)
    monkeypatch.setattr(chain, "hybrid_search", lambda q, bm25_extra="": ([make_chunk()], 0.10))
    assert answer("q", api_key="k", base_url="u", llm=lambda *a, **k: GOOD_LLM_OUT)["abstained"] is False


def test_memory_condenses_follow_up_only_when_enabled(monkeypatch):
    searched = []
    history = [{"role": "user", "content": "What is the 80C limit?"},
               {"role": "assistant", "content": "₹1,50,000 under section 123."}]

    def retriever(q):
        searched.append(q)
        return [make_chunk()]

    def llm(messages, **kwargs):
        if "standalone" in messages[0]["content"]:
            assert "80C limit" in messages[1]["content"]
            return {"content": "Does the section 123 limit apply under the new regime?"}
        return GOOD_LLM_OUT

    answer("And under the new regime?", "k", "u", history=history, retriever=retriever, llm=llm)
    monkeypatch.setattr(src.config, "CONVERSATION_MEMORY", True)
    out = answer("And under the new regime?", "k", "u", history=history, retriever=retriever, llm=llm)
    assert searched == ["And under the new regime?",
                        "Does the section 123 limit apply under the new regime?"]
    assert out["search_query"] == searched[1]


def test_memory_falls_back_to_original_question_when_llm_down():
    from src.generation.memory import condense_question
    from src.llm_client import LLMUnavailableError

    def down(*a, **k):
        raise LLMUnavailableError(["m"], ["RateLimitError"])

    hist = [{"role": "user", "content": "x"}]
    assert condense_question("and then?", hist, "k", "u", llm=down) == "and then?"
    assert condense_question("q", [], "k", "u", llm=down) == "q"


def test_query_log_records_stages(tmp_path, monkeypatch):
    log = tmp_path / "q.jsonl"
    monkeypatch.setattr(src.config, "QUERY_LOG_PATH", str(log))
    answer("80C limit?", "k", "u", retriever=lambda q: [make_chunk()],
           llm=lambda *a, **k: {**GOOD_LLM_OUT, "usage": {"total_tokens": 9}})
    monkeypatch.setattr(chain, "hybrid_search", lambda q, bm25_extra="": ([], 0.1))
    answer("GST?", "k", "u", llm=lambda *a, **k: pytest.fail("no llm"))
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["abstained"] for r in rows] == [False, True]
    assert rows[0]["model"] == "m1" and rows[0]["usage"] == {"total_tokens": 9}
    assert {"retrieval_ms", "llm_ms", "total_ms"} <= set(rows[0]["timings_ms"])
    assert rows[0]["citation_ok"] is not None
