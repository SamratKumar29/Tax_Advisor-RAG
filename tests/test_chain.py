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
