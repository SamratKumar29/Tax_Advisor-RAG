import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def make_chunk(text="Section 80C allows a deduction up to ₹1,50,000.", source="act.pdf"):
    return {
        "text": text,
        "metadata": {
            "source_doc": source,
            "source_type": "act",
            "topic": "deductions",
            "section_ref": "80C",
        },
    }


class FakeCompletions:
    """Scripted chat.completions: each item is an Exception or content (str/None)."""

    def __init__(self, script):
        self.script = dict(script)
        self.calls = []

    def create(self, model, **kwargs):
        self.calls.append(model)
        outcome = self.script[model]
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=outcome))],
            usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 42}),
        )


@pytest.fixture
def fake_client():
    def _make(script):
        completions = FakeCompletions(script)
        return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions
    return _make
