"""
Deterministic post-generation check of the model's citations.

The prompt asks the model to cite context blocks as [1], [2], ... after each
claim. This module verifies, without another LLM call, that:

1. every marker refers to a context block that was actually provided, and
2. section numbers and ₹ amounts stated in a cited sentence appear in at least
   one of the chunks that sentence cites.

It cannot judge whether prose is semantically correct; it catches the
high-risk failure for a tax assistant — an invented section number or figure
attributed to a source that does not contain it.
"""

import re

from src.text_utils import tokenize

_MARKER = re.compile(r"\[(?:Context\s*)?(\d+(?:\s*,\s*(?:Context\s*)?\d+)*)\]", re.IGNORECASE)
# Sentence boundary, except after common abbreviations ("Rs. 8 crore", "Sec. 80C", "No. 5").
_SENTENCE_SPLIT = re.compile(
    r"(?<![Rr]s\.)(?<![Ss]ec\.)(?<!No\.)(?<!no\.)(?<!i\.e\.)(?<!e\.g\.)(?<=[.!?])\s+|\n+"
)
_SECTION = re.compile(r"\bsections?\s+(\d{1,3}[A-Z]{0,4})\b", re.IGNORECASE)
# "₹1,50,000", "Rs. 8 crore", "INR 2.5 lakh"
_AMOUNT = re.compile(r"(?:₹|\brs\.?|\binr)\s?(\d[\d,]*(?:\.\d+)?)(?:\s*(lakh|crore)\b)?", re.IGNORECASE)
_SCALE = {"lakh": 100_000, "crore": 10_000_000}


def _markers(sentence: str) -> list[int]:
    nums = []
    for group in _MARKER.findall(sentence):
        nums += [int(n) for n in re.findall(r"\d+", group)]
    return nums


def _chunk_supports_section(chunk: dict, number: str) -> bool:
    if (chunk.get("metadata", {}).get("section_ref") or "").lower() == f"section {number.lower()}":
        return True
    text = chunk["text"].lower()
    n = re.escape(number.lower())
    # Not part of an amount: "80" must not match "80,000", "₹80" or "8.0".
    return bool(re.search(rf"(?<![a-z0-9₹,.]){n}(?![a-z0-9]|[,.]\d)", text))


def _amount_key(number: str, scale: str) -> tuple[str, str]:
    """(digits form, literal form) e.g. ("80000000", "8 crore") or ("150000", "")."""
    number = number.replace(",", "")
    if not scale:
        return number, ""
    digits = str(round(float(number) * _SCALE[scale.lower()]))
    return digits, f"{number} {scale.lower()}"


def _chunk_supports_amount(chunk: dict, amount: tuple[str, str]) -> bool:
    digits, literal = amount
    if digits in tokenize(chunk["text"]):
        return True
    if literal:
        number, scale = literal.split()
        return bool(re.search(rf"{re.escape(number)}\s*{scale}", chunk["text"], re.IGNORECASE))
    return False


def verify_citations(answer: str, chunks: list[dict]) -> dict:
    """
    Returns:
        {
          "markers_used": sorted list of cited context numbers,
          "invalid_markers": markers outside 1..len(chunks),
          "unsupported": [{"sentence", "claim", "cited"}] — figure not found in cited chunks,
          "uncited_figures": sentences stating a section/amount without any marker,
          "ok": True when there are no invalid markers and no unsupported claims,
        }
    """
    n_ctx = len(chunks)
    used, invalid, unsupported, uncited = set(), set(), [], []

    for sentence in filter(None, (s.strip() for s in _SENTENCE_SPLIT.split(answer or ""))):
        cited = _markers(sentence)
        bare = _MARKER.sub("", sentence)
        sections = _SECTION.findall(bare)
        amounts = [_amount_key(n, sc) for n, sc in _AMOUNT.findall(bare)]

        if not cited:
            if sections or amounts:
                uncited.append(sentence)
            continue

        valid = [c for c in cited if 1 <= c <= n_ctx]
        used.update(valid)
        invalid.update(c for c in cited if c not in valid)
        if not valid:
            continue
        cited_chunks = [chunks[c - 1] for c in valid]

        for sec in dict.fromkeys(sections):
            if not any(_chunk_supports_section(ch, sec) for ch in cited_chunks):
                unsupported.append({"sentence": sentence, "claim": f"Section {sec}", "cited": valid})
        for amt in dict.fromkeys(amounts):
            if not any(_chunk_supports_amount(ch, amt) for ch in cited_chunks):
                claim = f"₹{amt[1]}" if amt[1] else f"₹{amt[0]}"
                unsupported.append({"sentence": sentence, "claim": claim, "cited": valid})

    return {
        "markers_used": sorted(used),
        "invalid_markers": sorted(invalid),
        "unsupported": unsupported,
        "uncited_figures": uncited,
        "ok": not invalid and not unsupported,
    }
