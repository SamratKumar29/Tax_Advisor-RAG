"""
Shared text utilities used at both index time and query time.

Keeping the tokenizer in one place guarantees BM25 indexing and querying can
never drift apart (they previously had two copies that "must match exactly").
"""

import re

# ── Tokenizer (BM25) ──────────────────────────────────────────────────────

_DIGIT_COMMA = re.compile(r"(?<=\d),(?=\d)")
_TOKEN = re.compile(r"[a-z0-9]+(?:[a-z0-9./\-]*[a-z0-9])?")


# Question words and function words. They carry no topic, but because questions
# start with them ("What are the ...") they otherwise add noise to BM25 scores.
STOPWORDS = frozenset("""
a about am an and any are as at be been being but by can could did do does doing
for from had has have how i if in into is it its me my of on or our shall should
so than that the their them then there these this those to was we were what when
where which while who whom why will with would you your
""".split())


def tokenize(text: str) -> list[str]:
    """
    Lowercase, drop the rupee sign, and collapse digit-group commas so that
    "₹1,50,000", "Rs. 1,50,000" and "150000" all yield the token "150000".
    Single-character tokens and STOPWORDS are dropped.
    """
    text = _DIGIT_COMMA.sub("", text.lower()).replace("₹", " ")
    return [t for t in _TOKEN.findall(text) if len(t) > 1 and t not in STOPWORDS]


# ── Normalisation of extracted PDF text ───────────────────────────────────

# pypdf extracts the rupee glyph in the official PDFs as a backtick: "` 150000"
_BACKTICK_RUPEE = re.compile(r"`\s?(?=\d)")
# "( a)" / "( iii )" → "(a)" / "(iii)"
_PAREN_SPACING = re.compile(r"\(\s+([a-zA-Z0-9]{1,5})\s*\)")

_UNITS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {"crore": 10_000_000, "lakh": 100_000, "thousand": 1_000, "hundred": 100}

_NUM_WORD = "|".join(sorted([*_UNITS, *_TENS, *_SCALES, "and"], key=len, reverse=True))
# e.g. "fifteen lakh rupees", "one lakh fifty thousand rupees", "seventy-five thousand rupees"
_AMOUNT_IN_WORDS = re.compile(
    rf"\b((?:(?:{_NUM_WORD})(?:[\s\-]+|(?=\s*rupees)))+)\s*rupees\b(?!\s\(₹)",
    re.IGNORECASE,
)


def words_to_number(phrase: str) -> int | None:
    """Convert e.g. "one lakh fifty thousand" → 150000. Returns None if unparseable."""
    words = [w for w in re.split(r"[\s\-]+", phrase.lower().strip()) if w and w != "and"]
    if not words:
        return None
    total, current = 0, 0
    for w in words:
        if w in _UNITS:
            current += _UNITS[w]
        elif w in _TENS:
            current += _TENS[w]
        elif w == "hundred":
            current = (current or 1) * 100
        elif w in _SCALES:
            total += (current or 1) * _SCALES[w]
            current = 0
        else:
            return None
    total += current
    return total or None


def format_inr(n: int) -> str:
    """Indian digit grouping: 1500000 → "15,00,000"."""
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join(groups) + "," + tail


def _annotate_amount(m: re.Match) -> str:
    value = words_to_number(m.group(1))
    if value is None or value < 100:
        return m.group(0)
    return f"{m.group(0)} (₹{format_inr(value)})"


def normalize_text(text: str) -> str:
    """
    Normalise statute text extracted by pypdf so that amounts are searchable:
    - restore the rupee sign rendered as a backtick ("` 150000" → "₹150000")
    - append a numeric form to amounts written in words
      ("fifteen lakh rupees" → "fifteen lakh rupees (₹15,00,000)")
    - tidy clause markers ("( a)" → "(a)")

    Idempotent: running it twice gives the same result.
    """
    text = _BACKTICK_RUPEE.sub("₹", text)
    text = _PAREN_SPACING.sub(r"(\1)", text)
    text = _AMOUNT_IN_WORDS.sub(_annotate_amount, text)
    return text
