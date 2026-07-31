"""
Keyword-based metadata tagger for tax document chunks.

Tags each chunk with:
- source_type: primary / supplementary
- topic: keyword-matched topic category
- tax_year_scope: which financial year the content applies to

Design note: This is deliberately keyword-based, not ML-based.
It's transparent, free, and good enough at this corpus size.
You can explain exactly how it works in an interview — no black box.
"""

import re

# ── Source type mapping ───────────────────────────────────────────────────
# All our current sources are primary (official government docs)
SUPPLEMENTARY_SOURCES = {
    # Future supplementary sources would go here, e.g.:
    # "cleartax_explainer.pdf",
    # "kotak_deductions_guide.pdf",
}


# ── Topic keyword mapping ────────────────────────────────────────────────
# Dict of topic_name → list of keywords/patterns to match (case-insensitive)
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "slab_rates": [
        "slab", "tax rate", "tax slab", "rate of tax", "rate of income-tax",
        "115BAC", "section 202", "total income exceeds", "surcharge",
        "₹3,00,000", "₹7,00,000", "₹10,00,000", "₹12,00,000", "₹15,00,000",
        "tax payable", "marginal relief",
    ],
    "deductions_80c": [
        "80C", "section 123", "life insurance", "provident fund", "PPF",
        "ELSS", "NSC", "tuition fee", "home loan principal", "Schedule XV",
        "1,50,000", "150000",
    ],
    "deductions_nps": [
        "80CCD", "section 124", "NPS", "national pension", "pension fund",
        "50,000", "employer contribution",
    ],
    "deductions_80d": [
        "80D", "section 126", "medical insurance", "health insurance",
        "mediclaim", "preventive health",
    ],
    "deductions_education_loan": [
        "80E", "section 129", "education loan", "interest on loan.*education",
    ],
    "deductions_donations": [
        "80G", "section 131", "donation", "charitable", "approved fund",
    ],
    "hra": [
        "HRA", "house rent allowance", "10\\(13A\\)", "section 15.*2",
        "rent paid", "metro city", "50% of salary", "40% of salary",
    ],
    "house_property": [
        "house property", "self-occupied", "let out", "annual value",
        "section 24", "section 22", "home loan interest", "2,00,000",
        "standard deduction.*30%", "Schedule II",
    ],
    "regime_comparison": [
        "old regime", "new regime", "old tax regime", "new tax regime",
        "regime comparison", "opt.out", "default regime", "concessional rate",
        "which regime", "115BAC", "section 202",
    ],
    "standard_deduction_salary": [
        "standard deduction", "16\\(ia\\)", "section 19", "75,000", "50,000",
    ],
    "capital_gains": [
        "capital gain", "112A", "111A", "section 195", "section 196",
        "LTCG", "STCG", "long.term capital", "short.term capital",
        "equity", "1,25,000",
    ],
    "tds": [
        "TDS", "tax deducted at source", "192", "194A", "194C", "194H",
        "194I", "194J", "section 39[3-9]", "section 40[0-9]",
    ],
    "filing_returns": [
        "return of income", "ITR", "filing", "due date", "belated return",
        "revised return", "section 139", "section 263",
    ],
    "rebate": [
        "rebate", "87A", "section 202.*rebate", "tax rebate",
        "rebate under", "7,00,000", "income up to",
    ],
    "section_mapping": [
        "old act", "new act", "1961.*2025", "2025.*1961", "section mapping",
        "renumber", "transition", "interplay", "corresponding section",
    ],
    "senior_citizen": [
        "senior citizen", "super senior", "60 years", "80 years",
        "80TTB", "section 153", "higher exemption",
    ],
    "savings_interest": [
        "80TTA", "80TTB", "section 153", "savings account interest",
        "10,000", "50,000.*interest",
    ],
    "lta": [
        "LTA", "leave travel", "travel allowance", "10\\(5\\)",
    ],
}


def _match_topic(text: str) -> str:
    """
    Match chunk text against topic keywords. Returns the first matching
    topic, or 'general' if no specific topic matches.

    Uses case-insensitive regex matching for flexibility.
    """
    text_lower = text.lower()

    best_topic = "general"
    best_score = 0

    for topic, keywords in TOPIC_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            try:
                if re.search(keyword.lower(), text_lower):
                    score += 1
            except re.error:
                # If regex fails, try plain substring match
                if keyword.lower() in text_lower:
                    score += 1

        if score > best_score:
            best_score = score
            best_topic = topic

    return best_topic


def _determine_source_type(source_doc: str) -> str:
    """Determine if a source document is primary or supplementary."""
    for supp in SUPPLEMENTARY_SOURCES:
        if supp in source_doc:
            return "supplementary"
    return "primary"


def _determine_tax_year_scope(text: str, source_doc: str) -> str:
    """
    Determine the tax year scope of a chunk.

    - Transition FAQ content → marks as covering both old and new
    - Everything else under the new Act → FY2026-27
    """
    text_lower = text.lower()

    # Check for old-Act / transition content
    old_act_signals = [
        "income-tax act, 1961",
        "income tax act, 1961",
        "act of 1961",
        "fy 2025-26",
        "fy2025-26",
        "ay 2026-27",
        "ay2026-27",
        "prior to 1 april 2026",
    ]

    if "transition" in source_doc.lower() or "interplay" in source_doc.lower():
        return "transition_1961_to_2025"

    for signal in old_act_signals:
        if signal in text_lower:
            return "FY2025-26_AY2026-27_or_earlier"

    return "FY2026-27_AY2027-28"


def tag_chunk(chunk: dict) -> dict:
    """
    Apply metadata tags to a chunk dict in-place.

    Tags: source_type, topic, tax_year_scope.
    """
    chunk["source_type"] = _determine_source_type(chunk["source_doc"])
    chunk["topic"] = _match_topic(chunk["text"])
    chunk["tax_year_scope"] = _determine_tax_year_scope(
        chunk["text"], chunk["source_doc"]
    )
    return chunk
