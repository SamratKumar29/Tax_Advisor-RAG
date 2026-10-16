from src.generation.citations import verify_citations


def _c(text, section_ref=""):
    return {"text": text, "metadata": {"section_ref": section_ref, "source_doc": "act.pdf"}}


CHUNKS = [
    _c("123. Deduction ... aggregate does not exceed ₹150000 ...", "Section 123"),
    _c("Upto ₹400000 Nil; From ₹400001 to ₹800000 5% ...", "Section 202"),
]


def test_supported_claims_pass():
    ans = "Section 123 allows up to ₹1,50,000 [1]. Income up to ₹4,00,000 is taxed at nil [2]."
    r = verify_citations(ans, CHUNKS)
    assert r["ok"] and r["markers_used"] == [1, 2]
    assert r["unsupported"] == [] and r["uncited_figures"] == []


def test_invalid_marker_flagged():
    r = verify_citations("The limit is ₹1,50,000 [3].", CHUNKS)
    assert not r["ok"] and r["invalid_markers"] == [3]


def test_invented_section_flagged():
    r = verify_citations("This is covered by Section 999 [1].", CHUNKS)
    assert not r["ok"]
    assert r["unsupported"][0]["claim"] == "Section 999"


def test_amount_cited_to_wrong_chunk_flagged():
    r = verify_citations("The deduction limit is ₹1,50,000 [2].", CHUNKS)
    assert [u["claim"] for u in r["unsupported"]] == ["₹150000"]


def test_uncited_figures_reported_but_not_failing():
    r = verify_citations("The limit is ₹1,50,000.", CHUNKS)
    assert r["ok"] and r["uncited_figures"] == ["The limit is ₹1,50,000."]


def test_context_style_and_multi_markers():
    r = verify_citations("Rates are in Section 202 [Context 2]. See also [1, 2].", CHUNKS)
    assert r["ok"] and r["markers_used"] == [1, 2]


def test_empty_answer():
    r = verify_citations("", CHUNKS)
    assert r["ok"] and r["markers_used"] == []


def test_rs_inside_word_is_not_an_amount():
    # Regression: "years 8" was parsed as "rs 8" → ₹8
    r = verify_citations("Two years 8 months remain under Section 123 [1].", CHUNKS)
    assert r["ok"] and r["unsupported"] == []


def test_repeated_claim_reported_once():
    r = verify_citations("Section 999 and again section 999 [1].", CHUNKS)
    assert len(r["unsupported"]) == 1


def test_amount_with_scale_word():
    chunks = [_c("profit under the new Act is Rs 8 crore instead of Rs 8.5 crore")]
    assert verify_citations("Profit is Rs 8 crore [1].", chunks)["ok"]
    assert verify_citations("Profit is ₹8.5 crore [1].", chunks)["ok"]
    bad = verify_citations("Profit is Rs 9 crore [1].", chunks)
    assert [u["claim"] for u in bad["unsupported"]] == ["₹9 crore"]


def test_scaled_amount_matches_digit_form():
    chunks = [_c("deduction not exceeding ₹150000")]
    assert verify_citations("Up to Rs 1.5 lakh [1].", chunks)["ok"]


def test_abbreviation_does_not_split_sentence():
    # Regression: "Rs. 8 crore" was split at "Rs." so the amount was never checked
    chunks = [_c("profit is Rs 8 crore")]
    bad = verify_citations("Profit is Rs. 9 crore [1].", chunks)
    assert [u["claim"] for u in bad["unsupported"]] == ["₹9 crore"]
    assert verify_citations("See Sec. 80C and profit of Rs. 8 crore [1].", chunks)["ok"]


def test_section_number_not_matched_inside_amount():
    # Regression: "Section 80" was "supported" by "80,000" in the chunk
    chunks = [_c("a deduction of ₹80,000 is allowed")]
    r = verify_citations("This is Section 80 [1].", chunks)
    assert [u["claim"] for u in r["unsupported"]] == ["Section 80"]
