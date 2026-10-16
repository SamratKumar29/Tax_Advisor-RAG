from src.text_utils import format_inr, normalize_text, tokenize, words_to_number


def test_tokenize_collapses_indian_amount_formats():
    assert "150000" in tokenize("limit of ₹1,50,000")
    assert "150000" in tokenize("Rs. 1,50,000")
    assert tokenize("₹150000") == ["150000"]


def test_tokenize_keeps_section_tokens():
    toks = tokenize("Section 80C and 10(13A) under section 123.")
    assert "80c" in toks and "123" in toks and "section" in toks


def test_words_to_number():
    assert words_to_number("fifteen lakh") == 1_500_000
    assert words_to_number("one lakh fifty thousand") == 150_000
    assert words_to_number("seventy-five thousand") == 75_000
    assert words_to_number("two crore") == 20_000_000
    assert words_to_number("one hundred and eighty-two") == 182
    assert words_to_number("several") is None


def test_format_inr():
    assert format_inr(400000) == "4,00,000"
    assert format_inr(15000000) == "1,50,00,000"
    assert format_inr(999) == "999"


def test_normalize_backtick_rupee():
    assert normalize_text("Upto ` 400000 Nil") == "Upto ₹400000 Nil"
    assert normalize_text("exceeds `60000") == "exceeds ₹60000"


def test_normalize_amount_in_words():
    out = normalize_text("total income exceeding fifteen lakh rupees during")
    assert "fifteen lakh rupees (₹15,00,000)" in out


def test_normalize_clause_spacing():
    assert normalize_text("( a) is a citizen; ( iii ) other") == "(a) is a citizen; (iii) other"


def test_normalize_is_idempotent():
    s = "exceeding twelve lakh rupees or ` 60000, ( b) and (₹5) elsewhere"
    once = normalize_text(s)
    assert normalize_text(once) == once
    assert "(₹12,00,000)" in once  # pre-existing "(₹" elsewhere does not block annotation


def test_normalize_leaves_non_amounts():
    s = "the rupees paid were few"
    assert normalize_text(s) == s
