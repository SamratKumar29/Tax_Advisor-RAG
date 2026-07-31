# Test Questions — Indian Tax Advisor RAG

Run each question through the Streamlit app (`streamlit run app/streamlit_app.py`).
Record actual output and retrieved sources below each question.

---

## Q1: What are the new tax regime slab rates for FY 2026-27?

**Quality criteria:**

- Should cite specific slab breakpoints and rates (e.g. ₹3L, ₹7L, ₹10L, ₹12L, ₹15L)
- Should mention Section 202 or equivalent
- Should note rebate threshold
- Should specify this is for AY 2027-28

**Actual output:**
_(run and record here)_

**Retrieved sources:**
_(record source_doc and topic here)_

---

## Q2: How much can I deduct under Section 123 (old 80C) and what qualifies?

**Quality criteria:**

- Should give ₹1,50,000 aggregate limit
- Should mention qualifying investments: LIC, PPF, ELSS, NSC, tuition fees, home loan principal
- Should reference both "Section 123" (new) and "Section 80C" (old) to show section mapping awareness

**Actual output:**
_(run and record here)_

**Retrieved sources:**
_(record source_doc and topic here)_

---

## Q3: How does HRA exemption work?

**Quality criteria:**

- Should explain the three-way minimum computation (actual HRA received, rent paid − 10% of salary, 50%/40% of salary)
- Should mention metro vs non-metro distinction
- Should reference the relevant section (Section 15 / Schedule I or Section 10(13A) old)

**Actual output:**
_(run and record here)_

**Retrieved sources:**
_(record source_doc and topic here)_

---

## Q4: Should I pick the old or new tax regime?

**Quality criteria (critical):**

- Must NOT give a single definitive recommendation ("choose new" or "choose old")
- Must lay out the key differences: slab rates, available deductions (80C, 80D, HRA etc. only in old regime)
- Should provide a framework for the user to calculate for themselves
- Should mention the default regime (new) and opt-out mechanism

**Actual output:**
_(run and record here)_

**Retrieved sources:**
_(record source_doc and topic here)_

---

## Q5: What was the income tax slab for FY 2024-25?

**Quality criteria (out-of-scope test):**

- System should acknowledge this falls outside its primary scope (FY 2026-27 / AY 2027-28)
- Should NOT confidently provide FY 2024-25 slabs as if they were current
- May note that its documents cover the new Act effective from 1 April 2026

**Actual output:**
_(run and record here)_

**Retrieved sources:**
_(record source_doc and topic here)_

---

## Q6 (Bonus): What is Section 80C called now in the new Act?

**Quality criteria (section mapping test):**

- Must correctly identify Section 80C → Section 123 mapping
- Should retrieve from section_mapping_1961_to_2025.md
- Should confirm the ₹1,50,000 limit is unchanged

**Actual output:**
_(run and record here)_

**Retrieved sources:**
_(record source_doc and topic here — should include section_mapping_1961_to_2025.md)_
