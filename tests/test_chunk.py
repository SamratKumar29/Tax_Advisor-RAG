"""Tests for the chunker's section tracking, dedup and the keyword tagger."""

from src.ingest.chunk import (
    assign_section_refs,
    dedupe_chunks,
    split_recursive,
)
from src.ingest.tag_metadata import _match_topic


def _chunks(*texts):
    return [{"text": t, "source_doc": "act.pdf", "section_ref": ""} for t in texts]


def test_section_refs_follow_monotonic_headings():
    chunks = _chunks(
        "1. Short title and commencement.\n(1) This Act may be called ...",
        "continues the first section without a heading",
        "2. In this Act, unless the context otherwise requires ...",
        "text\n3. Charge of income-tax.\n(1) ...",
    )
    assign_section_refs(chunks, "Section")
    assert [c["section_ref"] for c in chunks] == [
        "Section 1", "Section 1", "Section 2", "Section 2",
    ]  # the last chunk STARTS inside section 2; section 3 begins mid-chunk


def test_table_rows_and_footnotes_do_not_reset_sections():
    chunks = _chunks(
        "1. Short title.\n",
        "2. Definitions.\n",
        "3. Charge of tax.\n",
        "TABLE\n1. Upto 400000 Nil\n2. From 400001 to 800000 5%",  # table rows go backwards
        "1. Substituted by the Finance Act, 2026, w.e.f. 1-4-2026.",  # footnote
        "4. Next section.\n",
    )
    assign_section_refs(chunks, "Section")
    refs = [c["section_ref"] for c in chunks]
    assert refs[3] == "Section 3" and refs[4] == "Section 3" and refs[5] == "Section 4"


def test_large_forward_jump_rejected():
    chunks = _chunks("1. Title.\n", "250. Not a real heading, a stray number.\n")
    assign_section_refs(chunks, "Section", max_gap=8)
    assert chunks[1]["section_ref"] == "Section 1"


def test_schedule_heading_takes_over():
    chunks = _chunks("1. Title.\n", "SCHEDULE XV\n[See section 123]\n1. Life insurance premium")
    assign_section_refs(chunks, "Section")
    assert chunks[1]["section_ref"] == "Schedule XV"


def test_split_recursive_records_page():
    page1 = "A" * 50 + " first page text. " * 10
    page2 = "Second page content here. " * 10
    text = page1 + "\n\n" + page2
    chunks = split_recursive(text, "doc.pdf", chunk_size=200, chunk_overlap=0,
                             page_starts=[0, len(page1) + 2])
    assert chunks[0]["page"] == 1
    assert chunks[-1]["page"] == 2


def test_dedupe_keeps_first():
    chunks = [
        {"id": "a", "text": "same", "source_doc": "x"},
        {"id": "b", "text": "same", "source_doc": "x"},
        {"id": "c", "text": "same", "source_doc": "y"},
    ]
    assert [c["id"] for c in dedupe_chunks(chunks)] == ["a", "c"]


def test_tagger_section_19_does_not_match_192():
    # Regression: "section 19" (standard deduction) used to match "section 192" (TDS)
    assert _match_topic("tax deducted at source under section 192 on salary") == "tds"


def test_tagger_number_boundary():
    assert _match_topic("enacted in the year 1925") == "general"


def test_tagger_basic_topics():
    assert _match_topic("Section 80D health insurance premium") == "deductions_80d"
    assert _match_topic("deduction under section 80C of rs 1,50,000") == "deductions_80c"


def test_mapping_table_is_curated_not_primary():
    from src.ingest.tag_metadata import _determine_source_type
    assert _determine_source_type("section_mapping_1961_to_2025.md") == "curated"
    assert _determine_source_type("act_2025_full_text.pdf") == "primary"


def test_lettered_section_after_base_accepted():
    chunks = _chunks("80. Base section.\n", "80A. Lettered section.\n", "81. Next.\n")
    assign_section_refs(chunks, "Section", max_gap=100)
    assert [c["section_ref"] for c in chunks] == ["Section 80", "Section 80A", "Section 81"]


def test_resyncs_after_a_large_gap():
    # A real jump bigger than max_gap (e.g. omitted sections) must not freeze tracking:
    # two consecutive headings beyond the gap re-anchor the tracker.
    chunks = _chunks("1. Title.\n", "20. Far ahead.\n", "21. Next.\n", "22. Next again.\n")
    assign_section_refs(chunks, "Section", max_gap=8)
    assert [c["section_ref"] for c in chunks] == ["Section 1", "Section 1", "Section 21", "Section 22"]
