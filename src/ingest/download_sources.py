"""
Data source reference and re-download helper.

This script documents all source URLs and can re-download them if needed.
In practice, the raw files were downloaded manually and placed in data/raw/.
Run this script to verify files exist and log their status.
"""

import os
from pathlib import Path
from datetime import datetime

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# ── Source registry ────────────────────────────────────────────────────────
# Each entry: (filename, description, url, source_type)
SOURCES = [
    (
        "act_2025_full_text.pdf",
        "Income-tax Act, 2025 (as amended by Finance Act 2026) — full text",
        "https://www.incometaxindia.gov.in/documents/d/guest/income_tax_act_2025_as_amended_by_fa_act_2026-pdf",
        "primary",
    ),
    (
        "press_release_act_2025_commencement.pdf",
        "Press release: IT Act 2025 comes into force from 1 Apr 2026",
        "https://www.incometaxindia.gov.in/documents/d/guest/press-release-income-tax-act-2025-comes-into-force-from-01-04-2026-pdf",
        "primary",
    ),
    (
        "income_tax_rules_2026.pdf",
        "Income-tax Rules, 2026 (operationalizes the Act — notified 20 Mar 2026)",
        "https://www.incometaxindia.gov.in/cbdt/ → Rules section",
        "primary",
    ),
    (
        "faq_transition_1961_to_2025.pdf",
        "Official FAQs on Interplay/Transition from 1961 Act to 2025 Act",
        "https://www.incometax.gov.in/ → FAQs section",
        "primary",
    ),
    (
        "circulars/Circular_No_5_2026.pdf",
        "CBDT Circular No. 5/2026",
        "https://www.incometaxindia.gov.in/circulars",
        "primary",
    ),
    (
        "circulars/Circular-No-06-2026.pdf",
        "CBDT Circular No. 6/2026",
        "https://www.incometaxindia.gov.in/circulars",
        "primary",
    ),
    (
        "section_mapping_1961_to_2025.md",
        "Hand-written section mapping table (old Act → new Act)",
        "Hand-created, cross-checked against ClearTax + official FAQs",
        "primary",
    ),
]


def verify_sources():
    """Check which source files exist in data/raw/ and report status."""
    print(f"{'='*70}")
    print(f"  Data Source Verification — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Raw directory: {RAW_DIR}")
    print(f"{'='*70}\n")

    found = 0
    missing = 0

    for filename, description, url, source_type in SOURCES:
        filepath = RAW_DIR / filename
        exists = filepath.exists()

        if exists:
            size_mb = filepath.stat().st_size / (1024 * 1024)
            status = f"✅ Found ({size_mb:.1f} MB)"
            found += 1
        else:
            status = "❌ MISSING"
            missing += 1

        print(f"  [{source_type.upper():13s}] {filename}")
        print(f"    {status}")
        print(f"    {description}")
        print()

    print(f"{'='*70}")
    print(f"  Summary: {found} found, {missing} missing out of {len(SOURCES)} sources")
    print(f"{'='*70}")


if __name__ == "__main__":
    verify_sources()
