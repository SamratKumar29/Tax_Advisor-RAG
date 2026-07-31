# Data Sources — Indian Tax Advisor RAG

> Retrieval date: 2026-07-31 (all files downloaded manually prior to build)

## Available Documents

| #   | Filename                                  | Description                                                                          | Source URL                                                                                                                   | Status        |
| --- | ----------------------------------------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ------------- |
| 1   | `act_2025_full_text.pdf`                  | Income-tax Act, 2025 (as amended by Finance Act 2026) — full text                    | `https://www.incometaxindia.gov.in/documents/d/guest/income_tax_act_2025_as_amended_by_fa_act_2026-pdf`                      | ✅ Downloaded |
| 2   | `press_release_act_2025_commencement.pdf` | Press release: IT Act 2025 comes into force from 1 Apr 2026 — plain-language summary | `https://www.incometaxindia.gov.in/documents/d/guest/press-release-income-tax-act-2025-comes-into-force-from-01-04-2026-pdf` | ✅ Downloaded |
| 3   | `income_tax_rules_2026.pdf`               | Income-tax Rules, 2026 (operationalizes the Act — notified 20 Mar 2026)              | `https://www.incometaxindia.gov.in/cbdt/` → Rules section                                                                    | ✅ Downloaded |
| 4   | `faq_transition_1961_to_2025.pdf`         | Official FAQs on Interplay/Transition from 1961 Act to 2025 Act                      | `https://www.incometax.gov.in/` → FAQs section                                                                               | ✅ Downloaded |
| 5   | `circulars/Circular_No_5_2026.pdf`        | CBDT Circular No. 5/2026                                                             | `https://www.incometaxindia.gov.in/circulars`                                                                                | ✅ Downloaded |
| 6   | `circulars/Circular-No-06-2026.pdf`       | CBDT Circular No. 6/2026                                                             | `https://www.incometaxindia.gov.in/circulars`                                                                                | ✅ Downloaded |

## Missing Documents (proceeding without)

| #   | Planned Filename                                               | Description                               | Reason                    |
| --- | -------------------------------------------------------------- | ----------------------------------------- | ------------------------- |
| 1   | `faq_act_2025.pdf`                                             | Official FAQs as per Income-tax Act, 2025 | Not found on portal       |
| 2   | `faq_senior_citizens.pdf`                                      | FAQs for Senior Citizens                  | Not found on portal       |
| 3   | `faq_house_property.pdf`                                       | FAQs on house property                    | Not found on portal       |
| 4   | `faq_filing_returns.pdf`                                       | FAQs on filing returns                    | Not found on portal       |
| 5   | `faq_tds.pdf`                                                  | FAQs on TDS                               | Not found on portal       |
| 6   | Supplementary explainers (ClearTax, Kotak, HRA worked example) | Plain-language context documents          | Not sourced in this phase |

## Notes

- The `section_mapping_1961_to_2025.md` file in this directory is hand-written, cross-checked against ClearTax mapping guide and official FAQs.
- All documents are tagged `source_type: primary` unless otherwise noted.
- Corpus scope: FY 2026-27 / AY 2027-28 under the Income-tax Act, 2025.
