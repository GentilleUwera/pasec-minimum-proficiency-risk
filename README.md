# Predicting Minimum Proficiency Risk in African Primary Education Using PASEC Survey Data

Course project for 04-652 AI System Design, Carnegie Mellon University Africa.
Author: Gentille Uwera

## Problem Statement

SDG 4, Target 4.1 measures whether children actually learn, through Indicator 4.1.1: the share of children reaching a minimum proficiency level in reading and mathematics. In Francophone Sub-Saharan Africa this is measured by PASEC, a regional assessment coordinated by CONFEMEN across 14 countries.

Knowing which Grade 6 students are below minimum proficiency requires a full learning assessment, which most education systems can only afford periodically and at national sample level. PASEC already collects background information on every tested student (household assets, parental literacy, grade repetition, age, teacher training, class size, school location, home language).

The objective is to test whether that background information alone can flag a Grade 6 student as likely below the PASEC minimum proficiency level in mathematics, without a new test. The intended output is a low-cost, interpretable decision-support prototype, not a classroom device.

## Dataset Overview

| Item | Value |
|---|---|
| Source | PASEC2019 international database, CONFEMEN data portal: https://pasec.confemen.org/en/data-portal |
| File | `PASEC2019_GRADE6_INT.dta` (Stata), inside `PASEC2019_INT_STATA.zip` |
| Documentation | PASEC2019 data manual (CONFEMEN); AFLEARN PASEC Analysis Guide: https://datafirst-aflearn.github.io/pasec-guide/ |
| Sample size (N) | 62,934 Grade 6 students in 3,298 schools, 14 countries |
| Raw columns (P) | 749 (identifiers, test items, plausible values, student, teacher and principal questionnaires, derived indices, survey weights) |
| Features used | 10 background columns (Phase 1 report, Section 2.2). Test items (f1 to f98, m1 to m84) and plausible values are excluded to prevent target leakage. |
| Target (Y) | `below_min_maths`: 1 if a student is below the PASEC minimum proficiency level in mathematics, 0 otherwise |

**How Y is built.** PASEC reports mathematics achievement as five plausible values per student (MATHS_PV1 to MATHS_PV5). The boundary of the PASEC minimum proficiency level for end-of-primary mathematics is 521.46 (PASEC2019 manual, Table 5.17). A student is labelled 1 when at least 3 of the 5 plausible values are at or below 521.46. Result: 39,685 students labelled 1 (63.1%) and 23,249 labelled 0 (36.9%).

**Schools.** `ID_ECOLE` identifies a school within a country (manual, Table 2.2), so the same code repeats across countries. A school is identified by country plus `ID_ECOLE` (3,298 schools).

## Task Type

**Binary classification.**

## Repository Structure

```
.
├── .gitignore
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                original PASEC zip (kept locally, not committed)
│   └── processed/          cleaned, encoded, scaled feature set (Phase 2)
├── phase1_strategy/
│   └── Phase1_Strategy_Report.pdf
├── phase2_implementation/
│   ├── notebooks/
│   │   └── pipeline_execution.ipynb      (Phase 2)
│   └── Phase2_Final_Report.pdf           (Phase 2)
├── src/
│   ├── __init__.py
│   ├── transformers.py         loading, missing-value cleaning, label construction
│   └── pipeline_builder.py     ColumnTransformer and Pipeline definitions
└── figures/                    diagnostic plots, PNG at 300 DPI (Phase 2)
```

Empty folders contain a `.gitkeep` file so that Git keeps them until their Phase 2 files are added.

## Environment Setup

Requires Python 3.11 or newer.

```bash
git clone https://github.com/GentilleUwera/pasec-minimum-proficiency-risk.git
cd pasec-minimum-proficiency-risk
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Getting the Data

The PASEC files are not stored in this repository. The Grade 6 file is 373 MB uncompressed, above GitHub's 100 MB file limit, and CONFEMEN distributes the data through its portal.

1. Download the PASEC2019 international database (Stata format) from https://pasec.confemen.org/en/data-portal
2. Save `PASEC2019_INT_STATA.zip` unchanged in `data/raw/`.

The zip contains `PASEC2019_GRADE6_INT.dta` (used here) and `PASEC2019_GRADE2_INT.dta` (not used). The code reads the Grade 6 file directly from the zip, so the original is never modified.

SHA-256 of the `PASEC2019_GRADE6_INT.dta` used in this project:
`34bea29630ddc5c6d5723f434abf8f1412a9d16c92c7bd7faa251cfef56453b9`

## Status

- Phase 1 (preprocessing strategy): `phase1_strategy/Phase1_Strategy_Report.pdf`
- Phase 2 (pipeline and diagnostics): in progress
