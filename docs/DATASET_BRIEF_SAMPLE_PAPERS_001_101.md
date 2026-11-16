# Dataset Brief: Sample Papers 001–101 (ML review)

**From:** ML (Udbhav Yadav) · **Dataset:** `data/tagged_questions/neet_samples_001_to_101_verified.json` (branch `data/new-sample-papers-001-101`)  
**Scope:** read-only review; the source file is unchanged. All findings are reproducible with `python -m model.sample_paper_ingestion`.

---

## 1. Headline numbers

| Metric | Value |
|---|---|
| Rows delivered | 16,689 (Biology 7,703 · Physics 5,310 · Chemistry 3,676) |
| **Usable new questions after review** | **9,448** (Physics 3,545 · Chemistry 1,701 · Biology 4,202) |
| Quarantined | 7,241 (list with reasons: `data/quarantine/sample_papers_rejected.json`) |
| Diagram references | 930 / 930 resolve to files (33 extra images are unreferenced) |
| Combined bank for mocks | 11,848 (2,400 PYQ + 9,448 sample) |

## 2. Why 7,241 rows were quarantined

| Reason | Rows | What the content team can do |
|---|---|---|
| Exact copy of an existing NEET PYQ | 4,088 | Nothing; these are already in the bank |
| Repeated within the sample papers (one question appears up to 15 times) | 1,359 | De-duplicate at source |
| Chapter is "General Biology / Chemistry / Physics" or unknown, and the classifier isn't ≥ 90% sure | 2,633 | Re-label: each row includes its top-3 suggested chapters |
| Duplicate options (two options with identical text) | 644 | Fix options |
| Empty question text | 184 | Re-extract |
| Subject–chapter conflict (e.g. Biology + "Motion in a Straight Line") | 9 | Fix subject or chapter |

A row can fail more than one rule; counts are by reason.

## 3. Label-quality issues (accepted rows)

| Issue | Evidence | Request |
|---|---|---|
| "Motion in a Straight Line" used as a catch-all | Its questions cover centre of mass, optics, electrostatics and more; Physics · Kinematics is 53.7% of Physics vs 12.5% in real NEET (4.3×) | Re-tag these 547 rows |
| Biology · Biological Classification over-used | 22.3% of Biology vs 8.3% in NEET (2.7×) | Spot-check |
| 152 chapter names for 65 chapters | NCERT names, "and" vs "&" variants, "General …" | Use the 65 canonical names (mapping: `model/chapter_resolver.py`) |
| 748 near-copies of PYQs (reworded, not exact) | TF-IDF cosine ≥ 0.85 | Kept, but the engine treats them as twins (never in the same paper, and a student who has seen one has seen both) |
| `ideal_time_sec` is a fixed value per tier (35 s / 60 s / 85 s / 120 s) | Same pattern as the PYQs (`synthetic_tier_formula`) | Real timing data is needed before time can be used as a signal |

## 4. Difficulty coverage

| Subject | Basic | Medium | Advanced |
|---|---|---|---|
| Physics (PYQ + sample) | 567 | 3,135 | 386 |
| Chemistry | 472 | 1,494 | 368 |
| Biology | 3,281 | 1,497 | 648 |

The Physics "zero basic" bug is fixed (445 new basic Physics items). Advanced items are now the scarce tier: only 763 of the new questions are advanced.

## 5. Impact on full-length mocks

| Bottleneck chapter | Questions available | Quota per paper | Fully fresh mocks |
|---|---|---|---|
| Environmental Chemistry | 1 | 1 | 1 |
| s-Block Elements | 4 | 1 | 4 |
| Surface Chemistry | 8 | 1 | 8 |
| States of Matter | 21 | 2 | 10 |
| General Principles & Metallurgy | 29 | 2 | 14 |

All five are in the NMC reduced-syllabus exclusion list already in `paper_generator.py`. **Decision needed (Pranay):** if the 180-Q blueprint follows the reduced syllabus, the guaranteed fully-fresh mock count rises from 1 to about 13. Otherwise these chapters need more questions.

## 6. Requests to the content team, in priority order

1. Re-label the 2,633 "General / unknown" rows using the provided suggestions.
2. Re-tag the 547 "Motion in a Straight Line" rows.
3. Add about 600 advanced questions, especially Biology and Chemistry.
4. Fix the 644 duplicate-option and 184 empty-text rows.
5. Decide whether the PYQ difficulty tiers should be re-labelled (see §7).

## 7. Note on PYQ difficulty labels

The 2,400 PYQ tiers come from a text-complexity heuristic (`difficulty_source = chapter_text_complexity_quantile_heuristic`). A leakage-free model trained on the **verified** sample-paper tiers agrees with the PYQ tiers only 43% of the time. The initial-assessment forms are balanced on PYQ tiers, so verified PYQ tiers would make that balance more trustworthy.
