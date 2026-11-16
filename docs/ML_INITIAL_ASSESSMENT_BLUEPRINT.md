# ML Handoff: 180-Question NEET Assessment, New Dataset, Mocks, Profiles & Analytics

**Owner:** ML intern (Udbhav Yadav) · **Branch:** `ml/neet-180-equated-initial-assessment`  
**Scope:** `model/`, `docs/` and the new dataset file only. No backend or frontend code is changed, and Pranay's repo is untouched (it was only read).

---

## 1. Deliverables

| # | Module | Purpose | Test |
|---|---|---|---|
| 1 | `model/assessment_blueprint.py` | Standardized 180-question blueprint, 3 IRT-equated disjoint initial-assessment forms, EAP scoring | suite 26 |
| 2 | `model/chapter_resolver.py` | Maps the dataset's 152 chapter names onto the 65 canonical chapters (exact → NCERT alias → classifier at ≥ 90% precision → quarantine) | suite 27 |
| 3 | `model/sample_paper_ingestion.py` | Validation, de-duplication, quarantine and cold-start IRT calibration of the sample-paper dataset | suite 27 |
| 4 | `model/mock_test_generator.py` | Unlimited 180-question mocks, equated to the initial assessment, fresh items first | suite 28 |
| 5 | `model/student_profile_initializer.py` | Initial assessment → adaptive profile for all 65 chapters, predicted NEET score, spec-v1 rule output | suite 29 |
| 6 | `model/test_analytics_engine.py` | Test-wise and question-wise analytics, equated score trend | suite 30 |
| 7 | `model/difficulty_model_v2.py` | Difficulty-classifier leakage audit + leakage-free v2 candidate trained on verified tiers | suite 31 |
| 8 | `model/unified_similarity_guard.py` | Near-duplicate guard over all 11,848 items, used by the mock generator | suite 32 |
| 9 | `model/unified_corpus_report.py`, `model/ml_extension_registry.py` | Combined-bank readiness report; SHA-256 registry for the 6 new artifacts | suite 32 |

---

## 2. New dataset: what it actually contains

Source: `data/tagged_questions/neet_samples_001_to_101_verified.json` (branch `data/new-sample-papers-001-101`).

| | Count |
|---|---|
| Rows in file | 16,689 |
| **Accepted: new, unique, valid, calibrated** | **9,448** (Physics 3,545 · Chemistry 1,701 · Biology 4,202) |
| Rejected → `data/quarantine/sample_papers_rejected.json` | 7,241 |

| Rejection reason | Rows |
|---|---|
| Exact copy of a PYQ already in the bank | 4,088 |
| Repeated copy within the sample papers (up to 15 copies of one question) | 1,359 |
| Chapter could not be resolved with ≥ 90% confidence ("General Biology / Chemistry / Physics" etc.) | 2,633 |
| Duplicate options | 644 |
| Empty question text | 184 |
| Subject–chapter conflict (e.g. Biology tagged "Motion in a Straight Line") | 9 |

All 930 diagram references resolve to files (33 extra images are unreferenced). Each quarantined row includes its top-3 chapter suggestions, so the content team can fix and re-submit it quickly.

**Label-quality red flags for the content team:**

| Chapter | Share of subject in sample papers | Share in real NEET PYQs | Over-representation |
|---|---|---|---|
| Physics · Kinematics (incl. "Motion in a Straight Line") | 53.7% | 12.5% | 4.3× |
| Biology · Biological Classification | 22.3% | 8.3% | 2.7× |

"Motion in a Straight Line" is used as a fallback label: its questions cover centre of mass, optics, electrostatics and more. Papers are unaffected because chapter quotas are fixed by the blueprint, but chapter-level analytics on these items are less reliable until the labels are fixed.

**Tier imbalance:** the new data has only 763 accepted advanced items, against 3,723 basic and 4,962 medium.

---

## 3. Calibration of the new items

| Step | Method | Evidence |
|---|---|---|
| Cold-start (a, b, c) | Existing ridge IRT predictor from question text | b correlates r = 0.88 with calibrated PYQ b (2,400 anchors) |
| Tier | Verified `difficulty_level` from the dataset | The predictor's own tier guess matched only 59%, so it is not used |
| Linking | Tier-anchored: within each subject × tier cell, the mean and spread of sample items are matched to the calibrated PYQ cell, keeping the cold-start ordering | After linking, sample and PYQ cell means agree to within 0.005 in a, b and c |
| Assumption | A verified tier means the same in sample papers as in PYQs | Replace with empirical calibration once real student responses exist |

Matching on PYQ-fitted constants instead left sample items systematically easier (e.g. Chemistry advanced b 1.39 vs 1.93), which broke mock equivalence. Tier-anchored linking fixed that.

---

## 4. Initial assessment (team question 1)

| Dimension | Specification |
|---|---|
| Length / time / marking | 180 questions · 180 minutes · +4 / −1 / 0 · max 720 |
| Subject / difficulty | 45 / 45 / 90 · 25% basic, 50% medium, 25% advanced per subject |
| Coverage | Every chapter ≥ 1 item; the rest by `topic_importance.json` |
| Items | Verified PYQs only (calibrated anchors); 3 disjoint forms, assigned by hash of `student_id` |
| Equivalence | Expected-marks gap ≤ 0.42 / 720 at any ability; information gap ≤ 0.66%; SEM 0.108 at θ = 0 |

Pranay's spec §6 describes 3 fixed diagnostic papers that are the same for every student. This design keeps that idea, but each student takes one of 3 equivalent forms, so answers can't circulate between students. Decision for Pranay: either keep this, or serve one form to everyone.

---

## 5. Profile initialization (team question 2)

Yes. Per-answer updates already happen on `/submit_paper`. In addition, `initialize_student_profile(responses)` returns, for all 65 chapters:

| Output | Method |
|---|---|
| `global_theta`, SEM, percentile | Bayesian EAP over all 180 responses |
| Chapter `mastery_prob`, `level`, `confidence_interval`, `irt_theta`, `pace_flag` | Static BKT posterior with calibrated guess/slip, DAG shrinkage for unobserved chapters; same fields and thresholds as `StudentTopicState` |
| `spec_v1_initial_level`, `spec_v1_flag_for_extra_support` | The exact rule in spec §6 (accuracy ≥ 0.7 → medium, else basic; < 0.4 → extra support), reported alongside for comparison |
| `predicted_neet_score` | Posterior-predictive 90% interval; 91% coverage on simulated students |
| `priority_chapters` | Top 10 by importance × mastery gap × prerequisite readiness |

---

## 6. Unlimited mocks (team question 3, ML side)

`generate_mock_180(student_id, attempt_index, seen_question_ids)`: 45/45/90 exactly, deterministic per (student, attempt), about 0.2–0.6 s, never uses initial-assessment items.

Measured with the real combined bank (PYQ + 9,448 sample items), 40 consecutive mocks for one student:

| Metric | Result |
|---|---|
| Repeated items by mock 10 / 20 / 40 | 4 / 19 / 94 of 180 |
| Worst expected-marks gap vs initial assessment | 5.11 / 720 (tolerance 6) |
| Mocks outside tolerance | 1 of 40 |
| Chapter weightage | Subject quotas exact; on average about 1 chapter is off by ±1 question (spec §1: weightage fixed) |

Tuning trade-off (`REPEAT_PENALTY` / `CHAPTER_PENALTY` in `mock_test_generator.py`): relaxing the chapter penalty from 3 to 1 gives 11 fully fresh mocks, but leaves about 16 chapters off by ±1 per paper and 4 of 40 mocks out of tolerance. The default favours the spec's fixed weightage.

**Capacity limit:** advanced items and a few small chapters (Environmental Chemistry has no new items). About 600 more advanced Biology/Chemistry items would give roughly 25+ fully fresh mocks.

---

## 7. Analytics (team question 4, ML side)

| Function | Output |
|---|---|
| `analyze_test(paper_ids, responses)` | Marks, attempted / correct / incorrect / unattempted by subject, chapter and tier; per-test θ; equated marks; weakest chapters; negative-marking audit (skip when P < 0.20); per-question flags (`unexpected_miss`, `strong_solve`, `should_have_skipped`, `skipped_likely_correct`, `rushed`, `time_sink`) |
| `score_trend(tests)` | θ and equated-marks series, weighted slope, significance, direction |

---

## 8. Backend integration (backend team; not changed here)

| Need | Call |
|---|---|
| Initial assessment | `load_forms()["forms"][form_index_for_student(sid)]["question_ids"]` |
| Profile on submit | `initialize_student_profile(...)` → upsert `topic_states`, set `global_theta` |
| Mock | `generate_mock_180(sid, n_prior_mocks, seen_ids)` |
| Analytics | `analyze_test(...)`, store `theta` / `theta_sem` per test → `score_trend(...)` |
| Load new questions into DB | Load only IDs in `model/sample_paper_irt_parameters.json`, using its resolved `chapter` (not the file's raw chapter) |

**Issues for the backend team:**

| # | Issue | Impact |
|---|---|---|
| 1 | `topic_id` = chapter name; "Thermodynamics" exists in both Physics and Chemistry | The two chapters merge in the DB |
| 2 | `extract_year()` defaults to 2024 when `paper_source` has no year | Sample papers would display as "NEET 2024" |
| 3 | No `source_type` on `Question` | PYQ vs sample-paper items can't be told apart |
| 4 | Test type and per-test θ not stored | Needed for unlimited attempts with separate records and trends |
| 5 | Loader skips "General …" chapters | 2,633 rows already quarantined here with suggestions for re-labelling |

---

## 9. Training status of every model

| Model | Trained on | Learns from real students yet? | Status |
|---|---|---|---|
| Topic importance | 14 years of PYQ weightage | n/a | ✅ Correct as is (sample papers have no exam year) |
| 3PL IRT item parameters | PYQ: tier heuristics; sample: cold-start + tier-anchored linking | ❌ 0 attempts | ⚠️ Priors only |
| BKT chapter parameters | PYQ-derived heuristics | ❌ | ⚠️ Priors only |
| DINA 8-skill parameters | PYQ Q-matrix heuristics | ❌ | ⚠️ Priors only |
| LinUCB bandit prior | BKT priors + DAG | ❌ | ⚠️ Priors only |
| Drift monitor / temperature scaling | RF out-of-fold probabilities | ❌ | ⚠️ Needs real outcomes |
| Difficulty RF (production) | 2,400 PYQs incl. `ideal_time_sec` | n/a | ⚠️ 97.5% is leakage; 74.9% without the leaked feature |
| Difficulty RF v2 candidate | 9,448 verified sample tiers, no leaked features | n/a | ✅ 78.5% on held-out papers (baseline 52.5%); not promoted |
| Chapter resolver | PYQ + verified sample chapters | n/a | ✅ ≥ 90% precision where it labels |
| Near-duplicate guard | All 11,848 items | n/a | ✅ |
| NCERT RAG index | 65-chapter NCERT KB | n/a | ✅ |

**What "fully trained" needs:** real student responses. `data/training/student_attempts.csv` has 0 rows. Once the initial assessment is live, about 300+ students × 180 responses lets the team re-estimate IRT a/b/c, BKT guess/slip/transit, DINA slip/guess, LinUCB and temperature scaling from data.

## 10. Verification

| Check | Result |
|---|---|
| Standalone ML suites | 32 / 32 pass |
| Backend suites (`backend/tests/test_v2_engine.py`) | 29 / 29 pass (no backend files changed) |
| Core SHA-256 registry | 21 / 21 unchanged |
| ML extension registry | 6 / 6 verified |
