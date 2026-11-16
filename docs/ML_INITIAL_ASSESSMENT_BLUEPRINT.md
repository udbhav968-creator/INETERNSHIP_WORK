# ML Handoff: 180-Question NEET Assessment, Mocks, Profile Initialization & Test Analytics

**Owner:** ML intern (Udbhav Yadav) · **Branch:** `ml/neet-180-equated-initial-assessment`  
**Scope:** `model/` only. No backend, frontend or database files are changed. Backend integration points are listed in §7.  
**Verification:** 30 / 30 standalone ML suites pass (`python -m model.tests.test_ml_standalone_suite`). Suites 26–30 are new.

---

## 1. Deliverables

| # | Module | Purpose | Artifact(s) | Test |
|---|---|---|---|---|
| 1 | `model/assessment_blueprint.py` | Standardized 180-question blueprint, 3 IRT-equated disjoint initial-assessment forms, form assignment, EAP scoring | `assessment_forms.json`, `assessment_equivalence_report.json` | suite 26 |
| 2 | `model/sample_paper_ingestion.py` | Validates and calibrates the 16,689-question sample-paper dataset when it lands in this repo | `cold_start_irt_linking.json`, `sample_paper_irt_parameters.json`, `sample_paper_ingestion_report.json`, `data/quarantine/sample_papers_rejected.json` | suite 27 |
| 3 | `model/mock_test_generator.py` | Unlimited full-length mocks, equated to the initial assessment, fresh items first | — (generated on request) | suite 28 |
| 4 | `model/student_profile_initializer.py` | Turns the initial assessment into a complete adaptive profile for all 65 chapters | — | suite 29 |
| 5 | `model/test_analytics_engine.py` | Test-wise and question-wise analysis plus equated score trends | — | suite 30 |

---

## 2. Initial assessment blueprint (team question 1)

| Dimension | Specification |
|---|---|
| Length / time / marking | 180 questions · 180 minutes · +4 / −1 / 0 · max 720 |
| Section order | Physics 1–45 · Chemistry 46–90 · Biology 91–180 |
| Difficulty share | 25% basic · 50% medium · 25% advanced per subject (Physics 11/23/11, Chemistry 11/23/11, Biology 23/45/22) |
| Chapter coverage | Every chapter gets ≥ 1 item; the rest are allocated by `topic_importance.json` |
| Item source | Verified NEET PYQs only, because they carry calibrated IRT parameters. Sample-paper items go to mocks. |
| Near-duplicates | Excluded within a form |
| Forms | 3 disjoint forms; a student's form is fixed by a hash of `student_id` |

**Equivalence guarantee.** After the stratified draw, item swaps within the same chapter × tier cell bring all forms onto one 3PL test characteristic curve (expected marks) and one test information curve, from θ = −3 to +3.

| Metric (measured) | Result | Tolerance |
|---|---|---|
| Max expected-marks gap between forms, any ability level | 0.42 / 720 | ≤ 6 |
| Max test-information gap | 0.66% | ≤ 5% |
| Mean difficulty (b) gap | 0.008 | ≤ 0.05 |
| Item overlap between forms | 0 | 0 |
| Measurement error at θ = 0 | 0.108 | ≤ 0.25 |
| Ability recovery on simulated students | error < 0.4 | — |

Known gap: one form's Physics section has 24 medium / 10 advanced instead of 23 / 11, because the pool ran out. The difficulty curves still match, and the gap is recorded in `tier_label_deviation_by_form`.

---

## 3. Profile initialization (team question 2)

**Already automatic:** each answer posted to `/submit_paper` updates BKT mastery, IRT θ, `global_theta`, DAG propagation, pacing and the bandit.

**New:** `initialize_student_profile(responses)` scores the whole paper in one pass. That's more accurate than replaying 180 sequential updates, which give early questions too much weight.

| Output | How it is computed |
|---|---|
| `global_theta`, SEM, percentile | Bayesian EAP over the full response set |
| Per-chapter `mastery_prob` | Static BKT posterior using each chapter's calibrated prior, guess and slip, then hierarchical shrinkage over the prerequisite DAG. Chapters with no responses still get a value. |
| Per-chapter `irt_theta` | EAP with a prior centred on the student's global θ |
| `level` | ≥ 0.70 advanced · ≥ 0.45 medium · else basic (same thresholds as the backend) |
| `confidence_interval`, `avg_time_ratio`, `pace_flag` | Same meaning and thresholds as `StudentTopicState` / `time_tracking.py` |
| `predicted_neet_score` | Posterior-predictive 90% interval, median, percentile and All-India-rank range. Coverage on simulated students: 91% (41/45). |
| `priority_chapters` | Top 10 by importance × mastery gap × prerequisite readiness, for the personalization screen |

`topic_states` rows match `StudentTopicState` columns, so the backend can write them directly.

---

## 4. Unlimited full-length mocks (ML side of team question 3)

`generate_mock_180(student_id, attempt_index, seen_question_ids)` returns a 180-question paper (45/45/90) that is:

- **Equated** to the initial assessment. Tested on 14 consecutive mocks with the PYQ pool and 40 with a pool the size of the new dataset: every one stays within 2.1 marks, so scores are comparable across attempts.
- **Fresh first:** never uses initial-assessment items and avoids items the student has already seen. When a cell runs out, it repeats the fewest possible items and reports `repeated_items`.
- **Deterministic per (student, attempt)**, so a page reload returns the same paper. Generation takes about 0.2–0.3 s.

| Pool | Guaranteed fully fresh mocks per student | Bottleneck |
|---|---|---|
| PYQ only (today) | 1 (repeats rise after that, but mocks stay equated) | Environmental Chemistry chapter size |
| PYQ + 16,689 sample papers (projected from the announced mix) | ≈ 30 | Only 933 advanced items in the new dataset |

**Recommendation to the content team:** about 600 more advanced Biology items would raise fresh-mock capacity to ≈ 50.

---

## 5. Test analytics (ML side of team question 4)

| Function | Output for the Dashboard / Analytics screens |
|---|---|
| `analyze_test(paper_ids, responses)` | Marks, attempted / correct / incorrect / unattempted, accuracy and time by subject, chapter and difficulty; per-test θ; equated marks (expected score on the reference paper with all questions attempted); weakest chapters by marks lost; negative-marking audit (marks lost, avoidable losses where P(correct) < 0.20); per-question flags: `unexpected_miss`, `strong_solve`, `should_have_skipped`, `skipped_likely_correct`, `rushed`, `time_sink` |
| `score_trend(tests)` | θ and equated-marks series, precision-weighted slope, significance of change, direction (improving / stable / declining), best and average scores |

---

## 6. New dataset (16,689 sample-paper questions)

The dataset is on branch `data/new-sample-papers-001-101` in the upstream repo and is not yet in this repo. When `data/tagged_questions/neet_samples_001_to_101_verified.json` (plus `assets/diagrams/`) is present, run:

```
python -m model.sample_paper_ingestion
python -m model.tests.test_ml_standalone_suite
```

| Step | Behaviour |
|---|---|
| Validation | Required fields, `Verified` status, subject / difficulty mapping, valid answer key, 4 distinct options, chapter mapping onto the 65 canonical chapters (`and` ↔ `&` aliases), diagram file exists, no ID clash, no duplicate of a PYQ or of another row. Rejects go to `data/quarantine/sample_papers_rejected.json` with reasons. |
| Calibration | Cold-start ridge IRT. Tiers come from the dataset's verified labels, because the predictor's own tier guess matched only 59%. Parameters are put on the PYQ scale by mean-sigma linking per subject × tier: linked b RMSE 0.16, r = 0.99 (in-sample on 2,400 anchors). |
| Runtime | ≈ 17 min for the first run (≈ 63 ms per item); later runs recompute only changed rows |
| Use | Mocks draw on PYQ + sample papers. The initial assessment stays on PYQ anchors. |

---

## 7. Backend integration (backend team; not changed in this PR)

| Need | Call |
|---|---|
| Serve initial assessment | `load_forms()["forms"][form_index_for_student(sid)]["question_ids"]`, then fetch from the DB in that order |
| Initialize profile on submit | `initialize_student_profile([{question_id, is_correct, time_taken_seconds}])` → upsert `topic_states`, set `global_theta` |
| Serve a mock | `generate_mock_180(sid, n_prior_mocks, seen_ids)` |
| Post-test analysis | `analyze_test(paper_ids, responses)`; store `theta` / `theta_sem` per test for `score_trend` |
| Adaptive / topic tests | Existing `paper_generator.generate_paper` (LinUCB + BKT); unchanged |

**Issues found for the backend team:**

| # | Issue | Impact |
|---|---|---|
| 1 | `load_real_data.py` uses the chapter name as `topic_id`, and "Thermodynamics" exists in both Physics and Chemistry | The two chapters merge into one topic with one subject, so subject splits are off by up to 21 items |
| 2 | `extract_year()` returns 2024 when `paper_source` has no year (e.g. "Sample Paper 12") | Sample-paper items would show as "NEET 2024" in `paper_source` and `years_asked` |
| 3 | `Question` has no `source_type` column | PYQ and sample-paper items can't be told apart for analytics or display |
| 4 | Test type (`initial_assessment` / `mock_180` / `adaptive` / `topic`) and per-test θ aren't stored | Needed for unlimited attempts with separate records, and for equated score trends |
