# ML Handoff: Standardized 180-Question NEET Initial Assessment

**Owner:** ML  
**Module:** `model/assessment_blueprint.py`  
**Artifacts:** `model/assessment_forms.json` (served forms), `model/assessment_equivalence_report.json` (audit)  
**Test:** `ml_suite_26_irt_equated_180q_initial_assessment_forms`

## 1. Blueprint (answers "what distribution should the initial assessment follow?")

| Dimension | Specification |
|---|---|
| Length / time | 180 questions, 180 minutes |
| Marking | +4 correct, −1 wrong, 0 unattempted (max 720) |
| Section order | Physics (1–45), Chemistry (46–90), Biology (91–180) |
| Subject quota | Physics 45, Chemistry 45, Biology 90 |
| Difficulty share | 25% basic, 50% medium, 25% advanced per subject |
| Subject × tier targets | Physics 11/23/11, Chemistry 11/23/11, Biology 23/45/22 |
| Chapter coverage | Every chapter gets ≥ 1 item; the rest are allocated by `topic_importance.json` weights (largest-remainder rounding) |
| Near-duplicates | No two near-duplicate PYQs in a form (`question_similarity_audit` siblings) |

Thermodynamics appears in both Physics and Chemistry in the bank. The blueprint therefore keys chapters by `subject::chapter`, so the subject quotas stay exact.

## 2. How "equivalent difficulty for every student" is guaranteed

Tier labels alone don't make forms equivalent, so the forms are **IRT-equated**:

1. Three disjoint forms are built from the stratified blueprint above, with zero shared items.
2. A swap search replaces items only within the same chapter and tier cell. It does this until all forms match a common **test characteristic curve** (expected marks) and **test information curve** across θ = −3 … +3.
3. Students are assigned a form deterministically by a hash of `student_id`.

Measured result (from `assessment_equivalence_report.json`):

| Metric | Result | Tolerance |
|---|---|---|
| Max expected-marks gap between forms (any θ) | 0.42 marks / 720 | ≤ 6 |
| Max test-information gap | 0.66% | ≤ 5% |
| Mean IRT difficulty (b) gap | 0.008 | ≤ 0.05 |
| Pairwise item overlap | 0 | 0 |
| SEM at θ = 0 | 0.108 | ≤ 0.25 |
| θ recovery on simulated students (θ = −1.5, 0, 1, 2) | error < 0.4, SEM < 0.25 | — |

The pool runs out in one cell, so form 2 Physics has 24 medium / 10 advanced instead of 23 / 11. The IRT curves still match. The deviation is recorded in `tier_label_deviation_by_form`.

## 3. Profile initialization after submission (answers "will mastery update automatically?")

- **Already in place:** every answer submitted through `/submit_paper` goes through `_process_single_attempt_no_commit`. That updates per-chapter BKT mastery, chapter IRT θ, `global_theta`, DAG propagation, pacing and the LinUCB bandit. So a 180-question submission initializes the adaptive profile without new ML code.
- **New:** `score_initial_assessment(responses)` gives the backend a whole-paper EAP θ, SEM, percentile, marks, subject marks and per-chapter accuracy in one call. Use it for the completion and personalization screen and as the baseline stored with the assessment record.

## 4. What the backend needs to do (not in this PR)

| Need | ML-side entry point |
|---|---|
| Serve the student's form | `load_forms()["forms"][form_index_for_student(student_id)]["question_ids"]`, then fetch those questions from the DB in that order |
| Header data for the exam screen | `duration_minutes`, `max_marks`, `marking`, `section_order` in `assessment_forms.json` |
| Score / baseline on submit | `score_initial_assessment([{"question_id", "is_correct"}, ...])` |
| Regenerate after the question bank changes | `python -m model.assessment_blueprint`, then run the ML suite |

Still the backend team's job: storing assessment-completed status, recording test type, and the history and analytics APIs.
