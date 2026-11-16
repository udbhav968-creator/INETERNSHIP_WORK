# ML Operations Runbook

Everything below is off or harmless by default. Each switch changes one thing and can be turned back.

## 1. Switches (environment variables)

| Variable | Default | Effect | Who decides |
|---|---|---|---|
| `KLASSED_JWT_SECRET` | none (production will not start) | Signs login tokens; at least 32 characters | DevOps |
| `DATABASE_URL` | local SQLite | `postgresql://…` uses psycopg 3; all new tables are created on start-up | DevOps |
| `KLASSED_EXPERIMENT_SELECTION_AB` | `off` | `on` splits students 50/50: adaptive (LinUCB) vs random questions from the same pools | Product + ML |
| `KLASSED_BANDIT_EXPLORATION_EPSILON` | `0` | 0.05–0.10 makes the adaptive arm explore, so the bandit can be evaluated offline | ML |
| `REDIS_URL` | none | Shares bandit state and student features across all servers | DevOps |
| `KLASSED_BEHAVIOURAL_TELEMETRY` | `on` | `off` stops storing answer switches and hesitation | Legal / management |
| `KLASSED_ANALYTICS_DATABASE_URL` (GitHub secret) | none | Read-only replica URL; enables the weekly retraining workflow | DevOps |

## 2. Weekly routine (ML)

```bash
python -m model.learning_analytics_reports          # all analytics; sections switch on as data arrives
python -m model.run_full_ml_pipeline --verify-reproducible
python -m model.release_registry create --notes "week N"
python -m model.release_registry list
```

Read `model/student_models/learning_analytics_report.json`. Anomaly flags go to a human; nobody is penalised automatically.

## 3. Releasing and rolling back models

```bash
python -m model.release_registry promote r0002-abc1234 --stage staging --canary 10
python -m model.release_registry promote r0002-abc1234 --stage production --reviewer "Name"
python -m model.release_registry verify            # files on disk = production release?
python -m model.release_registry rollback --reviewer "Name" --apply
```

Production promotion needs a staging release and a named reviewer. Rollback restores the previous production files from git.

## 4. Content team

```bash
python -m model.review_pack build     # data/review/difficulty_tier_review.csv (300 questions, blind) and explanation_rewrite_list.csv (131)
# reviewers fill reviewer_tier (basic / medium / advanced), chapter_tag_ok (yes / no), correct_chapter, reviewer_name
python -m model.review_pack score     # writes model/difficulty_review_decision.json: PROMOTE_V21 or DO_NOT_PROMOTE
```

v2.1 is promoted only if at least 200 reviewed items give kappa ≥ 0.40 with v2.1 and v2.1 agrees with reviewers more than the current tiers do.

Translations go in `data/translations/<lang>/questions.jsonl` (one JSON object per line: `question_id`, `question_text`, `option_a`–`option_d`, optional `explanation`, `translator`, `reviewer`, `status`). `python -m model.language_coverage` checks numbers, formulas, script and second-person approval.

## 5. Inputs only people can supply

| File | Columns | Used for |
|---|---|---|
| `data/training/neet_actual_scores.csv` | `student_id, exam_year, exam_date, actual_score` | Score-prediction validation |
| `data/training/student_attributes.csv` | `student_id` plus any of `language, region, state, device, gender, board, school_type` | Fairness report |

Both need the student's consent (and a parent's for under-18s under the DPDP Rules 2025).

## 6. Semantic retrieval (GPU machine)

```bash
pip install sentence-transformers torch
python -m model.retrieval_backends --encoder sentence_transformers --model intfloat/multilingual-e5-small
```

The report says `ELIGIBLE_FOR_REVIEW` only if top-1 chapter accuracy beats production by 2 points on the same 9,448 held-out questions.
