# Model Cards

Generated 2026-11-16T13:15:25+00:00 from commit `fc5dbb33b7c13a515705640eec5daefccf4730c3` by `python -m model.mlops_registry_and_cards`.

## Summary

| Model | Status | Key metrics |
|---|---|---|
| difficulty_rf_production | production | holdout_accuracy: 0.7792; holdout_macro_f1: 0.7825; cv_accuracy_train_split: 0.7953; oof_brier: 0.3072; holdout_ece: 0.052; temperature_decision: rejected_on_holdout_identity_used |
| difficulty_v21_text | candidate | accuracy: 0.8238; macro_f1: 0.7705; ece: 0.0189; accuracy_duplicate_free_rows: 0.8141; majority_baseline: 0.5252 |
| difficulty_v2_structural | superseded_candidate | accuracy: 0.7851; macro_f1: 0.7227; n: 9448; majority_baseline: 0.5252 |
| ncert_rag_retriever | production | eval_questions: 9448; eval_set: questions not in the index (no chapter hint); top1_chapter_accuracy: 0.3496; top3_chapter_accuracy: 0.5341; random_baseline_top1: 0.0154 |
| ncert_rag_retriever_unified | candidate | unified: eval_questions: 1747; eval_set: questions not in the index (no chapter hint); top1_chapter_accuracy: 0.3268; top3_chapter_accuracy: 0.5335; random_baseline_top1: 0.0154; production_on_same_set: eval_questions: 1747; eval_set: questions not in the index (no chapter hint); top1_chapter_accuracy: 0.3183; top3_chapter_accuracy: 0.5031; random_baseline_top1: 0.0154 |
| irt_cold_start_ridge | production_prior | a_5fold_r2: 0.948; b_5fold_r2: 0.9949; c_5fold_r2: 0.818 |
| student_response_models | waiting_for_data | status: SKIPPED_INSUFFICIENT_DATA; readiness_checks: students >= 200: False; responses >= 20000: False; items with >= 30 responses >= 50: False; irt: —; bkt: — |
| bkt_dina_linucb_priors | production_prior | student_attempts_available: 0; linucb_cold_start_regret_reduction_vs_random_percent: 96.8 |

## difficulty_rf_production

| Field | Value |
|---|---|
| status | production |
| task | predict the PYQ difficulty tier |
| training_data | 2,400 official PYQs, 11 non-leaking features |
| labels | chapter_text_complexity_quantile_heuristic |
| evaluation | holdout: 20% grouped stratified split made before any tuning; near-duplicate questions share a group; cross_validation: StratifiedGroupKFold(5) on the training split only; used for hyperparameter selection; calibration: temperature fitted on training out-of-fold probabilities, ECE reported on the untouched holdout; split_seed: 42 |
| metrics | holdout_accuracy: 0.7792; holdout_macro_f1: 0.7825; cv_accuracy_train_split: 0.7953; oof_brier: 0.3072; holdout_ece: 0.052; temperature_decision: rejected_on_holdout_identity_used |
| limitations | agreement with the heuristic tier formula, not empirical difficulty; heuristic vs human-verified tiers: kappa 0.0693 |
| artifacts | model/question_difficulty_model.joblib (a2f55761804f); model/rf_difficulty_forest.npz (cdfd08124ba1) |

## difficulty_v21_text

| Field | Value |
|---|---|
| status | candidate |
| task | predict the difficulty tier from question text |
| training_data | 9448 verified sample-paper questions from 83 papers |
| labels | verified sample-paper tiers (papers 001-101 accepted rows) |
| evaluation | GroupKFold(5) by paper_source; no paper appears in both train and test |
| metrics | accuracy: 0.8238; macro_f1: 0.7705; ece: 0.0189; accuracy_duplicate_free_rows: 0.8141; majority_baseline: 0.5252 |
| limitations | trained on sample papers; distribution may differ from real exam papers; advanced tier is 8% of labels, so its recall is the weakest |
| artifacts | model/question_difficulty_model_v21_text_candidate.joblib (e1158b6dba20) |

## difficulty_v2_structural

| Field | Value |
|---|---|
| status | superseded_candidate |
| task | predict the difficulty tier from structural features |
| training_data | verified sample-paper questions |
| labels | human-verified sample-paper tiers |
| evaluation | GroupKFold(5) by paper |
| metrics | accuracy: 0.7851; macro_f1: 0.7227; n: 9448; majority_baseline: 0.5252 |
| limitations | superseded by v2.1 |
| artifacts | model/question_difficulty_model_v2_candidate.joblib (86df54ccbc20) |

## ncert_rag_retriever

| Field | Value |
|---|---|
| status | production |
| task | retrieve the NCERT chapter note for a question |
| training_data | NCERT chapter notes + every training question stem, topic tag and explanation |
| labels | chapter of the question |
| evaluation | questions not in the index (no chapter hint) |
| metrics | eval_questions: 9448; eval_set: questions not in the index (no chapter hint); top1_chapter_accuracy: 0.3496; top3_chapter_accuracy: 0.5341; random_baseline_top1: 0.0154 |
| limitations | lexical retrieval; no dense embeddings; chapter retrieval is a proxy for answer usefulness |
| artifacts | model/ncert_rag_svd_index.npz (43bd1232ed96); model/ncert_rag_svd_meta.json (65fb624307f5) |

## ncert_rag_retriever_unified

| Field | Value |
|---|---|
| status | candidate |
| task | same retriever indexed on PYQ + sample papers |
| training_data | every question from the held-out sample papers |
| labels | chapter of the question |
| evaluation | held-out sample papers |
| metrics | unified: eval_questions: 1747; eval_set: questions not in the index (no chapter hint); top1_chapter_accuracy: 0.3268; top3_chapter_accuracy: 0.5335; random_baseline_top1: 0.0154; production_on_same_set: eval_questions: 1747; eval_set: questions not in the index (no chapter hint); top1_chapter_accuracy: 0.3183; top3_chapter_accuracy: 0.5031; random_baseline_top1: 0.0154 |
| limitations | small gain over production |
| artifacts | model/ncert_rag_svd_index_unified.npz (f355a356f5bd) |

## irt_cold_start_ridge

| Field | Value |
|---|---|
| status | production_prior |
| task | predict 3PL a/b/c for a new question |
| training_data | 2,400 PYQs |
| labels | a/b/c priors derived from tier and metadata, not fitted to responses |
| evaluation | 5-fold CV |
| metrics | a_5fold_r2: 0.948; b_5fold_r2: 0.9949; c_5fold_r2: 0.818 |
| limitations | high R2 reflects that targets are a function of the inputs (tier, time) |
| artifacts | model/irt_cold_start_ridge.npz (f8717be1f496) |

## student_response_models

| Field | Value |
|---|---|
| status | waiting_for_data |
| task | fit IRT a/b/c and per-chapter BKT from real student answers |
| training_data | 0 responses from 0 students |
| labels | observed correct/incorrect answers |
| evaluation | held-out students; predict the second half of each student's answers |
| metrics | status: SKIPPED_INSUFFICIENT_DATA; readiness_checks: students >= 200: False; responses >= 20000: False; items with >= 30 responses >= 50: False; irt: —; bkt: — |
| limitations | DINA and LinUCB are not re-fitted here (Q-matrix validation and logged propensities needed) |
| artifacts | model/student_models/irt_item_parameters_fitted.json (None); model/student_models/bkt_chapter_parameters_fitted.json (None) |

## bkt_dina_linucb_priors

| Field | Value |
|---|---|
| status | production_prior |
| task | student-state priors and item selection |
| training_data | question metadata only |
| labels | none (no student responses) |
| evaluation | simulation |
| metrics | student_attempts_available: 0; linucb_cold_start_regret_reduction_vs_random_percent: 96.8 |
| limitations | cannot be validated until real attempts exist |
| artifacts | model/bkt_chapter_parameters.json (f2a214d8adeb); model/dina_calibrated_parameters.json (0e38945e64c6); model/linucb_prior_init.json (7424ceec594f) |

## Promotion gates

Policy: a candidate is promoted only after every gate passes and a reviewer signs off

| Candidate | Compared with | Gates | Decision |
|---|---|---|---|
| difficulty_v21_text | difficulty_v2_structural on the same duplicate-free held-out papers | macro_f1_gain >= 0.02: True; ece <= 0.05: True | ELIGIBLE_PENDING_HUMAN_REVIEW |
| ncert_rag_retriever_unified | ncert_rag_retriever on the same held-out sample papers | top1_gain >= 0.02: False | HOLD |

## Lineage

| Dataset | Rows | SHA-256 |
|---|---|---|
| pyq_bank | 2400 | fc2dc78ec75dd52c |
| pyq_features | 2400 | bdcf6e7308aaa894 |
| sample_papers_verified | 16689 | 5112d99a9a30e761 |
| ncert_knowledge_base | 65 | a5a8f4311b92ce55 |
| student_attempts | 0 | d0790c7e03db72bd |

Libraries: python: 3.13.18; numpy: 2.5.3; pandas: 3.0.5; sklearn: 1.9.1; scipy: 1.18.1; joblib: 1.6.0
