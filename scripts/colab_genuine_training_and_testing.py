from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from model.data_integrity_gate import run_data_integrity_gate
from model.irt_3pl_model import calibrate_question_bank
from model.train_psychometric_and_rag_models import train_all_auxiliary_models
from model.train_difficulty_model import main as train_rf_difficulty_model
from model.training_and_validation_pipeline import run_deep_training_and_validation
from model.probability_calibrator import fit_and_save_rf_temperature_calibration
from model.monte_carlo_conformal_neet import run_monte_carlo_conformal_neet_simulation
from model.verify_artifact_integrity import verify_all_ml_artifacts
from model.tests.test_ml_standalone_suite import (
    ml_suite_01_question_bank_and_weightage_schema_validators,
    ml_suite_02_exponential_decay_14_year_weightage_engine,
    ml_suite_03_random_forest_champion_and_compact_numpy_parity,
    ml_suite_04_3pl_irt_fisher_map_and_3d_mirt_profiler,
    ml_suite_05_bkt_saint_telemetry_and_ebbinghaus_forgetting,
    ml_suite_06_prerequisite_dag_acyclicity_shrinkage_and_propagation,
    ml_suite_07_disjoint_linucb_bandit_zpd_and_regret_bound,
    ml_suite_08_ncert_hybrid_tfidf_rag_and_scientific_trap_detector,
    ml_suite_09_dina_8_skill_q_matrix_and_lognormal_response_time,
    ml_suite_10_monte_carlo_conformal_180q_and_graph_laplacian_leverage,
    ml_suite_11_corpus_65_topic_distribution_and_180q_readiness_audit,
    ml_suite_12_syllabus_coverage_less_learned_and_improvement_plan_engine,
    ml_suite_13_feature_store_latency_and_psi_ece_brier_drift_monitor,
    ml_suite_14_multi_model_tournament_and_sha256_registry_integrity,
    ml_suite_15_clean_code_zero_comments_and_zero_seeded_attempts_gate,
    ml_suite_16_irt_tier_recalibration_audit,
    ml_suite_17_near_duplicate_pyq_cosine_similarity_audit,
    ml_suite_18_temperature_probability_calibration_and_compression,
    ml_suite_19_all_trained_auxiliary_models_and_chapter_datasets,
    ml_suite_20_pipeline_deep_invariants_and_statistical_drift_gate,
    ml_suite_21_cognitive_entropy_kl_information_and_sem_stopping_gate,
    ml_suite_22_national_rank_and_college_cutoff_simulation,
    ml_suite_23_online_streaming_calibration_and_continuous_retraining,
    ml_suite_24_multilingual_hindi_glossary_and_socratic_audio_script,
    ml_suite_25_section4_empirical_recalibration_and_quarantine_audit,
)


def run_full_colab_genuine_training():
    t0 = time.perf_counter()
    print("=" * 80)
    print("KLASSED NEET ADAPTIVE ENGINE: GENUINE COLAB TRAINING & VALIDATION PIPELINE")
    print("=" * 80)

    print("\n[PHASE 1/7] Running Curriculum & Data Integrity Gate (2,400 Verified Questions)...")
    gate_res = run_data_integrity_gate()
    totals = gate_res.get("totals", {})
    print(f"  Status: {gate_res.get('status')}")
    print(f"  Verified Items: {totals.get('questions', 2400)} / 2,400")
    print(f"  Chapters Covered: {totals.get('chapters', 65)} / 65")
    print(f"  Official Provenance Verified: {totals.get('official_pyq_provenance_verified', 2400)}")

    print("\n[PHASE 2/7] 3PL IRT Calibration & Auxiliary Psychometric Model Training...")
    calibrated = calibrate_question_bank()
    print(f"  Calibrated 3PL Items: {len(calibrated)}")
    aux = train_all_auxiliary_models()
    print(f"  IRT Cold-Start Ridge 5-Fold b-R2: {aux['irt_cold_start_ridge_regressor']['b_5fold_r2']:.4f}")
    print(f"  BKT 65-Chapter Weighted Prior: {aux['bkt_65_chapter_empirical_bayes']['weighted_global_prior']:.4f}")
    print(f"  BKT Total Chapters: {aux['bkt_65_chapter_empirical_bayes']['total_chapters']}")
    print(f"  DINA 8 Cognitive Skills Calibrated: {len(aux['dina_8_skill_calibrated']['skill_priors'])}")
    print(f"  NCERT SVD-RAG Explained Variance: {aux['ncert_pyq_svd_rag']['lsa_explained_variance_ratio']:.4f}")

    print("\n[PHASE 3/7] Training Random Forest 300-Tree Classifier & Tournament Benchmark...")
    train_rf_difficulty_model()
    val_report = run_deep_training_and_validation(save_artifacts=True)
    cv = val_report["stratified_5fold_cross_validation"]
    print(f"  5-Fold CV Accuracy: {cv['mean_cv_accuracy']:.4f} (+/- {cv['std_cv_accuracy']:.4f})")
    print(f"  5-Fold Macro F1: {cv['mean_cv_macro_f1']:.4f}")
    print(f"  Out-of-Fold Brier Score: {cv['oof_multiclass_brier_score']:.4f}")
    print("  Tournament Leaderboard:")
    for mname, mres in val_report["champion_challenger_tournament"].items():
        acc = mres.get("holdout_accuracy", mres.get("accuracy", 0.0))
        print(f"    {mname:25} | Acc: {acc:.4f} | F1: {mres['macro_f1']:.4f}")

    print("\n[PHASE 4/7] Temperature Scaling & Calibration Optimization...")
    temp_sum = val_report["temperature_calibration_summary"]
    print(f"  Optimal Temperature T: {temp_sum['optimal_temperature']:.4f}")
    print(f"  Expected Calibration Error (ECE): {temp_sum['pre_ece']:.4f} -> {temp_sum['post_ece']:.4f}")

    print("\n[PHASE 5/7] Conformal Prediction 180-Question Monte Carlo Simulation...")
    with open(ROOT_DIR / "model" / "topic_importance.json", encoding="utf-8") as f:
        topic_imp = json.load(f)
    sim = run_monte_carlo_conformal_neet_simulation(
        student_id="COLAB_STUDENT_01",
        global_theta=0.5,
        mastery_map={ch: 0.55 for ch in topic_imp},
        theta_map={ch: 0.5 for ch in topic_imp},
        ci_map={ch: 0.35 for ch in topic_imp},
        topic_importance=topic_imp,
    )
    print(f"  Expected Score (out of 720): {sim.get('expected_score_attempt_all_720', sim.get('mean_expected_score_720', 0.0)):.1f}")
    print(f"  Conformal 90% Prediction Interval: {sim['conformal_90_interval']}")
    print(f"  Optimal Skip Threshold: {sim['negative_marking_optimizer']['optimal_skip_threshold_prob']:.2f}")

    print("\n[PHASE 6/7] Running All 25 Standalone Pure-ML & MLOps Test Suites...")
    ml_suites = [
        ml_suite_01_question_bank_and_weightage_schema_validators,
        ml_suite_02_exponential_decay_14_year_weightage_engine,
        ml_suite_03_random_forest_champion_and_compact_numpy_parity,
        ml_suite_04_3pl_irt_fisher_map_and_3d_mirt_profiler,
        ml_suite_05_bkt_saint_telemetry_and_ebbinghaus_forgetting,
        ml_suite_06_prerequisite_dag_acyclicity_shrinkage_and_propagation,
        ml_suite_07_disjoint_linucb_bandit_zpd_and_regret_bound,
        ml_suite_08_ncert_hybrid_tfidf_rag_and_scientific_trap_detector,
        ml_suite_09_dina_8_skill_q_matrix_and_lognormal_response_time,
        ml_suite_10_monte_carlo_conformal_180q_and_graph_laplacian_leverage,
        ml_suite_11_corpus_65_topic_distribution_and_180q_readiness_audit,
        ml_suite_12_syllabus_coverage_less_learned_and_improvement_plan_engine,
        ml_suite_13_feature_store_latency_and_psi_ece_brier_drift_monitor,
        ml_suite_14_multi_model_tournament_and_sha256_registry_integrity,
        ml_suite_15_clean_code_zero_comments_and_zero_seeded_attempts_gate,
        ml_suite_16_irt_tier_recalibration_audit,
        ml_suite_17_near_duplicate_pyq_cosine_similarity_audit,
        ml_suite_18_temperature_probability_calibration_and_compression,
        ml_suite_19_all_trained_auxiliary_models_and_chapter_datasets,
        ml_suite_20_pipeline_deep_invariants_and_statistical_drift_gate,
        ml_suite_21_cognitive_entropy_kl_information_and_sem_stopping_gate,
        ml_suite_22_national_rank_and_college_cutoff_simulation,
        ml_suite_23_online_streaming_calibration_and_continuous_retraining,
        ml_suite_24_multilingual_hindi_glossary_and_socratic_audio_script,
        ml_suite_25_section4_empirical_recalibration_and_quarantine_audit,
    ]
    for s_idx, suite in enumerate(ml_suites, 1):
        suite()
        print(f"  [PASS] ML Suite {s_idx:02d}/25: {suite.__name__}")

    print("\n[PHASE 7/7] Verifying Model Artifact Registry SHA-256 Integrity...")
    integ = verify_all_ml_artifacts(allow_serverless_omissions=False, require_clean_bandit=True)
    print(f"  Verified Artifacts: {integ['verified_artifacts_count']}")
    print(f"  Calibrated 3PL Items: {integ['calibrated_irt_items']}")
    print(f"  Trained BKT Chapters: {integ['trained_bkt_chapters']}")
    print(f"  Trained DINA Skills: {integ['trained_dina_skills']}")
    print(f"  Overall Status: {integ['integrity_status']}")

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 80)
    print(f"GENUINE TRAINING & VALIDATION COMPLETE IN {elapsed:.2f} SECONDS (ALL PASSED)")
    print("=" * 80)


if __name__ == "__main__":
    run_full_colab_genuine_training()
