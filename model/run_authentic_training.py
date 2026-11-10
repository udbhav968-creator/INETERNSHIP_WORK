import json
import sys
import time
from pathlib import Path

from model.data_integrity_gate import run_data_integrity_gate
from model.irt_3pl_model import calibrate_question_bank
from model.train_psychometric_and_rag_models import train_all_auxiliary_models
from model.train_difficulty_model import main as train_rf_difficulty_model
from model.training_and_validation_pipeline import run_deep_training_and_validation
from model.probability_calibrator import fit_and_save_rf_temperature_calibration
from model.verify_artifact_integrity import verify_all_ml_artifacts


def run_full_authentic_training():
    t_start = time.perf_counter()
    print("=" * 70)
    print("KLASSED NEET ADAPTIVE ENGINE: FULL AUTHENTIC TRAINING PIPELINE")
    print("=" * 70)

    print("\n[1/6] Running Data Integrity Gate on 2,400 Verified Questions...")
    gate_res = run_data_integrity_gate()
    print(f"      Status: {gate_res.get('status')}")
    print(f"      Verified Questions: {gate_res.get('verified_questions_count')}")
    print(f"      Quarantined Questions: {gate_res.get('quarantined_questions_count')}")

    print("\n[2/6] Calibrating 3PL IRT & Training Psychometric Auxiliary Models...")
    calibrated = calibrate_question_bank()
    print(f"      Calibrated 3PL Items: {len(calibrated)}")
    aux_metrics = train_all_auxiliary_models()
    print(f"      3PL Ridge 5-Fold b-R2: {aux_metrics['irt_cold_start_ridge_regressor']['b_5fold_r2']:.4f}")
    print(f"      BKT Chapters Calibrated: {aux_metrics['bkt_65_chapter_empirical_bayes']['total_chapters']}")
    print(f"      DINA Skills Calibrated: {len(aux_metrics['dina_8_skill_calibrated']['skill_priors'])}")
    print(f"      RAG SVD Variance Explained: {aux_metrics['ncert_pyq_svd_rag']['lsa_explained_variance_ratio']:.4f}")

    print("\n[3/5] Training Random Forest 300-Tree Classifier & Tournament Benchmark...")
    train_rf_difficulty_model()
    val_report = run_deep_training_and_validation(save_artifacts=True)
    cv_acc = val_report["stratified_5fold_cross_validation"]["mean_cv_accuracy"]
    cv_f1 = val_report["stratified_5fold_cross_validation"]["mean_cv_macro_f1"]
    print(f"      Mean 5-Fold CV Accuracy: {cv_acc:.4f}")
    print(f"      Mean 5-Fold Macro F1: {cv_f1:.4f}")

    print("\n[4/5] Temperature Scaling Probability Calibration Results...")
    temp_sum = val_report["temperature_calibration_summary"]
    print(f"      Optimal Temperature T: {temp_sum['optimal_temperature']:.4f}")
    print(f"      Pre-calibration ECE: {temp_sum['pre_ece']:.4f}")
    print(f"      Post-calibration ECE: {temp_sum['post_ece']:.4f}")

    print("\n[5/5] Verifying Artifact Registry Checksums...")
    integ = verify_all_ml_artifacts(allow_serverless_omissions=False, require_clean_bandit=True)
    print(f"      Verified Artifacts: {integ['verified_artifacts_count']}")
    print(f"      Overall Integrity: {integ['integrity_status']}")

    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 70)
    print(f"AUTHENTIC TRAINING COMPLETE IN {elapsed:.2f} SECONDS")
    print("=" * 70)
    return {
        "status": integ["integrity_status"],
        "elapsed_seconds": round(elapsed, 2),
        "cv_accuracy": cv_acc,
        "cv_macro_f1": cv_f1,
        "calibrated_items": len(calibrated),
        "verified_artifacts": integ["verified_artifacts_count"],
    }


if __name__ == "__main__":
    res = run_full_authentic_training()
    sys.exit(0 if res["status"] in ("PASSED", "VERIFIED") else 1)
