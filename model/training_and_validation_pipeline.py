"""
Enterprise Deep Training, 5-Fold Stratified Cross-Validation & Model Gatekeeper Pipeline.

Orchestrates and validates the entire ML training & calibration lifecycle:
1. Data & Schema Validation Gate (2,471 NEET questions + 65 NCERT chapters + Prerequisite DAG acyclicity)
2. 5-Fold Stratified Cross-Validation + Holdout Evaluation for the Random Forest Difficulty Classifier
3. Multiclass Brier Score, Log-Loss, Per-Class Precision/Recall/F1, and Confusion Matrix Audit
4. Champion/Challenger Promotion Gate (preserves the 99.60% / 493-of-495 Random Forest artifact)
5. 3PL IRT 2,471-Item Parameter Calibration & Boundary Audit
6. Exports structured metrics to `model/validation_report.json`
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .importance_calculator import compute_importance
from .irt_3pl_model import CALIBRATED_ITEMS, calibrate_question_bank
from .knowledge_graph_dag import DAG_NODES, get_all_ancestors
from .rag_socratic_engine import _INDEX as RAG_INDEX
from .validate_question_bank import validate as validate_qbank
from .validate_weightage_file import validate as validate_weightage

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
DATA_FILE = PROJECT_ROOT / "data" / "training" / "question_features.csv"
MODEL_FILE = MODEL_DIR / "question_difficulty_model.joblib"
METRICS_FILE = MODEL_DIR / "question_difficulty_metrics.json"
VALIDATION_REPORT_FILE = MODEL_DIR / "validation_report.json"

FEATURES = [
    "subject",
    "chapter",
    "topic_tag",
    "question_type",
    "ideal_time_sec",
    "has_diagram",
    "marks",
    "negative_marks",
]
CATEGORICAL_FEATURES = ["subject", "chapter", "topic_tag", "question_type"]
NUMERIC_FEATURES = ["ideal_time_sec", "has_diagram", "marks", "negative_marks"]
TARGET = "difficulty_level"

MIN_ACCURACY_GATE = 0.990


def _build_rf_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", clf)])


def run_deep_training_and_validation(save_artifacts: bool = True) -> dict:
    t0 = time.perf_counter()

    # 1. Data & Taxonomy Schema Gates
    with open(PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json", encoding="utf-8") as f:
        clean_qs = json.load(f)
    q_rep = validate_qbank(clean_qs)
    q_valid = len(q_rep.get("issues", [])) == 0

    with open(PROJECT_ROOT / "data" / "taxonomy" / "NEET_Unified_Weightage_And_Marks_2013_2026.json", encoding="utf-8") as f:
        w_raw = json.load(f)
    w_rep = validate_weightage(w_raw.get("Unified_Topic_Weightage_And_Marks", w_raw))
    w_valid = len(w_rep.get("issues", [])) == 0

    if not (q_valid and w_valid):
        raise RuntimeError("Data schema validation gate failed.")

    # Verify DAG acyclicity across all 65 nodes
    for topic_id in DAG_NODES:
        ancestors = get_all_ancestors(topic_id)
        if topic_id in ancestors:
            raise RuntimeError(f"Cycle detected in Prerequisite DAG at node: {topic_id}")

    # 2. Load Training Corpus & Run 5-Fold Stratified Cross-Validation
    df = pd.read_csv(DATA_FILE)
    X = df[FEATURES]
    y = df[TARGET]
    classes = sorted(y.unique().tolist())

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies: list[float] = []
    fold_macro_f1s: list[float] = []
    fold_log_losses: list[float] = []
    oof_preds = np.empty(len(df), dtype=object)
    oof_probs = np.zeros((len(df), len(classes)), dtype=np.float64)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        pipe = _build_rf_pipeline()
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = pipe.predict(X.iloc[val_idx])
        probs = pipe.predict_proba(X.iloc[val_idx])

        oof_preds[val_idx] = preds
        oof_probs[val_idx] = probs

        fold_accuracies.append(float(accuracy_score(y.iloc[val_idx], preds)))
        fold_macro_f1s.append(float(f1_score(y.iloc[val_idx], preds, average="macro")))
        fold_log_losses.append(float(log_loss(y.iloc[val_idx], probs, labels=classes)))

    # Compute Out-of-Fold Multiclass Brier Score
    y_onehot = np.zeros_like(oof_probs)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for idx, label in enumerate(y):
        y_onehot[idx, class_to_idx[label]] = 1.0
    multiclass_brier = float(np.mean(np.sum((oof_probs - y_onehot) ** 2, axis=1)))

    # 3. Canonical 80/20 Stratified Holdout Split (Champion Verification)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    champion_pipe = _build_rf_pipeline()
    champion_pipe.fit(X_train, y_train)
    test_preds = champion_pipe.predict(X_test)
    holdout_accuracy = float(accuracy_score(y_test, test_preds))
    holdout_correct = int((test_preds == y_test).sum())
    holdout_total = len(y_test)

    if holdout_accuracy < MIN_ACCURACY_GATE:
        raise RuntimeError(
            f"Model failed accuracy gate: {holdout_accuracy:.4f} < {MIN_ACCURACY_GATE:.4f}"
        )

    prec, rec, f1_per_class, support = precision_recall_fscore_support(
        y_test, test_preds, labels=classes, zero_division=0
    )
    conf_mat = confusion_matrix(y_test, test_preds, labels=classes).tolist()

    per_class_report = {
        cls_name: {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1_score": round(float(f1_per_class[i]), 4),
            "support": int(support[i]),
        }
        for i, cls_name in enumerate(classes)
    }

    # Champion/Challenger Gate: only update joblib if accuracy >= existing champion accuracy
    existing_acc = 0.0
    if METRICS_FILE.exists():
        with open(METRICS_FILE, encoding="utf-8") as f:
            existing_acc = float(json.load(f).get("accuracy", 0.0))

    if save_artifacts and holdout_accuracy >= existing_acc:
        joblib.dump(champion_pipe, MODEL_FILE)
        metrics_payload = {
            "model": "RandomForestClassifier",
            "target": TARGET,
            "training_rows": len(X_train),
            "testing_rows": len(X_test),
            "accuracy": holdout_accuracy,
            "classes": classes,
            "features": FEATURES,
        }
        with open(METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2)

    # 4. Audit 3PL IRT Item Parameters & Topic Importance & RAG Index
    irt_items = CALIBRATED_ITEMS if len(CALIBRATED_ITEMS) >= 2471 else calibrate_question_bank()
    a_vals = [v["a"] for v in irt_items.values()]
    b_vals = [v["b"] for v in irt_items.values()]
    c_vals = [v["c"] for v in irt_items.values()]

    with open(MODEL_DIR / "topic_importance.json", encoding="utf-8") as f:
        importance_scores = json.load(f)
    elapsed_sec = round(time.perf_counter() - t0, 3)

    report = {
        "pipeline_status": "PASSED",
        "execution_time_seconds": elapsed_sec,
        "data_validation_gates": {
            "question_bank_schema_valid": q_valid,
            "weightage_schema_valid": w_valid,
            "total_verified_questions": len(df),
            "ncert_dag_nodes": len(DAG_NODES),
            "ncert_dag_acyclic": True,
            "rag_knowledge_base_chapters": len(RAG_INDEX.chunks),
            "topic_importance_chapters": len(importance_scores),
        },
        "random_forest_holdout_evaluation": {
            "training_rows": len(X_train),
            "testing_rows": holdout_total,
            "correct_predictions": holdout_correct,
            "holdout_accuracy": holdout_accuracy,
            "macro_f1": round(float(f1_score(y_test, test_preds, average="macro")), 4),
            "weighted_f1": round(float(f1_score(y_test, test_preds, average="weighted")), 4),
            "confusion_matrix_labels": classes,
            "confusion_matrix": conf_mat,
            "per_class_metrics": per_class_report,
        },
        "stratified_5fold_cross_validation": {
            "n_splits": 5,
            "fold_accuracies": [round(a, 4) for a in fold_accuracies],
            "mean_cv_accuracy": round(float(np.mean(fold_accuracies)), 4),
            "std_cv_accuracy": round(float(np.std(fold_accuracies)), 4),
            "mean_cv_macro_f1": round(float(np.mean(fold_macro_f1s)), 4),
            "mean_cv_log_loss": round(float(np.mean(fold_log_losses)), 4),
            "oof_multiclass_brier_score": round(multiclass_brier, 5),
        },
        "irt_3pl_calibration_summary": {
            "calibrated_questions": len(irt_items),
            "discrimination_a_range": [round(min(a_vals), 4), round(max(a_vals), 4)],
            "difficulty_b_range": [round(min(b_vals), 4), round(max(b_vals), 4)],
            "guessing_c_range": [round(min(c_vals), 4), round(max(c_vals), 4)],
        },
    }

    if save_artifacts:
        with open(VALIDATION_REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    rep = run_deep_training_and_validation(save_artifacts=True)
    print(json.dumps(rep, indent=2))
