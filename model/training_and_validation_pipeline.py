import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .corpus_distribution_and_readiness_audit import run_corpus_distribution_and_readiness_audit
from .irt_3pl_model import CALIBRATED_ITEMS, calibrate_question_bank
from .knowledge_graph_dag import DAG_NODES, get_all_ancestors
from .rag_socratic_engine import _INDEX as RAG_INDEX
from .rf_compact_runtime import COMPACT_RF_RUNTIME, export_rf_compact_forest, reload_compact_rf_runtime
from .validate_question_bank import validate as validate_qbank
from .validate_weightage_file import validate as validate_weightage

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
DATA_FILE = PROJECT_ROOT / "data" / "training" / "question_features.csv"
MODEL_FILE = MODEL_DIR / "question_difficulty_model.joblib"
METRICS_FILE = MODEL_DIR / "question_difficulty_metrics.json"
VALIDATION_REPORT_FILE = MODEL_DIR / "validation_report.json"
REGISTRY_MANIFEST_FILE = MODEL_DIR / "model_registry_manifest.json"

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


def _artifact_canonical_size_and_sha256(path: Path) -> tuple[int, str]:
    if path.suffix.lower() in {".json", ".csv"}:
        canonical = path.read_bytes().replace(b"\r\n", b"\n")
        return len(canonical), hashlib.sha256(canonical).hexdigest()
    h = hashlib.sha256()
    total = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            total += len(chunk)
            h.update(chunk)
    return total, h.hexdigest()


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


def _build_et_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )
    clf = ExtraTreesClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", clf)])


def _build_logreg_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    clf = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", clf)])


def build_model_registry_manifest() -> dict:
    tracked_paths = [
        MODEL_FILE,
        MODEL_DIR / "rf_difficulty_forest.npz",
        MODEL_DIR / "rf_difficulty_meta.json",
        MODEL_DIR / "irt_item_parameters.json",
        MODEL_DIR / "topic_importance.json",
        PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json",
        PROJECT_ROOT / "data" / "ncert_knowledge_base.json",
        PROJECT_ROOT / "data" / "taxonomy" / "NEET_Unified_Weightage_And_Marks_2013_2026.json",
        DATA_FILE,
    ]
    artifacts: dict[str, dict] = {}
    for p in tracked_paths:
        rel_key = p.relative_to(PROJECT_ROOT).as_posix()
        c_size, c_hash = _artifact_canonical_size_and_sha256(p) if p.exists() else (0, "")
        artifacts[rel_key] = {
            "exists": p.exists(),
            "size_bytes": c_size,
            "sha256": c_hash,
        }
    return {
        "registry_version": "2.1.0",
        "champion_classifier": "RandomForestClassifier_300_Balanced",
        "serverless_runtime": "CompactNumPyForest_300_Trees",
        "artifacts": artifacts,
    }


def run_deep_training_and_validation(save_artifacts: bool = True) -> dict:
    t0 = time.perf_counter()

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

    for topic_id in DAG_NODES:
        ancestors = get_all_ancestors(topic_id)
        if topic_id in ancestors:
            raise RuntimeError(f"Cycle detected in Prerequisite DAG at node: {topic_id}")

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

    for _, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        pipe = _build_rf_pipeline()
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = pipe.predict(X.iloc[val_idx])
        probs = pipe.predict_proba(X.iloc[val_idx])

        oof_preds[val_idx] = preds
        oof_probs[val_idx] = probs

        fold_accuracies.append(float(accuracy_score(y.iloc[val_idx], preds)))
        fold_macro_f1s.append(float(f1_score(y.iloc[val_idx], preds, average="macro")))
        fold_log_losses.append(float(log_loss(y.iloc[val_idx], probs, labels=classes)))

    y_onehot = np.zeros_like(oof_probs)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for idx, label in enumerate(y):
        y_onehot[idx, class_to_idx[label]] = 1.0
    multiclass_brier = float(np.mean(np.sum((oof_probs - y_onehot) ** 2, axis=1)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    challenger_pipe = _build_rf_pipeline()
    challenger_pipe.fit(X_train, y_train)
    challenger_preds = challenger_pipe.predict(X_test)
    challenger_probs = challenger_pipe.predict_proba(X_test)
    challenger_acc = float(accuracy_score(y_test, challenger_preds))

    et_pipe = _build_et_pipeline()
    et_pipe.fit(X_train, y_train)
    et_preds = et_pipe.predict(X_test)
    et_probs = et_pipe.predict_proba(X_test)

    lr_pipe = _build_logreg_pipeline()
    lr_pipe.fit(X_train, y_train)
    lr_preds = lr_pipe.predict(X_test)
    lr_probs = lr_pipe.predict_proba(X_test)

    tournament = {
        "random_forest_300_balanced": {
            "holdout_accuracy": round(challenger_acc, 4),
            "macro_f1": round(float(f1_score(y_test, challenger_preds, average="macro")), 4),
            "log_loss": round(float(log_loss(y_test, challenger_probs, labels=classes)), 4),
        },
        "extra_trees_300_balanced": {
            "holdout_accuracy": round(float(accuracy_score(y_test, et_preds)), 4),
            "macro_f1": round(float(f1_score(y_test, et_preds, average="macro")), 4),
            "log_loss": round(float(log_loss(y_test, et_probs, labels=classes)), 4),
        },
        "multinomial_logistic_l2": {
            "holdout_accuracy": round(float(accuracy_score(y_test, lr_preds)), 4),
            "macro_f1": round(float(f1_score(y_test, lr_preds, average="macro")), 4),
            "log_loss": round(float(log_loss(y_test, lr_probs, labels=classes)), 4),
        },
    }

    champion_pipe = challenger_pipe
    test_preds = challenger_preds
    holdout_accuracy = challenger_acc
    challenger_promoted = True

    if MODEL_FILE.exists():
        try:
            existing_pipe = joblib.load(MODEL_FILE)
            existing_preds = existing_pipe.predict(X_test)
            existing_acc = float(accuracy_score(y_test, existing_preds))
            if existing_acc >= challenger_acc:
                champion_pipe = existing_pipe
                test_preds = existing_preds
                holdout_accuracy = existing_acc
                challenger_promoted = False
        except Exception:
            pass

    holdout_correct = int((test_preds == y_test).sum())
    holdout_total = len(y_test)

    if holdout_accuracy < MIN_ACCURACY_GATE:
        raise RuntimeError(
            f"Model failed accuracy gate: {holdout_accuracy:.4f} < {MIN_ACCURACY_GATE:.4f}"
        )

    perm = permutation_importance(
        champion_pipe, X_test, y_test, n_repeats=5, random_state=42, scoring="accuracy"
    )
    perm_importances = {
        feat: {
            "mean_accuracy_drop": round(float(perm.importances_mean[i]), 5),
            "std_accuracy_drop": round(float(perm.importances_std[i]), 5),
        }
        for i, feat in enumerate(FEATURES)
    }

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

    if save_artifacts:
        export_rf_compact_forest(champion_pipe)
        reload_compact_rf_runtime()
        if challenger_promoted:
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

    sk_all_probs = champion_pipe.predict_proba(X)
    champ_classes = [str(c) for c in champion_pipe.classes_]
    max_parity_diff = 0.0
    exact_argmax_matches = 0
    for idx, row in enumerate(X.itertuples(index=False)):
        np_p = COMPACT_RF_RUNTIME.predict_proba_single(
            subject=str(row.subject),
            chapter=str(row.chapter),
            topic_tag=str(row.topic_tag),
            question_type=str(row.question_type),
            ideal_time_sec=float(row.ideal_time_sec),
            has_diagram=bool(row.has_diagram),
            marks=float(row.marks),
            negative_marks=float(row.negative_marks),
        )
        sk_row = {c: float(p) for c, p in zip(champ_classes, sk_all_probs[idx])}
        if np_p is not None:
            for c in champ_classes:
                max_parity_diff = max(max_parity_diff, abs(sk_row[c] - np_p[c]))
            if max(sk_row, key=sk_row.get) == max(np_p, key=np_p.get):
                exact_argmax_matches += 1

    if max_parity_diff >= 1e-3 or exact_argmax_matches != len(X):
        raise RuntimeError(
            f"Serverless RF parity gate failed: max_diff={max_parity_diff}, matches={exact_argmax_matches}/{len(X)}"
        )

    irt_items = CALIBRATED_ITEMS if len(CALIBRATED_ITEMS) >= 2471 else calibrate_question_bank()
    a_vals = [float(v["a"] if isinstance(v, dict) else v.a) for v in irt_items.values()]
    b_vals = [float(v["b"] if isinstance(v, dict) else v.b) for v in irt_items.values()]
    c_vals = [float(v["c"] if isinstance(v, dict) else v.c) for v in irt_items.values()]

    dist_audit = run_corpus_distribution_and_readiness_audit(save_report=save_artifacts)

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
            "adjacent_fallback_covers_all_65_chapters": dist_audit["gap_summary"][
                "adjacent_tier_fallback_resolves_100_percent_topics"
            ],
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
        "champion_challenger_tournament": tournament,
        "permutation_feature_importance": perm_importances,
        "serverless_compact_rf_parity": {
            "verified_rows": len(X),
            "exact_class_matches": exact_argmax_matches,
            "max_abs_probability_diff": round(max_parity_diff, 6),
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
        manifest = build_model_registry_manifest()
        with open(REGISTRY_MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    return report


if __name__ == "__main__":
    rep = run_deep_training_and_validation(save_artifacts=True)
    print(json.dumps(rep, indent=2))
