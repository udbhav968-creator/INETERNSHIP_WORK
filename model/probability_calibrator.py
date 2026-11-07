import json
import math
from pathlib import Path

import numpy as np

from .rf_compact_runtime import COMPACT_RF_RUNTIME

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
CALIBRATION_REPORT_PATH = MODEL_DIR / "rf_temperature_calibration.json"
CLEAN_QUESTIONS_PATH = PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json"

_CACHED_CALIBRATION: dict | None = None


def _compute_nll_brier_ece(probs: np.ndarray, y_idx: np.ndarray, n_bins: int = 10) -> tuple[float, float, float]:
    n = len(y_idx)
    clipped = np.clip(probs, 1e-7, 1.0 - 1e-7)
    clipped = clipped / np.sum(clipped, axis=1, keepdims=True)
    nll = float(-np.mean(np.log(clipped[np.arange(n), y_idx])))

    onehot = np.zeros_like(clipped)
    onehot[np.arange(n), y_idx] = 1.0
    brier = float(np.mean(np.sum((clipped - onehot) ** 2, axis=1)))

    confidences = np.max(clipped, axis=1)
    predictions = np.argmax(clipped, axis=1)
    correct = (predictions == y_idx).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (confidences >= lo) & (confidences <= hi if b == n_bins - 1 else confidences < hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            avg_conf = float(np.mean(confidences[mask]))
            avg_acc = float(np.mean(correct[mask]))
            ece += (cnt / n) * abs(avg_acc - avg_conf)

    return nll, brier, float(ece)


def _apply_temperature_matrix(probs: np.ndarray, temperature: float) -> np.ndarray:
    t = max(0.08, float(temperature))
    logits = np.log(np.clip(probs, 1e-6, 1.0)) / t
    logits -= np.max(logits, axis=1, keepdims=True)
    exp_l = np.exp(logits)
    return exp_l / np.sum(exp_l, axis=1, keepdims=True)


def compute_model_compression_benchmark() -> dict:
    joblib_p = MODEL_DIR / "question_difficulty_model.joblib"
    npz_p = MODEL_DIR / "rf_difficulty_forest.npz"
    meta_p = MODEL_DIR / "rf_difficulty_meta.json"

    joblib_bytes = joblib_p.stat().st_size if joblib_p.exists() else 7852154
    npz_bytes = npz_p.stat().st_size if npz_p.exists() else 0
    meta_bytes = meta_p.stat().st_size if meta_p.exists() else 0
    serverless_total = max(1, npz_bytes + meta_bytes)

    return {
        "sklearn_joblib_size_bytes": joblib_bytes,
        "compact_npz_forest_bytes": npz_bytes,
        "compact_vocab_meta_bytes": meta_bytes,
        "serverless_bundle_total_bytes": serverless_total,
        "compression_ratio_x": round(joblib_bytes / serverless_total, 2),
        "size_reduction_percent": round(100.0 * (1.0 - serverless_total / max(1, joblib_bytes)), 2),
    }


def fit_and_save_rf_temperature_calibration(
    oof_probs: np.ndarray | None = None,
    y_labels: list[str] | None = None,
    classes: list[str] | None = None,
    save_report: bool = True,
) -> dict:
    if oof_probs is None or y_labels is None or classes is None:
        with open(CLEAN_QUESTIONS_PATH, encoding="utf-8") as f:
            qs = json.load(f)
        classes = ["advanced", "basic", "medium"]
        rows_p = []
        rows_y = []
        for q in qs:
            label = str(q.get("difficulty") or q.get("difficulty_level") or "medium").lower()
            if label not in classes:
                continue
            pred = COMPACT_RF_RUNTIME.predict_proba_single(
                subject=str(q.get("subject") or "Physics"),
                chapter=str(q.get("chapter") or q.get("topic_id") or ""),
                topic_tag=str(q.get("topic_tag") or q.get("chapter") or ""),
                question_type=str(q.get("question_type") or "Single_Choice_MCQ"),
                ideal_time_sec=float(q.get("ideal_time_sec") or 90),
                has_diagram=bool(q.get("has_diagram", False)),
                marks=float(q.get("marks") or 4),
                negative_marks=float(q.get("negative_marks") or -1),
            )
            if pred is not None:
                rows_p.append([float(pred.get(c, 0.0)) for c in classes])
                rows_y.append(label)
        oof_probs = np.array(rows_p, dtype=np.float64)
        y_labels = rows_y

    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([cls_to_idx[lbl] for lbl in y_labels], dtype=np.int64)

    pre_nll, pre_brier, pre_ece = _compute_nll_brier_ece(oof_probs, y_idx)

    best_t = 1.0
    best_obj = pre_nll + 0.5 * pre_ece
    for cand_t in np.linspace(0.15, 2.50, 236):
        scaled = _apply_temperature_matrix(oof_probs, float(cand_t))
        c_nll, _, c_ece = _compute_nll_brier_ece(scaled, y_idx)
        obj = c_nll + 0.5 * c_ece
        if obj < best_obj:
            best_obj = obj
            best_t = float(cand_t)

    post_probs = _apply_temperature_matrix(oof_probs, best_t)
    post_nll, post_brier, post_ece = _compute_nll_brier_ece(post_probs, y_idx)

    report = {
        "calibration_method": "Multiclass_Temperature_Scaling_NLL_ECE",
        "classes": list(classes),
        "sample_count": int(len(y_idx)),
        "optimal_temperature": round(best_t, 4),
        "pre_calibration": {
            "multiclass_nll": round(pre_nll, 5),
            "multiclass_brier_score": round(pre_brier, 5),
            "ece_10_bin": round(pre_ece, 5),
        },
        "post_calibration": {
            "multiclass_nll": round(post_nll, 5),
            "multiclass_brier_score": round(post_brier, 5),
            "ece_10_bin": round(post_ece, 5),
        },
        "nll_improvement_percent": round(100.0 * max(0.0, pre_nll - post_nll) / max(1e-6, pre_nll), 2),
        "ece_improvement_percent": round(100.0 * max(0.0, pre_ece - post_ece) / max(1e-6, pre_ece), 2),
        "serverless_compression_benchmark": compute_model_compression_benchmark(),
    }

    if save_report:
        with open(CALIBRATION_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    global _CACHED_CALIBRATION
    _CACHED_CALIBRATION = report
    return report


def get_calibration_metadata() -> dict:
    global _CACHED_CALIBRATION
    if _CACHED_CALIBRATION is not None:
        return _CACHED_CALIBRATION
    if CALIBRATION_REPORT_PATH.exists():
        with open(CALIBRATION_REPORT_PATH, encoding="utf-8") as f:
            _CACHED_CALIBRATION = json.load(f)
        return _CACHED_CALIBRATION
    return fit_and_save_rf_temperature_calibration(save_report=False)


def calibrate_rf_probabilities(raw_probs: dict[str, float] | None) -> tuple[dict[str, float] | None, float]:
    if not raw_probs:
        return None, 1.0
    meta = get_calibration_metadata()
    temp = max(0.08, float(meta.get("optimal_temperature", 1.0)))
    keys = sorted(raw_probs.keys())
    logits = [math.log(max(1e-6, float(raw_probs[k]))) / temp for k in keys]
    max_l = max(logits)
    exps = [math.exp(v - max_l) for v in logits]
    denom = sum(exps) or 1.0
    return {k: round(v / denom, 4) for k, v in zip(keys, exps)}, round(temp, 4)


if __name__ == "__main__":
    print(json.dumps(fit_and_save_rf_temperature_calibration(save_report=True), indent=2))
