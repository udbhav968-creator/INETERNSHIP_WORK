import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .rf_compact_runtime import COMPACT_RF_RUNTIME

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
QUESTIONS_PATH = PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json"
ATTEMPTS_CSV_PATH = PROJECT_ROOT / "data" / "training" / "student_attempts.csv"
IMPORTANCE_PATH = MODEL_DIR / "topic_importance.json"
IRT_PARAMS_PATH = MODEL_DIR / "irt_item_parameters.json"
IRT_RIDGE_PATH = MODEL_DIR / "irt_cold_start_ridge.npz"

D_SCALING = 1.702


@dataclass
class IRTItemParams:
    a: float = 1.25
    b: float = 0.00
    c: float = 0.22


_DEFAULT_DIFFICULTY_B: dict[str, IRTItemParams] = {
    "basic": IRTItemParams(a=1.05, b=-1.15, c=0.25),
    "base": IRTItemParams(a=1.05, b=-1.15, c=0.25),
    "medium": IRTItemParams(a=1.32, b=0.05, c=0.22),
    "intermediate": IRTItemParams(a=1.32, b=0.05, c=0.22),
    "advanced": IRTItemParams(a=1.68, b=1.30, c=0.18),
}

QUESTION_TYPE_MODIFIERS: dict[str, tuple[float, float]] = {
    "Numerical": (0.18, 0.22),
    "Statement-Based": (0.14, 0.16),
    "Statement_Based": (0.14, 0.16),
    "Match-the-Column": (0.10, 0.08),
    "Match_The_Column": (0.10, 0.08),
    "Assertion-Reason": (0.16, 0.20),
    "Assertion_Reason": (0.16, 0.20),
    "Diagram-Based": (0.15, 0.18),
    "Diagram_Based": (0.15, 0.18),
    "Conceptual": (0.04, 0.00),
    "Single_Choice_MCQ": (0.04, 0.00),
    "Direct Formula": (-0.05, -0.12),
}

CALIBRATED_ITEMS: dict[str, dict] = {}


def _load_empirical_attempt_stats() -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    if not ATTEMPTS_CSV_PATH.exists():
        return stats
    try:
        with open(ATTEMPTS_CSV_PATH, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            counts: dict[str, list[tuple[int, float]]] = {}
            for row in reader:
                qid = row.get("question_id", "")
                if not qid:
                    continue
                corr = 1 if str(row.get("is_correct", "")).lower() in ("1", "true") else 0
                t_ratio = float(row.get("time_ratio") or 1.0)
                counts.setdefault(qid, []).append((corr, t_ratio))
            for qid, pairs in counts.items():
                n = len(pairs)
                acc = sum(c for c, _ in pairs) / n
                avg_tr = sum(tr for _, tr in pairs) / n
                stats[qid] = {"n": float(n), "accuracy": acc, "avg_time_ratio": avg_tr}
    except Exception:
        pass
    return stats


def calibrate_question_bank() -> dict[str, dict]:
    global CALIBRATED_ITEMS
    if not QUESTIONS_PATH.exists():
        return {}

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    importance = {}
    if IMPORTANCE_PATH.exists():
        with open(IMPORTANCE_PATH, encoding="utf-8") as f:
            importance = json.load(f)

    empirical_stats = _load_empirical_attempt_stats()

    calibrated: dict[str, dict] = {}
    for q in questions:
        qid = q["question_id"]
        diff_raw = str(q.get("difficulty_level", "medium")).lower()
        base = _DEFAULT_DIFFICULTY_B.get(diff_raw, _DEFAULT_DIFFICULTY_B["medium"])

        ideal_time = float(q.get("ideal_time_sec", 85))
        has_diagram = bool(q.get("has_diagram", False))
        chapter = q.get("chapter", "")
        q_type = str(q.get("question_type", "Conceptual"))
        q_text = str(q.get("question_text", ""))
        explanation = str(q.get("explanation", ""))

        time_delta = (ideal_time - 85.0) / 55.0
        length_delta = min(0.25, max(-0.15, (len(q_text) - 140.0) / 400.0))
        expl_steps = max(1, explanation.count(".") + explanation.count("="))
        expl_delta = min(0.22, (expl_steps - 2.0) * 0.04)
        diagram_delta = 0.18 if has_diagram else 0.0
        a_type_mod, b_type_mod = QUESTION_TYPE_MODIFIERS.get(q_type, (0.05, 0.02))

        opt_lengths = [
            len(str(q.get(k, ""))) for k in ("option_a", "option_b", "option_c", "option_d")
        ]
        mean_opt_len = sum(opt_lengths) / 4.0 if opt_lengths else 10.0
        opt_var = sum((ol - mean_opt_len) ** 2 for ol in opt_lengths) / 4.0
        distractor_spread = min(0.04, math.sqrt(opt_var) / 250.0)

        emp_b_shift = 0.0
        if qid in empirical_stats:
            emp = empirical_stats[qid]
            weight = min(0.35, emp["n"] * 0.10)
            emp_b_shift = weight * ((0.60 - emp["accuracy"]) * 1.5 + (emp["avg_time_ratio"] - 1.0) * 0.4)

        b_i = max(
            -2.85,
            min(
                2.85,
                base.b + 0.45 * time_delta + length_delta + expl_delta + diagram_delta + b_type_mod + emp_b_shift,
            ),
        )
        imp_score = float(importance.get(chapter, 1.5))
        a_i = max(
            0.65,
            min(
                2.40,
                base.a
                + 0.04 * min(imp_score, 5.0)
                + (0.10 if has_diagram else 0.0)
                + a_type_mod
                + 0.02 * min(expl_steps, 6),
            ),
        )
        c_i = max(
            0.12,
            min(
                0.28,
                base.c - (0.02 if ideal_time >= 100 else 0.0) - distractor_spread,
            ),
        )

        calibrated[qid] = {
            "a": round(a_i, 4),
            "b": round(b_i, 4),
            "c": round(c_i, 4),
        }

    try:
        with open(IRT_PARAMS_PATH, "w", encoding="utf-8") as f:
            json.dump(calibrated, f, indent=2)
    except OSError:
        pass

    CALIBRATED_ITEMS = calibrated
    return calibrated


if IRT_PARAMS_PATH.exists():
    try:
        with open(IRT_PARAMS_PATH, encoding="utf-8") as f:
            CALIBRATED_ITEMS = json.load(f)
    except Exception:
        CALIBRATED_ITEMS = calibrate_question_bank()
else:
    CALIBRATED_ITEMS = calibrate_question_bank()


_IRT_RIDGE_WEIGHTS: np.ndarray | None = None
_IRT_RIDGE_MEAN: np.ndarray | None = None
_IRT_RIDGE_STD: np.ndarray | None = None


def reload_irt_ridge_model() -> bool:
    global _IRT_RIDGE_WEIGHTS, _IRT_RIDGE_MEAN, _IRT_RIDGE_STD
    if not IRT_RIDGE_PATH.exists():
        return False
    try:
        arrs = np.load(IRT_RIDGE_PATH)
        _IRT_RIDGE_WEIGHTS = arrs["weights"].astype(np.float64)
        _IRT_RIDGE_MEAN = arrs["feature_mean"].astype(np.float64)
        _IRT_RIDGE_STD = arrs["feature_std"].astype(np.float64)
        return True
    except Exception:
        return False


reload_irt_ridge_model()


def get_item_params(
    question_id: str, difficulty: str = "medium", expected_time_seconds: int = 90
) -> IRTItemParams:
    if question_id in CALIBRATED_ITEMS:
        raw = CALIBRATED_ITEMS[question_id]
        return IRTItemParams(
            a=float(raw.get("a", 1.25)),
            b=float(raw.get("b", 0.0)),
            c=float(raw.get("c", 0.22)),
        )
    base = _DEFAULT_DIFFICULTY_B.get(difficulty, _DEFAULT_DIFFICULTY_B["medium"])
    time_offset = (expected_time_seconds - 85.0) / 60.0
    b_val = max(-3.0, min(3.0, base.b + 0.35 * time_offset))
    return IRTItemParams(a=base.a, b=round(b_val, 4), c=base.c)


def irt_3pl_probability(theta: float, params: IRTItemParams) -> float:
    logit = -D_SCALING * params.a * (theta - params.b)
    logit = max(-30.0, min(30.0, logit))
    p_star = 1.0 / (1.0 + math.exp(logit))
    return params.c + (1.0 - params.c) * p_star


def fisher_information(theta: float, params: IRTItemParams) -> float:
    p = irt_3pl_probability(theta, params)
    q = 1.0 - p
    if p <= params.c or p >= 0.9999 or q <= 1e-6:
        return 0.001
    ratio = (p - params.c) / (1.0 - params.c)
    info = (D_SCALING ** 2) * (params.a ** 2) * (q / p) * (ratio ** 2)
    return round(info, 8)


def compute_posterior_sem(
    theta: float, items: list[IRTItemParams], prior_var: float = 1.5
) -> dict:
    test_info = (1.0 / prior_var) + sum(fisher_information(theta, it) for it in items)
    sem = 1.0 / math.sqrt(max(0.05, test_info))
    ci_95_low = round(theta - 1.96 * sem, 4)
    ci_95_high = round(theta + 1.96 * sem, 4)
    is_calibrated = sem <= 0.32
    return {
        "theta": round(theta, 4),
        "test_information": round(test_info, 4),
        "sem": round(sem, 4),
        "ci_95_bounds": [ci_95_low, ci_95_high],
        "precision_status": "HIGH" if sem <= 0.32 else ("MODERATE" if sem <= 0.48 else "EXPLORATORY"),
        "stopping_criterion_met": is_calibrated,
    }


def kullback_leibler_item_information(
    theta: float, params: IRTItemParams, delta: float = 0.5, num_points: int = 5
) -> float:
    p0 = irt_3pl_probability(theta, params)
    p0 = max(1e-5, min(1.0 - 1e-5, p0))
    grid = [theta - delta + (2.0 * delta * i) / max(1, num_points - 1) for i in range(num_points)]
    total_kl = 0.0
    for th in grid:
        pt = irt_3pl_probability(th, params)
        pt = max(1e-5, min(1.0 - 1e-5, pt))
        kl = p0 * math.log(p0 / pt) + (1.0 - p0) * math.log((1.0 - p0) / (1.0 - pt))
        total_kl += kl
    return round(total_kl / num_points, 6)


def update_student_theta_map(
    current_theta: float,
    is_correct: bool,
    params: IRTItemParams,
    prior_mean: float = 0.0,
    prior_var: float = 1.5,
    learning_rate_cap: float = 0.45,
) -> float:
    theta = float(current_theta if current_theta is not None else 0.0)
    p = irt_3pl_probability(theta, params)
    y = 1.0 if is_correct else 0.0

    p_star = (p - params.c) / max(1e-6, (1.0 - params.c))
    grad_ll = D_SCALING * params.a * p_star * ((y - p) / max(1e-6, p * (1.0 - p)))
    grad_prior = -(theta - prior_mean) / prior_var
    first_deriv = grad_ll + grad_prior

    info = fisher_information(theta, params)
    hessian_diag = info + (1.0 / prior_var)

    step = first_deriv / max(0.25, hessian_diag)
    step = max(-learning_rate_cap, min(learning_rate_cap, step))

    return round(max(-3.5, min(3.5, theta + step)), 4)


def theta_to_percentile(theta: float) -> float:
    cdf = 0.5 * (1.0 + math.erf(theta / math.sqrt(2.0)))
    return round(max(1.0, min(99.9, cdf * 100.0)), 1)


def extract_cold_start_nlp_features(question_text: str, options: list[str] | None = None) -> dict[str, float]:
    text = str(question_text or "")
    opts = [str(o) for o in (options or []) if o is not None]
    words = re.findall(r"[a-zA-Z]{2,}", text.lower())
    word_count = len(words)

    formula_tokens = re.findall(r"[=+\-*/^<>]|\\frac|\\sqrt|\b(?:sin|cos|tan|log|ln|lim|mol|kg|Hz|ohm|volt|joule)\b", text)
    formula_density = min(1.0, len(formula_tokens) / max(8.0, word_count * 0.35))

    numeric_tokens = re.findall(r"\b\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?\b", text)
    numerical_complexity = min(1.0, len(numeric_tokens) / 4.0)

    negation_matches = re.findall(
        r"\b(?:not|incorrect|false|except|wrong|never|neither|assertion|reason|match\s+the)\b",
        text.lower(),
    )
    negation_trap = 1.0 if negation_matches else 0.0

    opt_token_sets = [set(re.findall(r"[a-zA-Z0-9]+", o.lower())) for o in opts if o.strip()]
    pair_sims: list[float] = []
    for i in range(len(opt_token_sets)):
        for j in range(i + 1, len(opt_token_sets)):
            union = len(opt_token_sets[i] | opt_token_sets[j])
            if union > 0:
                pair_sims.append(len(opt_token_sets[i] & opt_token_sets[j]) / union)
    distractor_similarity = sum(pair_sims) / len(pair_sims) if pair_sims else 0.25

    return {
        "word_count": float(word_count),
        "formula_density": round(formula_density, 4),
        "numerical_complexity": round(numerical_complexity, 4),
        "negation_trap": negation_trap,
        "distractor_similarity": round(distractor_similarity, 4),
    }


_RF_PIPELINE = None


def reload_sklearn_rf_pipeline() -> None:
    global _RF_PIPELINE
    try:
        import joblib

        rf_path = MODEL_DIR / "question_difficulty_model.joblib"
        if rf_path.exists():
            _RF_PIPELINE = joblib.load(rf_path)
    except Exception:
        _RF_PIPELINE = None


reload_sklearn_rf_pipeline()


def predict_cold_start_difficulty_and_irt(
    question_text: str,
    options: list[str] | None = None,
    has_diagram: bool = False,
    subject: str = "Physics",
    chapter: str = "Laws of Motion",
    topic_tag: str = "Newton's Laws",
    question_type: str | None = None,
    ideal_time_sec: int | None = None,
) -> dict:
    feats = extract_cold_start_nlp_features(question_text, options)
    char_len = float(len(str(question_text or "")))
    length_norm = min(1.0, feats["word_count"] / 55.0)
    diag_bonus = 0.20 if has_diagram else 0.0

    nlp_score = (
        0.30 * feats["formula_density"]
        + 0.30 * feats["numerical_complexity"]
        + 0.15 * feats["negation_trap"]
        + 0.15 * length_norm
        + 0.10 * feats["distractor_similarity"]
        + diag_bonus
    )

    text_low = (question_text or "").lower()
    if question_type and question_type.strip():
        resolved_qtype = question_type.strip()
    elif has_diagram:
        resolved_qtype = "Diagram_Based"
    elif "assertion" in text_low and "reason" in text_low:
        resolved_qtype = "Assertion_Reason"
    elif "match" in text_low and ("column" in text_low or "list" in text_low):
        resolved_qtype = "Match_The_Column"
    else:
        resolved_qtype = "Single_Choice_MCQ"

    resolved_time = (
        int(ideal_time_sec)
        if ideal_time_sec is not None and int(ideal_time_sec) > 0
        else int(round(45 + 75 * min(1.0, nlp_score)))
    )

    rf_probs: dict[str, float] | None = None
    rf_engine = "unavailable"
    complexity_score = nlp_score

    if _RF_PIPELINE is not None:
        try:
            import pandas as pd

            row_df = pd.DataFrame(
                [
                    {
                        "subject": subject,
                        "chapter": chapter,
                        "topic_tag": topic_tag,
                        "question_type": resolved_qtype,
                        "ideal_time_sec": resolved_time,
                        "has_diagram": int(bool(has_diagram)),
                        "word_count": feats["word_count"],
                        "char_length": char_len,
                        "formula_density": feats["formula_density"],
                        "numerical_complexity": feats["numerical_complexity"],
                        "negation_trap": feats["negation_trap"],
                        "distractor_similarity": feats["distractor_similarity"],
                        "marks": 4,
                        "negative_marks": -1,
                    }
                ]
            )
            probas = _RF_PIPELINE.predict_proba(row_df)[0]
            classes = list(_RF_PIPELINE.classes_)
            rf_probs = {str(c): round(float(p), 4) for c, p in zip(classes, probas)}
            rf_engine = "sklearn_300_tree_joblib"
        except Exception:
            rf_probs = None

    if rf_probs is None and COMPACT_RF_RUNTIME.loaded:
        probs_compact = COMPACT_RF_RUNTIME.predict_proba_single(
            subject=subject,
            chapter=chapter,
            topic_tag=topic_tag,
            question_type=resolved_qtype,
            ideal_time_sec=resolved_time,
            has_diagram=bool(has_diagram),
            word_count=feats["word_count"],
            char_length=char_len,
            formula_density=feats["formula_density"],
            numerical_complexity=feats["numerical_complexity"],
            negation_trap=feats["negation_trap"],
            distractor_similarity=feats["distractor_similarity"],
            marks=4.0,
            negative_marks=-1.0,
        )
        if probs_compact:
            rf_probs = probs_compact
            rf_engine = "numpy_300_tree_compact_forest"

    rf_predicted_tier: str | None = None
    if rf_probs:
        rf_predicted_tier = max(rf_probs.items(), key=lambda kv: kv[1])[0]
        rf_expected = (
            0.15 * rf_probs.get("basic", 0.0)
            + 0.45 * rf_probs.get("medium", 0.0)
            + 0.85 * rf_probs.get("advanced", 0.0)
        )
        complexity_score = 0.65 * nlp_score + 0.35 * rf_expected

    if complexity_score < 0.28:
        tier = "basic"
    elif complexity_score < 0.56:
        tier = "medium"
    else:
        tier = "advanced"

    if _IRT_RIDGE_WEIGHTS is not None and _IRT_RIDGE_MEAN is not None and _IRT_RIDGE_STD is not None:
        expl_steps_est = max(1.0, round(1.0 + 3.0 * complexity_score))
        tier_num = {"basic": 0.0, "medium": 0.5, "advanced": 1.0}.get(tier, 0.5)
        x_raw = np.array(
            [
                feats["word_count"],
                char_len,
                feats["formula_density"],
                feats["numerical_complexity"],
                feats["negation_trap"],
                feats["distractor_similarity"],
                expl_steps_est,
                1.0 if has_diagram else 0.0,
                float(resolved_time),
                1.0 if subject == "Biology" else 0.0,
                1.0 if subject == "Chemistry" else 0.0,
                1.0 if subject == "Physics" else 0.0,
                tier_num,
            ],
            dtype=np.float64,
        )
        x_aug = np.concatenate([[1.0], (x_raw - _IRT_RIDGE_MEAN) / _IRT_RIDGE_STD])
        pred_abc = x_aug @ _IRT_RIDGE_WEIGHTS
        a_est = round(float(np.clip(pred_abc[0], 0.65, 2.40)), 4)
        b_est = round(float(np.clip(pred_abc[1], -2.85, 2.85)), 4)
        c_est = round(float(np.clip(pred_abc[2], 0.12, 0.28)), 4)
    else:
        b_est = round(max(-2.5, min(2.5, (complexity_score - 0.42) * 3.8)), 4)
        a_est = round(max(0.75, min(2.35, 1.05 + 0.65 * complexity_score + 0.25 * feats["distractor_similarity"])), 4)
        c_est = round(max(0.14, min(0.26, 0.25 - 0.06 * feats["distractor_similarity"] - 0.04 * feats["numerical_complexity"])), 4)

    from .probability_calibrator import calibrate_rf_probabilities
    calibrated_probs, cal_temp = calibrate_rf_probabilities(rf_probs)

    return {
        "predicted_difficulty": tier,
        "rf_predicted_difficulty": rf_predicted_tier,
        "rf_engine": rf_engine,
        "irt_estimator": "multi_output_ridge_npz" if _IRT_RIDGE_WEIGHTS is not None else "analytic_fallback",
        "complexity_score": round(complexity_score, 4),
        "irt_params": {"a": a_est, "b": b_est, "c": c_est},
        "nlp_features": feats,
        "rf_probabilities": rf_probs,
        "calibrated_probabilities": calibrated_probs,
        "calibration_temperature": cal_temp,
        "resolved_metadata": {
            "subject": subject,
            "chapter": chapter,
            "topic_tag": topic_tag,
            "question_type": resolved_qtype,
            "ideal_time_sec": resolved_time,
            "has_diagram": bool(has_diagram),
        },
    }


def classify_cognitive_skill_dimension(question_text: str, question_type: str = "") -> str:
    feats = extract_cold_start_nlp_features(question_text)
    qt = (question_type or "").lower()
    if "numerical" in qt or feats["numerical_complexity"] >= 0.50:
        return "numerical"
    if "formula" in qt or feats["formula_density"] >= 0.25 or "=" in (question_text or ""):
        return "formula"
    return "conceptual"


def compute_mirt_3d_profile(attempt_records: list[dict]) -> dict[str, float]:
    dims = {"conceptual": 0.0, "formula": 0.0, "numerical": 0.0}
    counts = {"conceptual": 0, "formula": 0, "numerical": 0}

    for rec in attempt_records:
        q_text = str(rec.get("question_text", ""))
        q_type = str(rec.get("question_type", ""))
        dim = classify_cognitive_skill_dimension(q_text, q_type)
        params = get_item_params(
            str(rec.get("question_id", "")),
            str(rec.get("difficulty", "medium")),
            int(rec.get("expected_time_seconds") or 90),
        )
        dims[dim] = update_student_theta_map(
            current_theta=dims[dim],
            is_correct=bool(rec.get("is_correct", False)),
            params=params,
        )
        counts[dim] += 1

    return {
        "theta_conceptual": round(dims["conceptual"], 4),
        "theta_formula": round(dims["formula"], 4),
        "theta_numerical": round(dims["numerical"], 4),
        "attempts_conceptual": counts["conceptual"],
        "attempts_formula": counts["formula"],
        "attempts_numerical": counts["numerical"],
    }


def recalibrate_item_parameters_online(
    question_id: str,
    student_theta: float,
    is_correct: bool,
    time_ratio: float = 1.0,
    lr: float = 0.06,
) -> IRTItemParams:
    current = get_item_params(question_id)
    pred_p = irt_3pl_probability(student_theta, current)
    y = 1.0 if is_correct else 0.0
    residual = y - pred_p

    new_b = max(-2.85, min(2.85, current.b - lr * residual + 0.02 * (time_ratio - 1.0)))
    alignment = (student_theta - current.b) * residual
    new_a = max(0.65, min(2.40, current.a + 0.5 * lr * max(-0.2, min(0.2, alignment))))

    updated = IRTItemParams(a=round(new_a, 4), b=round(new_b, 4), c=current.c)
    CALIBRATED_ITEMS[question_id] = {"a": updated.a, "b": updated.b, "c": updated.c}
    return updated


if __name__ == "__main__":
    items = calibrate_question_bank()
    print(f"Calibrated IRT 3PL parameters for {len(items)} questions -> {IRT_PARAMS_PATH}")
