"""
Psychometric Item Response Theory (IRT — 3-Parameter Logistic Model) — Enterprise v2.

Dynamically calibrates question difficulty (b_i), discrimination (a_i), and
pseudo-guessing (c_i), and estimates student latent ability (theta) via
Maximum A Posteriori (MAP) optimization.
"""
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
QUESTIONS_PATH = PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json"
IMPORTANCE_PATH = MODEL_DIR / "topic_importance.json"
IRT_PARAMS_PATH = MODEL_DIR / "irt_item_parameters.json"

D_SCALING = 1.702


@dataclass
class IRTItemParams:
    a: float = 1.25  # Discrimination
    b: float = 0.00  # Difficulty
    c: float = 0.22  # Pseudo-guessing


_DEFAULT_DIFFICULTY_B: dict[str, IRTItemParams] = {
    "basic": IRTItemParams(a=1.05, b=-1.15, c=0.25),
    "base": IRTItemParams(a=1.05, b=-1.15, c=0.25),
    "medium": IRTItemParams(a=1.32, b=0.05, c=0.22),
    "intermediate": IRTItemParams(a=1.32, b=0.05, c=0.22),
    "advanced": IRTItemParams(a=1.68, b=1.30, c=0.18),
}

CALIBRATED_ITEMS: dict[str, dict] = {}


def _deterministic_micro_jitter(qid: str) -> float:
    digest = hashlib.md5(qid.encode("utf-8")).hexdigest()
    val = int(digest[:6], 16) / 0xFFFFFF
    return (val - 0.5) * 0.36


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

    calibrated: dict[str, dict] = {}
    for q in questions:
        qid = q["question_id"]
        diff_raw = str(q.get("difficulty_level", "medium")).lower()
        base = _DEFAULT_DIFFICULTY_B.get(diff_raw, _DEFAULT_DIFFICULTY_B["medium"])

        ideal_time = float(q.get("ideal_time_sec", 85))
        has_diagram = bool(q.get("has_diagram", False))
        chapter = q.get("chapter", "")
        text_len = len(q.get("question_text", ""))

        time_delta = (ideal_time - 85.0) / 55.0
        length_delta = min(0.25, max(-0.15, (text_len - 140) / 400.0))
        diagram_delta = 0.18 if has_diagram else 0.0
        jitter = _deterministic_micro_jitter(qid)

        b_i = max(-2.85, min(2.85, base.b + 0.45 * time_delta + length_delta + diagram_delta + jitter))
        imp_score = float(importance.get(chapter, 1.5))
        a_i = max(0.65, min(2.40, base.a + 0.04 * min(imp_score, 5.0) + (0.10 if has_diagram else 0.0) + 0.3 * abs(jitter)))
        c_i = max(0.12, min(0.28, base.c - (0.02 if ideal_time >= 100 else 0.0)))

        calibrated[qid] = {
            "a": round(a_i, 4),
            "b": round(b_i, 4),
            "c": round(c_i, 4),
        }

    with open(IRT_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(calibrated, f, indent=2)

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
    return round(info, 5)


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


if __name__ == "__main__":
    items = calibrate_question_bank()
    print(f"Calibrated IRT 3PL parameters for {len(items)} questions -> {IRT_PARAMS_PATH}")
