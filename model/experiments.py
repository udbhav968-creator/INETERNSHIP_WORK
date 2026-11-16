from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass

import numpy as np

SELECTION_EXPERIMENT = "selection_policy_v1"
ARM_ADAPTIVE = "adaptive"
ARM_CONTROL = "random_control"


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    arms: tuple[str, ...]
    weights: tuple[float, ...]
    salt: str
    env_flag: str


EXPERIMENTS: dict[str, ExperimentSpec] = {
    SELECTION_EXPERIMENT: ExperimentSpec(
        name=SELECTION_EXPERIMENT,
        arms=(ARM_ADAPTIVE, ARM_CONTROL),
        weights=(0.5, 0.5),
        salt="klassed-selection-2026",
        env_flag="KLASSED_EXPERIMENT_SELECTION_AB",
    )
}


def experiment_enabled(name: str = SELECTION_EXPERIMENT) -> bool:
    spec = EXPERIMENTS[name]
    return os.environ.get(spec.env_flag, "off").strip().lower() in ("1", "on", "true", "yes")


def _unit_hash(student_id: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{student_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def assign_arm(student_id: str, name: str = SELECTION_EXPERIMENT, force_enabled: bool | None = None) -> str:
    spec = EXPERIMENTS[name]
    enabled = experiment_enabled(name) if force_enabled is None else force_enabled
    if not enabled:
        return spec.arms[0]
    u = _unit_hash(str(student_id), spec.salt)
    cum = 0.0
    for arm, w in zip(spec.arms, spec.weights):
        cum += w
        if u < cum:
            return arm
    return spec.arms[-1]


def sample_ratio_mismatch(counts: dict[str, int], name: str = SELECTION_EXPERIMENT) -> dict:
    from scipy.stats import chisquare

    spec = EXPERIMENTS[name]
    observed = np.array([float(counts.get(a, 0)) for a in spec.arms])
    total = float(observed.sum())
    if total <= 0:
        return {"total": 0, "p_value": None, "srm_detected": None}
    expected = np.array(spec.weights) * total
    stat, p = chisquare(observed, expected)
    return {
        "total": int(total),
        "observed": {a: int(c) for a, c in zip(spec.arms, observed)},
        "expected": {a: round(float(e), 1) for a, e in zip(spec.arms, expected)},
        "chi2": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "srm_detected": bool(p < 0.001),
    }


def bootstrap_mean_difference(a: np.ndarray, b: np.ndarray, n_boot: int = 4000, seed: int = 20261116) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = rng.choice(a, a.size).mean() - rng.choice(b, b.size).mean()
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0) if a.size > 1 and b.size > 1 else float("nan")
    return {
        "difference": round(float(a.mean() - b.mean()), 5),
        "ci95": [round(float(lo), 5), round(float(hi), 5)],
        "cohens_d": round(float((a.mean() - b.mean()) / pooled), 4) if pooled and pooled == pooled and pooled > 0 else None,
        "significant": bool(lo > 0 or hi < 0),
    }


def minimum_detectable_effect(n_per_arm: int, sd: float, alpha: float = 0.05, power: float = 0.8) -> float | None:
    from scipy.stats import norm

    if n_per_arm < 2 or sd <= 0:
        return None
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return round(float(z * sd * math.sqrt(2.0 / n_per_arm)), 5)


def learning_gain_per_student(attempts, min_papers: int = 2) -> dict[str, float]:
    if attempts is None or len(attempts) == 0:
        return {}
    df = attempts.copy()
    df["is_correct"] = df["is_correct"].astype(float)
    per_paper = df.groupby(["student_id", "paper_number"])["is_correct"].mean().reset_index()
    gains: dict[str, float] = {}
    for sid, grp in per_paper.groupby("student_id"):
        grp = grp.sort_values("paper_number")
        if len(grp) < min_papers:
            continue
        gains[str(sid)] = float(grp["is_correct"].iloc[-1] - grp["is_correct"].iloc[0])
    return gains


def analyse_selection_experiment(attempts, arms_by_student: dict[str, str], min_students_per_arm: int = 100) -> dict:
    spec = EXPERIMENTS[SELECTION_EXPERIMENT]
    counts = {a: 0 for a in spec.arms}
    for arm in arms_by_student.values():
        if arm in counts:
            counts[arm] += 1
    srm = sample_ratio_mismatch(counts)
    gains = learning_gain_per_student(attempts)
    by_arm = {a: np.array([g for s, g in gains.items() if arms_by_student.get(s) == a]) for a in spec.arms}
    n = {a: int(v.size) for a, v in by_arm.items()}
    report = {
        "experiment": SELECTION_EXPERIMENT,
        "outcome": "change in accuracy from a student's first to last paper",
        "students_assigned": counts,
        "students_with_outcome": n,
        "sample_ratio_check": srm,
        "min_students_per_arm": min_students_per_arm,
    }
    if min(n.values()) < min_students_per_arm:
        report["status"] = "INSUFFICIENT_DATA"
        return report
    a, b = by_arm[ARM_ADAPTIVE], by_arm[ARM_CONTROL]
    report["mean_gain"] = {ARM_ADAPTIVE: round(float(a.mean()), 5), ARM_CONTROL: round(float(b.mean()), 5)}
    report["adaptive_minus_control"] = bootstrap_mean_difference(a, b)
    sd = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0))
    report["minimum_detectable_effect"] = minimum_detectable_effect(min(n.values()), sd)
    if srm.get("srm_detected"):
        report["status"] = "INVALID_SAMPLE_RATIO_MISMATCH"
    else:
        report["status"] = "ADAPTIVE_BETTER" if report["adaptive_minus_control"]["ci95"][0] > 0 else (
            "CONTROL_BETTER" if report["adaptive_minus_control"]["ci95"][1] < 0 else "NO_SIGNIFICANT_DIFFERENCE"
        )
    return report
