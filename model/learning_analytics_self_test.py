from __future__ import annotations

import numpy as np
import pandas as pd

from .anomaly_detection import _p3pl, detect_anomalies
from .decision_logging import POLICY_UNIFORM, rank_with_propensities
from .experiments import ARM_ADAPTIVE, ARM_CONTROL, analyse_selection_experiment
from .offline_policy_evaluation import estimate_policy_value

SEED = 20261116


def ope_self_test(n_decisions: int = 6000, n_candidates: int = 12) -> dict:
    rng = np.random.default_rng(SEED)
    rows, true_greedy = [], []
    for _ in range(n_decisions):
        true_r = rng.uniform(0.2, 0.9, n_candidates)
        scores = true_r + rng.normal(0, 0.1, n_candidates)
        ids = [f"Q{j}" for j in range(n_candidates)]
        slot = rank_with_propensities(ids, list(scores), POLICY_UNIFORM, rng=rng)[0]
        j = int(slot.question_id[1:])
        rows.append({"question_id": slot.question_id, "propensity": slot.propensity, "reward": float(rng.binomial(1, true_r[j])), "candidate_ids": ids, "candidate_scores": list(scores), "candidate_count": n_candidates})
        true_greedy.append(true_r[int(np.argmax(scores))])
    est = estimate_policy_value(rows, "greedy_linucb")
    truth = float(np.mean(true_greedy))
    ok = est["snips"] is not None and abs(est["snips"] - truth) < 0.03
    return {"true_value_greedy": round(truth, 4), "snips_estimate": est["snips"], "ips_estimate": est["ips"], "passed": bool(ok)}


def _simulated_attempts(n_students: int, effect: float, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    rows, arms = [], {}
    for s in range(n_students):
        sid = f"AB{s:05d}"
        arm = ARM_ADAPTIVE if s % 2 == 0 else ARM_CONTROL
        arms[sid] = arm
        base = rng.uniform(0.3, 0.7)
        gain = rng.normal(0.05 + (effect if arm == ARM_ADAPTIVE else 0.0), 0.08)
        for paper, p in ((1, base), (2, min(0.98, max(0.02, base + gain)))):
            for k in range(60):
                rows.append({"student_id": sid, "paper_number": paper, "is_correct": int(rng.random() < p)})
    return pd.DataFrame(rows), arms


def ab_self_test() -> dict:
    rng = np.random.default_rng(SEED)
    df_eff, arms_eff = _simulated_attempts(800, 0.05, rng)
    df_null, arms_null = _simulated_attempts(800, 0.0, rng)
    with_effect = analyse_selection_experiment(df_eff, arms_eff)
    no_effect = analyse_selection_experiment(df_null, arms_null)
    ok = with_effect["status"] == "ADAPTIVE_BETTER" and no_effect["status"] == "NO_SIGNIFICANT_DIFFERENCE"
    return {"planted_effect": with_effect["status"], "estimated_difference": with_effect["adaptive_minus_control"]["difference"], "no_effect": no_effect["status"], "passed": bool(ok)}


def anomaly_self_test() -> dict:
    rng = np.random.default_rng(3)
    items = {f"Q{i}": (float(np.exp(rng.normal(0, 0.3))), float(rng.normal()), 0.2) for i in range(80)}
    rows = []
    for s in range(120):
        th = rng.normal()
        for q, (a, b, c) in items.items():
            correct = bool(rng.random() < _p3pl(a, b, c, th))
            rows.append({"student_id": f"S{s}", "question_id": q, "paper_number": 1, "is_correct": int(correct), "time_taken_seconds": float(rng.lognormal(np.log(60), 0.3)), "selected_option": "a" if correct else str(rng.choice(list("bcd"))), "correct_answer": "a", "expected_time_seconds": 60})
    df = pd.DataFrame(rows)
    df.loc[df.student_id == "S0", "time_taken_seconds"] = 2.0
    src = df[df.student_id == "S1"].set_index("question_id")
    m2 = df.student_id == "S2"
    df.loc[m2, "selected_option"] = df.loc[m2, "question_id"].map(src["selected_option"]).values
    df.loc[m2, "is_correct"] = df.loc[m2, "question_id"].map(src["is_correct"]).values
    hard = {q: p[1] for q, p in items.items()}
    m3 = df.student_id == "S3"
    df.loc[m3, "is_correct"] = df.loc[m3, "question_id"].map(lambda q: int(hard[q] > 0.3)).values
    rep = detect_anomalies(df, items)
    flagged = {e["student_id"]: e["flags"] for e in rep["flagged_students"]}
    expected = {"S0": "rapid_guessing", "S1": "possible_answer_copying", "S2": "possible_answer_copying", "S3": "person_misfit"}
    found = all(v in flagged.get(k, []) for k, v in expected.items())
    false_positives = sorted(set(flagged) - set(expected))
    return {"planted_found": bool(found), "false_positives": false_positives, "passed": bool(found and len(false_positives) <= 2)}


def run_learning_analytics_self_test() -> dict:
    out = {"ope": ope_self_test(), "ab_test": ab_self_test(), "anomalies": anomaly_self_test()}
    out["passed"] = all(v["passed"] for v in out.values())
    return out
