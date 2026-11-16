from __future__ import annotations

import numpy as np

from .decision_logging import POLICY_UNIFORM

MIN_DECISIONS = 2000


def _target_prob_greedy(row: dict) -> float:
    scores = [s for s in row.get("candidate_scores") or [] if s is not None]
    ids = row.get("candidate_ids") or []
    if not scores or len(scores) != len(ids):
        return float("nan")
    best = ids[int(np.argmax(scores))]
    return 1.0 if row["question_id"] == best else 0.0


def _target_prob_uniform(row: dict) -> float:
    n = int(row.get("candidate_count") or 0)
    return 1.0 / n if n > 0 else float("nan")


TARGET_POLICIES = {"greedy_linucb": _target_prob_greedy, "uniform_random": _target_prob_uniform}


def estimate_policy_value(rows: list[dict], target: str, clip: float = 20.0) -> dict:
    fn = TARGET_POLICIES[target]
    w, r = [], []
    for row in rows:
        p_log = float(row.get("propensity") or 0.0)
        reward = row.get("reward")
        if p_log <= 0 or reward is None:
            continue
        p_t = fn(row)
        if p_t != p_t:
            continue
        w.append(min(clip, p_t / p_log))
        r.append(float(reward))
    w_arr, r_arr = np.asarray(w, dtype=float), np.asarray(r, dtype=float)
    n = int(w_arr.size)
    if n == 0:
        return {"target_policy": target, "n": 0, "ips": None, "snips": None}
    ips = float(np.mean(w_arr * r_arr))
    snips = float(np.sum(w_arr * r_arr) / np.sum(w_arr)) if np.sum(w_arr) > 0 else None
    ess = float(np.sum(w_arr) ** 2 / np.sum(w_arr**2)) if np.sum(w_arr**2) > 0 else 0.0
    se = float(np.std(w_arr * r_arr, ddof=1) / np.sqrt(n)) if n > 1 else None
    return {
        "target_policy": target,
        "n": n,
        "ips": round(ips, 5),
        "ips_ci95": None if se is None else [round(ips - 1.96 * se, 5), round(ips + 1.96 * se, 5)],
        "snips": None if snips is None else round(snips, 5),
        "effective_sample_size": round(ess, 1),
        "weight_clip": clip,
    }


def evaluate_logged_decisions(rows: list[dict], min_decisions: int = MIN_DECISIONS) -> dict:
    rewarded = [r for r in rows if r.get("reward") is not None]
    exploring = [r for r in rewarded if float(r.get("propensity") or 0) < 1.0]
    uniform_logged = [r for r in rewarded if r.get("policy") == POLICY_UNIFORM]
    report = {
        "decisions_logged": len(rows),
        "decisions_with_reward": len(rewarded),
        "decisions_with_exploration": len(exploring),
        "uniform_random_decisions": len(uniform_logged),
        "min_decisions": min_decisions,
        "note": "Unbiased estimates need logged propensities below 1: run the A/B control arm or set KLASSED_BANDIT_EXPLORATION_EPSILON > 0.",
    }
    if len(exploring) < min_decisions:
        report["status"] = "INSUFFICIENT_DATA"
        return report
    report["policies"] = {name: estimate_policy_value(exploring, name) for name in TARGET_POLICIES}
    observed = np.array([float(r["reward"]) for r in rewarded])
    report["logged_policy_mean_reward"] = round(float(observed.mean()), 5)
    report["status"] = "EVALUATED"
    return report
