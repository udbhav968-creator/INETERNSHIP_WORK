import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .assessment_blueprint import (
    MARKS_CORRECT,
    MARKS_WRONG,
    SUBJECT_ORDER,
    load_forms,
    load_question_bank,
)
from .irt_3pl_model import D_SCALING, theta_to_percentile
from .knowledge_graph_dag import compute_hierarchical_shrinkage_prior, compute_prerequisite_readiness
from .national_rank_simulator import score_to_all_india_rank, score_to_neet_percentile
from .state_machine.bkt_model import CHAPTER_BKT_REGISTRY, get_chapter_bkt_params, get_chapter_bkt_prior

MODEL_DIR = Path(__file__).resolve().parent
PROFILE_VERSION = "initial-profile-v1"
THETA_GRID = np.linspace(-4.0, 4.0, 161)
CHAPTER_THETA_PRIOR_SD = 0.6
PACE_THRESHOLD = 1.2
ADVANCED_THRESHOLD = 0.70
MEDIUM_THRESHOLD = 0.45
SCORE_DRAWS = 4000

with open(MODEL_DIR / "topic_importance.json", encoding="utf-8") as f:
    _IMPORTANCE = json.load(f)

_BANK: dict[str, dict] = {}


def _bank_by_id(source: str = "unified") -> dict[str, dict]:
    if source not in _BANK:
        _BANK[source] = {q["question_id"]: q for q in load_question_bank(source=source)}
    return _BANK[source]


def _p_correct(theta: np.ndarray, q: dict) -> np.ndarray:
    z = np.clip(D_SCALING * q["a"] * (theta - q["b"]), -30.0, 30.0)
    return q["c"] + (1.0 - q["c"]) / (1.0 + np.exp(-z))


def _eap(obs: list[tuple[dict, int]], prior_mean: float, prior_sd: float) -> tuple[float, float, np.ndarray]:
    log_post = -0.5 * ((THETA_GRID - prior_mean) / prior_sd) ** 2
    for q, y in obs:
        p = _p_correct(THETA_GRID, q)
        log_post += y * np.log(np.maximum(p, 1e-12)) + (1 - y) * np.log(np.maximum(1.0 - p, 1e-12))
    post = np.exp(log_post - log_post.max())
    post /= post.sum()
    mean = float(np.sum(THETA_GRID * post))
    sd = float(math.sqrt(max(1e-9, np.sum((THETA_GRID - mean) ** 2 * post))))
    return mean, sd, post


def _static_bkt_posterior(chapter: str, obs: list[tuple[dict, int]]) -> float:
    prior = get_chapter_bkt_prior(chapter)
    log_odds = math.log(prior / (1.0 - prior))
    for q, y in obs:
        p = get_chapter_bkt_params(chapter, q["tier"])
        g = min(0.45, max(0.05, float(p.p_guess)))
        s = min(0.45, max(0.03, float(p.p_slip)))
        log_odds += math.log((1.0 - s) / g) if y else math.log(s / (1.0 - g))
    return 1.0 / (1.0 + math.exp(-log_odds))


def _level(m: float) -> str:
    return "advanced" if m >= ADVANCED_THRESHOLD else ("medium" if m >= MEDIUM_THRESHOLD else "basic")


def _predicted_neet_score(post: np.ndarray, bank: dict[str, dict]) -> dict:
    forms = load_forms()["forms"]
    ref = [bank[qid] for qid in forms[0]["question_ids"] if qid in bank]
    p_grid = np.stack([_p_correct(THETA_GRID, q) for q in ref], axis=1)
    grid_tcc = (MARKS_CORRECT * p_grid + MARKS_WRONG * (1.0 - p_grid)).sum(axis=1)
    rng = np.random.default_rng(20260)
    draw_idx = rng.choice(len(THETA_GRID), size=SCORE_DRAWS, p=post)
    correct = rng.random((SCORE_DRAWS, len(ref))) < p_grid[draw_idx]
    scores = correct.sum(axis=1) * MARKS_CORRECT + (~correct).sum(axis=1) * MARKS_WRONG
    lo, med, hi = (float(np.percentile(scores, q)) for q in (5, 50, 95))
    return {
        "expected_marks": round(float(np.sum(post * grid_tcc)), 1),
        "interval_90": [int(round(lo)), int(round(hi))],
        "median_marks": int(round(med)),
        "neet_percentile_median": round(score_to_neet_percentile(med), 3),
        "all_india_rank_range": [score_to_all_india_rank(hi), score_to_all_india_rank(lo)],
        "basis": "posterior predictive: theta posterior draws x simulated responses on the equated 180-question form (all attempted)",
    }


def initialize_student_profile(responses: list[dict], source: str = "unified") -> dict:
    bank = _bank_by_id(source)
    obs_all: list[tuple[dict, int]] = []
    by_chapter: dict[str, list[tuple[dict, int]]] = defaultdict(list)
    times_by_chapter: dict[str, list[float]] = defaultdict(list)
    subject_of_chapter: dict[str, str] = {}
    marks = 0
    unknown = 0
    for r in responses:
        q = bank.get(r.get("question_id"))
        if q is None:
            unknown += 1
            continue
        if r.get("is_correct") is None:
            continue
        y = 1 if r["is_correct"] else 0
        obs_all.append((q, y))
        by_chapter[q["chapter"]].append((q, y))
        subject_of_chapter[q["chapter"]] = q["subject"]
        marks += MARKS_CORRECT if y else MARKS_WRONG
        t = r.get("time_taken_seconds")
        if y and t is not None:
            times_by_chapter[q["chapter"]].append(float(t) / max(1.0, float(q["ideal_time_sec"])))

    g_theta, g_sd, g_post = _eap(obs_all, 0.0, 1.0)
    all_chapters = sorted(set(CHAPTER_BKT_REGISTRY) | set(_IMPORTANCE) | set(by_chapter))
    observed_mastery: dict[str, float] = {}
    chapter_rows: dict[str, dict] = {}
    for ch in all_chapters:
        obs = by_chapter.get(ch, [])
        if not obs:
            continue
        th, sd, _ = _eap(obs, g_theta, CHAPTER_THETA_PRIOR_SD)
        m = _static_bkt_posterior(ch, obs)
        observed_mastery[ch] = m
        chapter_rows[ch] = {"theta": th, "theta_sd": sd, "raw_mastery": m, "n": len(obs), "correct": sum(y for _, y in obs)}

    topic_states = []
    for ch in all_chapters:
        row = chapter_rows.get(ch)
        n = row["n"] if row else 0
        m = compute_hierarchical_shrinkage_prior(ch, observed_mastery, attempts_count=n)
        m = float(min(0.95, max(0.05, m)))
        ci = round(min(0.45, 1.96 * math.sqrt(m * (1.0 - m) / (n + 2.0))), 4)
        ratios = times_by_chapter.get(ch, [])
        avg_ratio = round(float(np.mean(ratios)), 4) if ratios else None
        topic_states.append(
            {
                "topic_id": ch,
                "subject": subject_of_chapter.get(ch) or CHAPTER_BKT_REGISTRY.get(ch, {}).get("subject"),
                "level": _level(m),
                "mastery_prob": round(m, 4),
                "confidence_interval": ci,
                "irt_theta": round(row["theta"], 4) if row else round(g_theta, 4),
                "irt_theta_sd": round(row["theta_sd"], 4) if row else round(math.sqrt(g_sd ** 2 + CHAPTER_THETA_PRIOR_SD ** 2), 4),
                "avg_time_ratio": avg_ratio,
                "pace_flag": (None if avg_ratio is None else ("slow" if avg_ratio > PACE_THRESHOLD else "fast")),
                "attempts_counted": n,
                "correct_in_assessment": row["correct"] if row else 0,
                "observed": bool(row),
                "streak": 0,
                "last_3_results": [],
            }
        )

    mastery_map = {s["topic_id"]: s["mastery_prob"] for s in topic_states}
    priorities = []
    for s in topic_states:
        imp = float(_IMPORTANCE.get(s["topic_id"], 1.0))
        ready = compute_prerequisite_readiness(s["topic_id"], mastery_map)
        gap = 1.0 - s["mastery_prob"]
        priorities.append(
            {
                "topic_id": s["topic_id"],
                "subject": s["subject"],
                "mastery_prob": s["mastery_prob"],
                "importance": round(imp, 3),
                "prerequisite_readiness": round(ready, 4),
                "priority_score": round(imp * gap * (0.5 + 0.5 * ready), 4),
                "recommended_start_level": s["level"],
            }
        )
    priorities.sort(key=lambda p: p["priority_score"], reverse=True)

    subject_summary = {}
    for subj in SUBJECT_ORDER:
        sub = [s for s in topic_states if s["subject"] == subj]
        if sub:
            subject_summary[subj] = {
                "mean_mastery": round(float(np.mean([s["mastery_prob"] for s in sub])), 4),
                "chapters_basic": sum(1 for s in sub if s["level"] == "basic"),
                "chapters_medium": sum(1 for s in sub if s["level"] == "medium"),
                "chapters_advanced": sum(1 for s in sub if s["level"] == "advanced"),
            }

    return {
        "profile_version": PROFILE_VERSION,
        "responses_scored": len(obs_all),
        "unknown_question_ids": unknown,
        "marks": marks,
        "global_theta": round(g_theta, 4),
        "global_theta_sem": round(g_sd, 4),
        "theta_percentile": theta_to_percentile(g_theta),
        "predicted_neet_score": _predicted_neet_score(g_post, bank),
        "subject_summary": subject_summary,
        "topic_states": topic_states,
        "priority_chapters": priorities[:10],
    }
