from __future__ import annotations

import math
from typing import Any
import numpy as np

SCORE_ANCHORS = np.array([0.0, 80.0, 130.0, 165.0, 200.0, 300.0, 400.0, 500.0, 550.0, 600.0, 650.0, 680.0, 700.0, 715.0, 720.0], dtype=np.float64)
PERCENTILE_ANCHORS = np.array([0.0, 12.0, 27.0, 38.5, 48.0, 71.0, 85.5, 93.8, 96.5, 98.6, 99.65, 99.92, 99.988, 99.998, 100.0], dtype=np.float64)

NEET_YEAR_FACTS = {
    2026: {"registered": 2279743, "appeared": 1999895, "qualified": 1121185, "general_cutoff_score": 213, "topper_score": 715},
    2025: {"registered": 2276069, "appeared": 2209318, "qualified": 1236531, "general_cutoff_score": 144},
    2024: {"registered": 2406079, "appeared": 2333297, "general_cutoff_score": 164},
}
REFERENCE_EXAM_YEAR = 2026
TOTAL_NEET_CANDIDATES = NEET_YEAR_FACTS[REFERENCE_EXAM_YEAR]["appeared"]

QUALIFYING_CUTOFF_SOURCE = {
    "exam_year": REFERENCE_EXAM_YEAR,
    "category": "General/EWS 50th percentile",
    "general_cutoff_score": NEET_YEAR_FACTS[REFERENCE_EXAM_YEAR]["general_cutoff_score"],
    "source": "NTA NEET UG 2026 result press coverage (re-exam June 21, result July 16, 2026)",
}

MODEL_ASSUMPTIONS = {
    "candidate_count": {"value": TOTAL_NEET_CANDIDATES, "status": "sourced", "basis": "NEET UG 2026 candidates appeared"},
    "qualifying_cutoff": {"value": QUALIFYING_CUTOFF_SOURCE["general_cutoff_score"], "status": "sourced", "basis": QUALIFYING_CUTOFF_SOURCE["source"]},
    "score_to_percentile_curve": {"status": "approximate", "basis": "hand-set piecewise-linear anchors; NTA does not publish a full score-percentile table; year-to-year shifts of tens of marks are normal"},
    "college_tier_thresholds": {"status": "approximate", "basis": "indicative closing scores/ranks from past counselling rounds; vary by year, category, quota and round"},
    "rank_formula": {"status": "derived", "basis": "rank = (1 - percentile/100) x candidates appeared"},
}

COLLEGE_TIERS = [
    {"id": "aiims_delhi", "name": "AIIMS New Delhi", "score_threshold": 705, "air_threshold": 55},
    {"id": "central_institutes", "name": "Central Institutes (MAMC, VMMC, JIPMER, AIIMS Top 5)", "score_threshold": 680, "air_threshold": 1200},
    {"id": "top_state_govt", "name": "Top State Govt Medical Colleges", "score_threshold": 630, "air_threshold": 15000},
    {"id": "peripheral_govt", "name": "All-India Govt MBBS Floor", "score_threshold": 595, "air_threshold": 32000},
    {"id": "state_quota_govt", "name": "State Quota Govt Medical Seats", "score_threshold": 540, "air_threshold": 75000},
    {"id": "deemed_semi_govt", "name": "Deemed / Semi-Govt Merit Seats", "score_threshold": 480, "air_threshold": 150000},
    {"id": "qualifying_floor", "name": "NEET Qualifying Cutoff", "score_threshold": QUALIFYING_CUTOFF_SOURCE["general_cutoff_score"], "air_threshold": NEET_YEAR_FACTS[REFERENCE_EXAM_YEAR]["qualified"]},
]


def score_to_neet_percentile(score: float) -> float:
    clamped = max(0.0, min(720.0, float(score)))
    return float(np.interp(clamped, SCORE_ANCHORS, PERCENTILE_ANCHORS))


def score_to_all_india_rank(score: float, n_candidates: int = TOTAL_NEET_CANDIDATES) -> int:
    pct = score_to_neet_percentile(score)
    fraction_above = (100.0 - pct) / 100.0
    rank = int(round(fraction_above * n_candidates))
    return max(1, rank)


def simulate_national_rank_and_college_cutoffs(
    total_scores_trials: np.ndarray,
    conformal_score_interval: list[int] | tuple[int, int],
    mean_score: int | float,
    n_candidates: int = TOTAL_NEET_CANDIDATES,
) -> dict[str, Any]:
    low_score, high_score = conformal_score_interval
    air_best = score_to_all_india_rank(float(high_score), n_candidates=n_candidates)
    air_worst = score_to_all_india_rank(float(low_score), n_candidates=n_candidates)
    air_median = score_to_all_india_rank(float(mean_score), n_candidates=n_candidates)

    pct_low = round(score_to_neet_percentile(float(low_score)), 3)
    pct_high = round(score_to_neet_percentile(float(high_score)), 3)
    pct_median = round(score_to_neet_percentile(float(mean_score)), 3)

    college_admissions: list[dict[str, Any]] = []
    n_trials = len(total_scores_trials) if len(total_scores_trials) > 0 else 1

    for tier in COLLEGE_TIERS:
        th = tier["score_threshold"]
        prob = float(np.mean(total_scores_trials >= th) * 100.0) if len(total_scores_trials) > 0 else (100.0 if mean_score >= th else 0.0)
        college_admissions.append({
            "college_id": tier["id"],
            "college_tier": tier["name"],
            "target_score_720": th,
            "target_air_cutoff": tier["air_threshold"],
            "admission_probability_percent": round(prob, 1),
            "threshold_status": "sourced" if tier["id"] == "qualifying_floor" else "approximate",
            "status": "SECURE" if prob >= 80.0 else ("COMPETITIVE" if prob >= 40.0 else ("REACH" if prob >= 10.0 else "UNLIKELY")),
        })

    return {
        "total_national_candidates": n_candidates,
        "conformal_air_interval": [air_best, air_worst],
        "projected_air_median": air_median,
        "conformal_percentile_interval": [pct_low, pct_high],
        "projected_percentile_median": pct_median,
        "college_admission_probabilities": college_admissions,
        "reference_exam_year": REFERENCE_EXAM_YEAR,
        "model_assumptions": MODEL_ASSUMPTIONS,
    }
