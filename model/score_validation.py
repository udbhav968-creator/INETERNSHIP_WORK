from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MIN_STUDENTS = 100
MIN_DAYS_BEFORE_EXAM = 7


def pair_predictions_with_results(snapshots: pd.DataFrame, actuals: pd.DataFrame, min_days_before: int = MIN_DAYS_BEFORE_EXAM) -> pd.DataFrame:
    if snapshots is None or actuals is None or snapshots.empty or actuals.empty:
        return pd.DataFrame()
    snaps = snapshots.copy()
    snaps["student_id"] = snaps["student_id"].astype(str)
    snaps["snapshot_date"] = pd.to_datetime(snaps["snapshot_date"], errors="coerce")
    rows = []
    for _, a in actuals.iterrows():
        exam_date = pd.to_datetime(a.get("exam_date"), errors="coerce")
        if pd.isna(exam_date) or pd.isna(a.get("actual_score")):
            continue
        cutoff = exam_date - pd.Timedelta(days=min_days_before)
        s = snaps[(snaps["student_id"] == str(a["student_id"])) & (snaps["snapshot_date"] <= cutoff)]
        if s.empty:
            continue
        last = s.sort_values("snapshot_date").iloc[-1]
        rows.append(
            {
                "student_id": str(a["student_id"]),
                "exam_year": a.get("exam_year"),
                "days_before_exam": int((exam_date - last["snapshot_date"]).days),
                "predicted": float(last["expected_score"]),
                "low_90": float(last["low_90"]),
                "high_90": float(last["high_90"]),
                "actual": float(a["actual_score"]),
            }
        )
    return pd.DataFrame(rows)


def score_metrics(pairs: pd.DataFrame) -> dict:
    err = pairs["predicted"] - pairs["actual"]
    inside = (pairs["actual"] >= pairs["low_90"]) & (pairs["actual"] <= pairs["high_90"])
    rho = spearmanr(pairs["predicted"], pairs["actual"]).statistic if len(pairs) > 2 else None
    deciles = pd.qcut(pairs["predicted"], q=min(10, max(2, len(pairs) // 20)), duplicates="drop")
    by_decile = pairs.groupby(deciles, observed=True).agg(predicted=("predicted", "mean"), actual=("actual", "mean"), n=("actual", "size"))
    return {
        "n_students": int(len(pairs)),
        "mae": round(float(err.abs().mean()), 2),
        "rmse": round(float(np.sqrt((err**2).mean())), 2),
        "bias_predicted_minus_actual": round(float(err.mean()), 2),
        "coverage_of_90_range": round(float(inside.mean()), 4),
        "mean_range_width": round(float((pairs["high_90"] - pairs["low_90"]).mean()), 1),
        "spearman": None if rho is None else round(float(rho), 4),
        "calibration_by_decile": [
            {"predicted": round(float(r.predicted), 1), "actual": round(float(r.actual), 1), "n": int(r.n)} for r in by_decile.itertuples()
        ],
    }


def validate_score_predictions(snapshots: pd.DataFrame, actuals: pd.DataFrame, min_students: int = MIN_STUDENTS) -> dict:
    pairs = pair_predictions_with_results(snapshots, actuals)
    report = {
        "what": "latest score prediction made at least 7 days before the exam vs the student's real NEET score",
        "snapshots_available": 0 if snapshots is None else int(len(snapshots)),
        "actual_scores_available": 0 if actuals is None else int(len(actuals)),
        "paired_students": int(len(pairs)),
        "min_students": min_students,
        "how_to_supply_results": "data/training/neet_actual_scores.csv with columns student_id, exam_year, exam_date, actual_score (collected with consent)",
    }
    if len(pairs) < min_students:
        report["status"] = "INSUFFICIENT_DATA"
        return report
    report.update(score_metrics(pairs))
    cov = report["coverage_of_90_range"]
    report["status"] = "WELL_CALIBRATED" if 0.85 <= cov <= 0.95 else ("RANGE_TOO_NARROW" if cov < 0.85 else "RANGE_TOO_WIDE")
    return report
