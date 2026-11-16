from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = MODEL_DIR / "student_models" / "empirical_item_times.json"
MIN_RESPONSES_PER_ITEM = 30
SHRINKAGE_K = 20.0
MIN_SECONDS = 3.0
MAX_SECONDS = 900.0


def estimate_item_times(attempts: pd.DataFrame, min_responses: int = MIN_RESPONSES_PER_ITEM, shrinkage_k: float = SHRINKAGE_K) -> dict:
    if attempts is None or attempts.empty or "time_taken_seconds" not in attempts:
        return {"status": "INSUFFICIENT_DATA", "items": {}, "items_estimated": 0}
    df = attempts.copy()
    df["t"] = pd.to_numeric(df["time_taken_seconds"], errors="coerce")
    df = df[(df["t"] >= MIN_SECONDS) & (df["t"] <= MAX_SECONDS)]
    if "expected_time_seconds" not in df:
        df["expected_time_seconds"] = np.nan
    items = {}
    log_ratio_all = []
    for qid, g in df.groupby("question_id"):
        prior = float(g["expected_time_seconds"].dropna().iloc[0]) if g["expected_time_seconds"].notna().any() else float(g["t"].median())
        n = int(len(g))
        log_t = np.log(g["t"].to_numpy())
        if n >= min_responses:
            log_ratio_all.append(float(np.median(log_t) - np.log(max(1.0, prior))))
        w = n / (n + shrinkage_k)
        est = float(np.exp(w * np.median(log_t) + (1 - w) * np.log(max(1.0, prior))))
        items[str(qid)] = {
            "responses": n,
            "formula_seconds": round(prior, 1),
            "observed_median_seconds": round(float(np.median(g["t"])), 1),
            "observed_p25_p75": [round(float(np.percentile(g["t"], 25)), 1), round(float(np.percentile(g["t"], 75)), 1)],
            "estimated_seconds": round(est, 1),
            "shrinkage_weight_on_data": round(w, 3),
            "usable": n >= min_responses,
        }
    usable = sum(1 for v in items.values() if v["usable"])
    return {
        "status": "ESTIMATED" if usable else "INSUFFICIENT_DATA",
        "method": f"median log-time shrunk toward the formula value with weight n/(n+{shrinkage_k:g}); times outside {MIN_SECONDS:g}-{MAX_SECONDS:g}s dropped",
        "items_with_any_data": len(items),
        "items_estimated": usable,
        "min_responses_per_item": min_responses,
        "median_observed_over_formula": round(float(np.exp(np.median(log_ratio_all))), 3) if log_ratio_all else None,
        "items": items,
    }


def save_item_times(report: dict, path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
