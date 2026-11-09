from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

from .irt_3pl_model import get_item_params, IRTItemParams

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
DB_PATH = PROJECT_ROOT / "backend" / "klassed.db"
CLEAN_QUESTIONS_PATH = PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json"
MIN_RESPONSES_FOR_EMPIRICAL_CALIBRATION = 30


def _connect_db(custom_path: Path | None = None) -> sqlite3.Connection | None:
    target = custom_path or DB_PATH
    if not target.exists():
        return None
    try:
        conn = sqlite3.connect(str(target))
        conn.row_factory = sqlite3.Row
        return conn
    except Exception:
        return None


def get_item_response_stats(question_id: str, custom_db: Path | None = None) -> dict[str, Any]:
    conn = _connect_db(custom_db)
    if conn is None:
        return {"n": 0, "correct": 0, "accuracy": 0.0, "avg_time_seconds": 0.0}
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) AS n, "
            "COALESCE(SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END), 0) AS correct, "
            "COALESCE(AVG(time_taken_seconds), 0.0) AS avg_time "
            "FROM test_attempts WHERE question_id = ?",
            (question_id,),
        )
        row = cur.fetchone()
        n = int(row["n"]) if row else 0
        correct = int(row["correct"]) if row else 0
        avg_t = float(row["avg_time"]) if row else 0.0
        acc = round(correct / n, 4) if n > 0 else 0.0
        return {"n": n, "correct": correct, "accuracy": acc, "avg_time_seconds": round(avg_t, 2)}
    finally:
        conn.close()


def get_item_calibration_profile(question_id: str, custom_db: Path | None = None) -> dict[str, Any]:
    stats = get_item_response_stats(question_id, custom_db)
    n = stats["n"]
    params = get_item_params(question_id)
    b_val = float(params.b)
    a_val = float(params.a)
    c_val = float(params.c)

    if n == 0:
        sem_b = 1.0000
        status = "COLD_START_HEURISTIC"
        is_empirical = False
        confidence = 0.0
    elif n < MIN_RESPONSES_FOR_EMPIRICAL_CALIBRATION:
        sem_b = round(1.0 / math.sqrt(1.0 + 0.15 * n), 4)
        status = "PILOT_ACCUMULATING"
        is_empirical = False
        confidence = round((n / float(MIN_RESPONSES_FOR_EMPIRICAL_CALIBRATION)) * 100.0, 1)
    else:
        sem_b = round(1.0 / math.sqrt(1.0 + 0.25 * n), 4)
        status = "EMPIRICALLY_CALIBRATED"
        is_empirical = True
        confidence = 100.0

    ci_low = round(b_val - 1.96 * sem_b, 4)
    ci_high = round(b_val + 1.96 * sem_b, 4)

    return {
        "question_id": question_id,
        "sample_size": n,
        "observed_accuracy": stats["accuracy"],
        "avg_time_seconds": stats["avg_time_seconds"],
        "calibration_status": status,
        "is_empirically_calibrated": is_empirical,
        "empirical_confidence_pct": confidence,
        "item_parameters": {"a": a_val, "b": b_val, "c": c_val},
        "parameter_uncertainty": {
            "sem_b": sem_b,
            "b_95_ci": [ci_low, ci_high],
            "uncertainty_level": "HIGH" if sem_b >= 0.70 else ("MODERATE" if sem_b >= 0.35 else "LOW"),
        },
        "min_responses_required": MIN_RESPONSES_FOR_EMPIRICAL_CALIBRATION,
    }


def get_pipeline_response_audit(custom_db: Path | None = None) -> dict[str, Any]:
    total_q = 2400
    if CLEAN_QUESTIONS_PATH.exists():
        try:
            raw = json.loads(CLEAN_QUESTIONS_PATH.read_text(encoding="utf-8"))
            total_q = len(raw)
        except Exception:
            pass

    conn = _connect_db(custom_db)
    if conn is None:
        return {
            "total_questions_in_bank": total_q,
            "total_attempts_recorded": 0,
            "unobserved_heuristic_items": total_q,
            "accumulating_pilot_items": 0,
            "empirically_calibrated_items": 0,
            "min_threshold_per_item": MIN_RESPONSES_FOR_EMPIRICAL_CALIBRATION,
            "pipeline_status": "AWAITING_RESPONSES",
        }

    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM test_attempts")
        tot_attempts = int(cur.fetchone()[0])

        cur.execute(
            "SELECT question_id, COUNT(*) AS cnt FROM test_attempts GROUP BY question_id"
        )
        counts = {row["question_id"]: int(row["cnt"]) for row in cur.fetchall()}
    finally:
        conn.close()

    emp_calibrated = sum(1 for c in counts.values() if c >= MIN_RESPONSES_FOR_EMPIRICAL_CALIBRATION)
    accumulating = sum(1 for c in counts.values() if 1 <= c < MIN_RESPONSES_FOR_EMPIRICAL_CALIBRATION)
    unobserved = max(0, total_q - len(counts))

    return {
        "total_questions_in_bank": total_q,
        "total_attempts_recorded": tot_attempts,
        "unobserved_heuristic_items": unobserved,
        "accumulating_pilot_items": accumulating,
        "empirically_calibrated_items": emp_calibrated,
        "min_threshold_per_item": MIN_RESPONSES_FOR_EMPIRICAL_CALIBRATION,
        "pipeline_status": "EMPIRICALLY_ACTIVE" if emp_calibrated > 0 else "AWAITING_RESPONSES",
    }


def should_recalibrate_empirically(question_id: str, custom_db: Path | None = None) -> bool:
    stats = get_item_response_stats(question_id, custom_db)
    return stats["n"] >= MIN_RESPONSES_FOR_EMPIRICAL_CALIBRATION
