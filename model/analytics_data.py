from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
SQLITE_DB = PROJECT_ROOT / "backend" / "klassed.db"
TRAINING_DIR = PROJECT_ROOT / "data" / "training"
ACTUAL_SCORES_CSV = TRAINING_DIR / "neet_actual_scores.csv"
STUDENT_ATTRIBUTES_CSV = TRAINING_DIR / "student_attributes.csv"


def database_url() -> str | None:
    url = os.environ.get("DATABASE_URL") or os.environ.get("KLASSED_DB_URL")
    if url and url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    if url and url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    if not url and SQLITE_DB.exists():
        url = f"sqlite:///{SQLITE_DB.as_posix()}"
    return url


def read_table(sql: str, url: str | None = None) -> pd.DataFrame:
    url = url or database_url()
    if not url:
        return pd.DataFrame()
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(url)
        with engine.connect() as conn:
            return pd.read_sql(text(sql), conn)
    except Exception:
        return pd.DataFrame()


def load_full_attempts(url: str | None = None) -> pd.DataFrame:
    df = read_table(
        "SELECT a.attempt_id, a.student_id, a.question_id, a.paper_number, a.is_correct, a.time_taken_seconds, "
        "a.selected_option, a.option_switches, a.hesitation_ms, a.dwell_time_ms, a.timestamp, "
        "q.correct_answer, q.expected_time_seconds, q.difficulty, q.topic_id "
        "FROM test_attempts a JOIN questions q ON q.question_id = a.question_id",
        url,
    )
    if df.empty:
        return df
    df["student_id"] = df["student_id"].astype(str)
    df["question_id"] = df["question_id"].astype(str)
    df["is_correct"] = df["is_correct"].map(lambda v: 1 if str(v).strip().lower() in {"1", "true", "t", "yes"} else 0)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return df.sort_values(["student_id", "timestamp", "attempt_id"]).reset_index(drop=True)


def load_decisions(url: str | None = None) -> list[dict]:
    df = read_table(
        "SELECT student_id, question_id, topic_id, policy, arm, epsilon, propensity, chosen_score, "
        "candidate_ids, candidate_scores, candidate_count, reward FROM selection_decisions",
        url,
    )
    if df.empty:
        return []
    import json

    rows = df.to_dict("records")
    for r in rows:
        for k in ("candidate_ids", "candidate_scores"):
            if isinstance(r.get(k), str):
                try:
                    r[k] = json.loads(r[k])
                except ValueError:
                    r[k] = []
        if r.get("reward") is not None and r["reward"] != r["reward"]:
            r["reward"] = None
    return rows


def load_score_snapshots(url: str | None = None) -> pd.DataFrame:
    return read_table(
        "SELECT student_id, snapshot_date, expected_score, low_90, high_90, attempts_so_far FROM score_prediction_snapshots",
        url,
    )


def load_actual_scores(path: Path = ACTUAL_SCORES_CSV) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["student_id", "exam_year", "exam_date", "actual_score"])
    df = pd.read_csv(path)
    df["student_id"] = df["student_id"].astype(str)
    return df


def load_student_attributes(path: Path = STUDENT_ATTRIBUTES_CSV, url: str | None = None) -> pd.DataFrame:
    base = read_table("SELECT student_id, class_level, target_neet_year FROM students", url)
    if not base.empty:
        base["student_id"] = base["student_id"].astype(str)
    if path.exists():
        extra = pd.read_csv(path)
        extra["student_id"] = extra["student_id"].astype(str)
        base = extra if base.empty else base.merge(extra, on="student_id", how="outer")
    return base
