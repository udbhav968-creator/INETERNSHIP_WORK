import json
import sqlite3
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_FILE = PROJECT_ROOT / "backend" / "klassed.db"
QUESTION_FILE = PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "training"
OUTPUT_FILE = OUTPUT_DIR / "student_attempts.csv"

EXPECTED_COLUMNS = [
    "attempt_id",
    "student_id",
    "question_id",
    "paper_number",
    "is_correct",
    "time_taken_seconds",
    "selected_option",
    "timestamp",
    "paper_source",
    "subject",
    "chapter",
    "topic_tag",
    "skill",
    "difficulty_level",
    "question_type",
    "ideal_time_sec",
    "has_diagram",
    "marks",
    "negative_marks",
    "time_ratio",
]


def main():
    with open(QUESTION_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    question_df = pd.DataFrame(questions)

    db = sqlite3.connect(DB_FILE)
    query = (
        "SELECT attempt_id, student_id, question_id, paper_number, "
        "is_correct, time_taken_seconds, selected_option, timestamp "
        "FROM test_attempts ORDER BY timestamp"
    )
    attempts = pd.read_sql_query(query, db)
    db.close()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if attempts.empty:
        pd.DataFrame(columns=EXPECTED_COLUMNS).to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8",
        )
        print(f"No live attempts in DB; wrote clean header schema to {OUTPUT_FILE}")
        return

    metadata_columns = [
        "question_id",
        "paper_source",
        "subject",
        "chapter",
        "topic_tag",
        "skill",
        "difficulty_level",
        "question_type",
        "ideal_time_sec",
        "has_diagram",
        "marks",
        "negative_marks",
    ]
    question_metadata = question_df[metadata_columns]
    df = attempts.merge(question_metadata, on="question_id", how="left")
    df["time_ratio"] = df["time_taken_seconds"] / df["ideal_time_sec"]
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"Saved {len(df)} attempt rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
