import json
import sqlite3
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DB_FILE = PROJECT_ROOT / "backend" / "klassed.db"

QUESTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "tagged_questions"
    / "neet_questions_clean.json"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "training"
OUTPUT_FILE = OUTPUT_DIR / "student_attempts.csv"


def main():
    print("Loading question metadata...")

    with open(QUESTION_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    question_df = pd.DataFrame(questions)

    print(f"Questions loaded: {len(question_df)}")

    print("Loading student attempts from database...")

    db = sqlite3.connect(DB_FILE)

    attempts = pd.read_sql_query(
        """
        SELECT
            attempt_id,
            student_id,
            question_id,
            paper_number,
            is_correct,
            time_taken_seconds,
            selected_option,
            timestamp
        FROM test_attempts
        ORDER BY timestamp
        """,
        db,
    )

    db.close()

    print(f"Attempts loaded: {len(attempts)}")

    if attempts.empty:
        print("ERROR: No attempts found.")
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

    df = attempts.merge(
        question_metadata,
        on="question_id",
        how="left",
    )

    df["time_ratio"] = (
        df["time_taken_seconds"] / df["ideal_time_sec"]
    )

    print("\nTraining dataset:")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nMissing values after joining:")
    print(df.isnull().sum())

    print("\nCorrectness distribution:")
    print(df["is_correct"].value_counts(dropna=False))

    print("\nStudents:")
    print(df["student_id"].nunique())

    print("\nQuestions attempted:")
    print(df["question_id"].nunique())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    print("\nTraining dataset saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()