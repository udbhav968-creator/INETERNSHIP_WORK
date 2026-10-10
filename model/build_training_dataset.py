import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "tagged_questions"
    / "neet_questions_clean.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "question_features.csv"
)


def main():
    print("Loading question bank...")

    with open(INPUT_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} questions.")

    rows = []

    for q in questions:
        rows.append(
            {
                "question_id": q.get("question_id"),
                "paper_source": q.get("paper_source"),
                "subject": q.get("subject"),
                "chapter": q.get("chapter"),
                "topic_tag": q.get("topic_tag"),
                "skill": q.get("skill"),
                "difficulty_level": q.get("difficulty_level"),
                "question_type": q.get("question_type"),
                "ideal_time_sec": q.get("ideal_time_sec"),
                "has_diagram": q.get("has_diagram"),
                "marks": q.get("marks"),
                "negative_marks": q.get("negative_marks"),
                "verification_status": q.get("verification_status"),
            }
        )

    df = pd.DataFrame(rows)

    print("\nFeature dataset created.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nMissing values:")
    print(df.isnull().sum())

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\nSaved to:")
    print(OUTPUT_FILE)

    print("\nDifficulty distribution:")
    print(df["difficulty_level"].value_counts(dropna=False))

    print("\nSubject distribution:")
    print(df["subject"].value_counts(dropna=False))

    print("\nChapter count:")
    print(df["chapter"].nunique())

    print("\nTopic count:")
    print(df["topic_tag"].nunique())


if __name__ == "__main__":
    main()