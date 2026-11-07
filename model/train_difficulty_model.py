import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from model.rf_compact_runtime import export_rf_compact_forest, reload_compact_rf_runtime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "training" / "question_features.csv"
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_FILE = MODEL_DIR / "question_difficulty_model.joblib"
METRICS_FILE = MODEL_DIR / "question_difficulty_metrics.json"

CATEGORICAL_FEATURES = ["subject", "chapter", "topic_tag", "question_type"]
NUMERIC_FEATURES = [
    "ideal_time_sec",
    "has_diagram",
    "word_count",
    "char_length",
    "formula_density",
    "numerical_complexity",
    "negation_trap",
    "distractor_similarity",
    "marks",
    "negative_marks",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def main():
    print("Loading training dataset...")
    df = pd.read_csv(DATA_FILE)
    print(f"Rows: {len(df)}")

    target = "difficulty_level"
    X = df[FEATURES]
    y = df[target]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=300,
        max_features=0.35,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\nAccuracy:", round(accuracy, 4))
    print("\nClassification report:")
    print(classification_report(y_test, predictions))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE)
    export_rf_compact_forest(pipeline, numeric_features=NUMERIC_FEATURES)
    reload_compact_rf_runtime()

    metrics = {
        "model": "RandomForestClassifier",
        "target": target,
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "accuracy": float(accuracy),
        "classes": sorted(y.unique().tolist()),
        "features": FEATURES,
    }
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
