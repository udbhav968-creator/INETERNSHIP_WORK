from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
CLEAN_QBANK_PATH = PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json"
ATTEMPTS_CSV_PATH = PROJECT_ROOT / "data" / "training" / "student_attempts.csv"

TIER_TO_INT = {"basic": 0, "medium": 1, "advanced": 2}
SUBJECT_TO_INT = {"Physics": 0, "Chemistry": 1, "Biology": 2}


def _count_logged_attempts() -> int:
    if not ATTEMPTS_CSV_PATH.exists():
        return 0
    with open(ATTEMPTS_CSV_PATH, encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def run_label_validity_audit(questions: list[dict[str, Any]] | None = None, db_attempt_count: int = 0, save_report: bool = True) -> dict[str, Any]:
    if questions is None:
        questions = json.loads(CLEAN_QBANK_PATH.read_text(encoding="utf-8"))

    y = np.array([TIER_TO_INT[str(q["difficulty_level"]).lower()] for q in questions])
    t_only = np.array([[float(q["ideal_time_sec"])] for q in questions])
    t_subj = np.array([[float(q["ideal_time_sec"]), SUBJECT_TO_INT[q["subject"]]] for q in questions])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def cv_acc(X: np.ndarray) -> float:
        clf = DecisionTreeClassifier(max_depth=4, random_state=42)
        return float(np.mean(cross_val_score(clf, X, y, cv=skf, scoring="accuracy")))

    majority = Counter(y.tolist()).most_common(1)[0][1] / len(y)

    paired = [
        (TIER_TO_INT[str(q["difficulty_level"]).lower()], TIER_TO_INT[str(q["original_difficulty_level"]).lower()])
        for q in questions
        if str(q.get("original_difficulty_level", "")).lower() in TIER_TO_INT
    ]
    agreement: dict[str, Any] = {"available": bool(paired)}
    if paired:
        cur = [p[0] for p in paired]
        orig = [p[1] for p in paired]
        same = sum(1 for a, b in paired if a == b)
        agreement.update(
            {
                "n": len(paired),
                "exact_agreement_percent": round(100.0 * same / len(paired), 2),
                "cohens_kappa": round(float(cohen_kappa_score(cur, orig)), 4),
                "changed_from_original": len(paired) - same,
            }
        )

    attempts = max(_count_logged_attempts(), int(db_attempt_count))
    rep = {
        "what_the_rf_accuracy_measures": "reproduction_of_heuristic_tier_labels_not_empirical_difficulty",
        "baselines_5fold_cv_accuracy": {
            "majority_class": round(float(majority), 4),
            "depth4_tree_on_ideal_time_sec_only": round(cv_acc(t_only), 4),
            "depth4_tree_on_ideal_time_sec_and_subject": round(cv_acc(t_subj), 4),
        },
        "tier_vs_original_source_tags": agreement,
        "student_responses_available": attempts,
        "empirical_difficulty_validation_possible": attempts > 0,
        "interpretation": (
            "Tiers are within-chapter text-complexity quantiles and ideal_time_sec is a function of the tier, "
            "so high accuracy here shows the labelling rule is learnable, not that difficulty is predicted. "
            "Empirical validation needs logged student responses."
        ),
    }
    if save_report:
        (MODEL_DIR / "label_validity_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return rep


if __name__ == "__main__":
    print(json.dumps(run_label_validity_audit(), indent=2))
