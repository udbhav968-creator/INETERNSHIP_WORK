"""
Builds model/topic_importance.json directly from a tagged question bank's
`years_asked` field — no separately maintained frequency dict needed.

This deliberately depends on real tagged data (data/tagged_questions/*.json)
existing with accurate years_asked values. Do not run this against the
TEMPLATE files — it will produce meaningless scores from 1 example question.

Usage:
    python model/build_topic_importance.py data/tagged_questions/biology_questions.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.paths import MODEL_DIR  # noqa: E402
from importance_calculator import compute_importance  # noqa: E402


def build_frequency_from_questions(questions: list[dict]) -> dict:
    """{"BIO-07-01": {2019: 2, 2021: 1, ...}, ...} derived from years_asked."""
    freq: dict[str, dict[int, int]] = {}
    for q in questions:
        topic_id = q["topic_id"]
        freq.setdefault(topic_id, {})
        for year in q.get("years_asked", []):
            freq[topic_id][year] = freq[topic_id].get(year, 0) + 1
    return freq


def main(questions_path: str, current_year: int = 2026):
    with open(questions_path) as f:
        questions = json.load(f)

    if len(questions) < 20:
        print(
            f"WARNING: only {len(questions)} questions found. Importance scores "
            f"from this few questions won't be meaningful — wait for a larger "
            f"tagged batch before trusting this output.",
            file=sys.stderr,
        )

    frequency = build_frequency_from_questions(questions)
    importance = compute_importance(frequency, current_year=current_year)

    out_path = MODEL_DIR / "topic_importance.json"
    with open(out_path, "w") as f:
        json.dump(importance, f, indent=2)

    print(f"Wrote {out_path} — {len(importance)} topics scored.")
    print(json.dumps(importance, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python build_topic_importance.py <path-to-tagged-questions.json>")
        sys.exit(1)
    main(sys.argv[1])
