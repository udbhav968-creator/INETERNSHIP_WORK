"""
Builds model/topic_importance.json directly from a tagged question bank.

Handles BOTH schema shapes:
  - our original template schema (topic_id, years_asked: [list of years])
  - the intern's real schema (chapter, paper_source: "NEET 2013" - one year per entry)

This was only ever tested against the template schema before — fixed here
after discovering the real data uses different field names entirely.

Usage:
    python model/build_topic_importance.py data/tagged_questions/<real_file>.json
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.paths import MODEL_DIR  # noqa: E402
from importance_calculator import compute_importance  # noqa: E402

GENERIC_CHAPTER_PREFIXES = ("General Biology", "General Chemistry", "General Physics")


def _extract_year(paper_source: str) -> int | None:
    match = re.search(r"(\d{4})", paper_source or "")
    return int(match.group(1)) if match else None


def build_frequency_from_questions(questions: list[dict]) -> dict:
    """{"Chapter Name": {2019: 2, 2021: 1, ...}, ...} derived from whichever
    schema the question bank actually uses."""
    freq: dict[str, dict[int, int]] = {}
    skipped_generic = 0

    for q in questions:
        topic_id = q.get("topic_id") or q.get("chapter")
        if not topic_id or topic_id in GENERIC_CHAPTER_PREFIXES:
            skipped_generic += 1
            continue

        freq.setdefault(topic_id, {})

        years = q.get("years_asked")
        if not years:
            year = _extract_year(q.get("paper_source", ""))
            years = [year] if year else []

        for year in years:
            freq[topic_id][year] = freq[topic_id].get(year, 0) + 1

    if skipped_generic:
        print(
            f"WARNING: skipped {skipped_generic} questions with missing/generic chapter tags "
            f"(e.g. 'General Biology') — these won't contribute to any topic's importance score "
            f"until they're re-tagged to a real chapter.",
            file=sys.stderr,
        )

    return freq


def main(questions_path: str, current_year: int = 2026):
    with open(questions_path) as f:
        questions = json.load(f)

    if len(questions) < 20:
        print(
            f"WARNING: only {len(questions)} questions found. Importance scores "
            f"from this few questions won't be meaningful.",
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
