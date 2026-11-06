import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.paths import MODEL_DIR
from importance_calculator import compute_importance

GENERIC_CHAPTER_PREFIXES = ("General Biology", "General Chemistry", "General Physics")


def _extract_year(paper_source: str) -> int | None:
    match = re.search(r"(\d{4})", paper_source or "")
    return int(match.group(1)) if match else None


def build_frequency_from_questions(questions: list[dict]) -> dict:
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

    return freq


def build_topic_importance(
    questions: list[dict], current_year: int = 2026, decay: float = 0.9
) -> dict:
    freq = build_frequency_from_questions(questions)
    return compute_importance(freq, current_year=current_year, decay=decay)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python model/build_topic_importance.py <path_to_questions.json>")
        sys.exit(1)

    src = Path(sys.argv[1])
    with open(src, encoding="utf-8") as f:
        questions = json.load(f)

    importance = build_topic_importance(questions)
    dest = MODEL_DIR / "topic_importance.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(importance, f, indent=2)

    print(f"Wrote {len(importance)} topic importance scores to {dest}")


if __name__ == "__main__":
    main()
