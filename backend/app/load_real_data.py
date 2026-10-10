"""
Loads a VALIDATED real question bank into the database — topics and
questions. Only run this after model/validate_question_bank.py comes back
clean (or with issues you've consciously decided to accept).

Topic IDs are derived directly from the question bank's own `chapter`
field — no separate taxonomy file needed, since the intern's data uses a
locked canonical chapter naming scheme (see the file's "taxonomy_version"
metadata field).

Usage:
    cd backend
    python -m app.load_real_data path/to/real_question_bank.json
    python -m app.load_real_data path/to/real_question_bank.json --wipe   # clears existing topics/questions first
"""
import json
import re
import sys
from .database import SessionLocal, init_db
from .db_models import Topic, Question

# Maps the intern's difficulty labels to our required basic/medium/advanced.
# If a future batch introduces a new label, this will raise clearly rather
# than silently dropping questions.
DIFFICULTY_MAP = {
    "basic": "basic",
    "base": "basic",
    "medium": "medium",
    "intermediate": "medium",
    "advanced": "advanced",
}


def extract_year(paper_source: str) -> int:
    match = re.search(r"(\d{4})", paper_source or "")
    if not match:
        raise ValueError(f"Could not extract year from paper_source: {paper_source!r}")
    return int(match.group(1))


def load(question_bank_path: str, wipe_existing: bool = False):
    with open(question_bank_path) as f:
        questions = json.load(f)

    init_db()
    db = SessionLocal()

    if wipe_existing:
        db.query(Question).delete()
        db.query(Topic).delete()
        db.commit()

    seen_topics = set()
    loaded, skipped = 0, 0
    skip_reasons = {}

    for q in questions:
        chapter = q.get("chapter")
        subject = q.get("subject")
        difficulty_raw = str(q.get("difficulty_level", "")).lower()
        difficulty = DIFFICULTY_MAP.get(difficulty_raw)

        if not chapter or chapter.startswith("General ") or not difficulty:
            skipped += 1
            reason = "missing/generic chapter" if not chapter or chapter.startswith("General ") else f"unmapped difficulty '{difficulty_raw}'"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            continue

        if chapter not in seen_topics:
            if not db.query(Topic).filter_by(topic_id=chapter).first():
                db.add(Topic(topic_id=chapter, subject=subject, name=chapter))
            seen_topics.add(chapter)

        if db.query(Question).filter_by(question_id=q["question_id"]).first():
            skipped += 1
            skip_reasons["duplicate question_id"] = skip_reasons.get("duplicate question_id", 0) + 1
            continue

        options = [q.get("option_a"), q.get("option_b"), q.get("option_c"), q.get("option_d")]
        correct = str(q.get("correct_option", "")).lower()

        db.add(Question(
            question_id=q["question_id"],
            topic_id=chapter,
            difficulty=difficulty,
            expected_time_seconds=q.get("ideal_time_sec", 90),
            text=q.get("question_text", ""),
            options=options,
            correct_answer=correct,
            wrong_option_tags=None,  # not yet supplied by intern — mistake-fingerprinting on hold until it is
            years_asked=[extract_year(q.get("paper_source", ""))],
        ))
        loaded += 1

    db.commit()
    db.close()

    print(f"Loaded {loaded} questions across {len(seen_topics)} topics.")
    if skipped:
        print(f"Skipped {skipped} questions:")
        for reason, count in skip_reasons.items():
            print(f"  - {count} due to: {reason}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m app.load_real_data <path-to-question-bank.json> [--wipe]")
        sys.exit(1)
    load(sys.argv[1], wipe_existing="--wipe" in sys.argv)
