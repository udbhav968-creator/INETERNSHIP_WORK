import json
import re
import sys
from .database import SessionLocal, init_db
from .db_models import Topic, Question

DIFFICULTY_MAP = {
    "basic": "basic",
    "base": "basic",
    "easy": "basic",
    "medium": "medium",
    "intermediate": "medium",
    "advanced": "advanced",
    "hard": "advanced",
}

GENERIC_CHAPTERS = {"General Biology", "General Chemistry", "General Physics"}


def normalize_topic_name(raw_name: str) -> str:
    return re.sub(r"\s+", " ", (raw_name or "").strip())


def extract_year(paper_source: str) -> int:
    match = re.search(r"(\d{4})", paper_source or "")
    if not match:
        raise ValueError(f"Could not extract year from paper_source: {paper_source!r}")
    return int(match.group(1))


def load(question_bank_path: str, wipe_existing: bool = False):
    with open(question_bank_path, encoding="utf-8") as f:
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
        chapter = normalize_topic_name(q.get("chapter") or "")
        subject = normalize_topic_name(q.get("subject") or "")
        difficulty_raw = str(q.get("difficulty_level", "")).strip().lower()
        difficulty = DIFFICULTY_MAP.get(difficulty_raw)

        if not chapter or chapter in GENERIC_CHAPTERS or not difficulty:
            skipped += 1
            reason = (
                "missing/generic chapter"
                if not chapter or chapter in GENERIC_CHAPTERS
                else f"unmapped difficulty '{difficulty_raw}'"
            )
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            continue

        if chapter not in seen_topics:
            if not db.query(Topic).filter_by(topic_id=chapter).first():
                db.add(Topic(topic_id=chapter, subject=subject, name=chapter))
            seen_topics.add(chapter)

        q_id = str(q["question_id"]).strip()
        if db.query(Question).filter_by(question_id=q_id).first():
            skipped += 1
            skip_reasons["duplicate question_id"] = skip_reasons.get("duplicate question_id", 0) + 1
            continue

        options = [q.get("option_a"), q.get("option_b"), q.get("option_c"), q.get("option_d")]
        correct = str(q.get("correct_option", "")).strip().lower()

        db.add(
            Question(
                question_id=q_id,
                topic_id=chapter,
                difficulty=difficulty,
                expected_time_seconds=int(q.get("ideal_time_sec") or 90),
                text=q.get("question_text", ""),
                options=options,
                correct_answer=correct,
                wrong_option_tags=q.get("wrong_option_tags"),
                years_asked=[extract_year(q.get("paper_source", "NEET_2024"))],
            )
        )
        loaded += 1

    db.commit()
    db.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    load(sys.argv[1], wipe_existing="--wipe" in sys.argv)
