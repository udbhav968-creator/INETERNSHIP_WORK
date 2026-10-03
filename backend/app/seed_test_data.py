"""
Seeds a handful of FAKE test rows so you can exercise the full API loop
(submit_attempt -> state update -> next_paper) before real tagged data
exists. This data is deliberately obviously fake — do not mistake it for
real content, and don't commit klassen.db with this seed data as if it
were real.

Run with:
    cd backend
    python -m app.seed_test_data
"""
import json
from .database import SessionLocal, init_db
from .db_models import Student, Topic, Question
from .paths import MODEL_DIR

FAKE_TOPIC_ID = "TEST-01"


def seed():
    init_db()
    db = SessionLocal()

    if db.query(Student).filter_by(student_id="TEST_STUDENT").first():
        print("Seed data already present — skipping.")
        db.close()
        return

    db.add(Student(student_id="TEST_STUDENT", name="Test Student"))
    db.add(Topic(topic_id=FAKE_TOPIC_ID, subject="TestSubject", name="Fake Test Topic"))

    for difficulty in ["basic", "medium", "advanced"]:
        for i in range(3):
            db.add(Question(
                question_id=f"TESTQ-{difficulty}-{i}",
                topic_id=FAKE_TOPIC_ID,
                difficulty=difficulty,
                expected_time_seconds={"basic": 60, "medium": 90, "advanced": 120}[difficulty],
                text=f"[FAKE] Sample {difficulty} question #{i}",
                options=["A", "B", "C", "D"],
                correct_answer="A",
                wrong_option_tags={},
                years_asked=[2026],
            ))

    db.commit()
    db.close()

    # also write a minimal fake topic_importance.json so paper_generator.py
    # has something to load — delete this once the real one exists
    fake_importance_path = MODEL_DIR / "topic_importance.json"
    if not fake_importance_path.exists():
        with open(fake_importance_path, "w") as f:
            json.dump({FAKE_TOPIC_ID: 100.0}, f, indent=2)
        print(f"Wrote FAKE {fake_importance_path} (100% weight, one topic only, for testing only)")

    print("Seeded fake student, topic, and 9 questions (3 per difficulty).")


if __name__ == "__main__":
    seed()
