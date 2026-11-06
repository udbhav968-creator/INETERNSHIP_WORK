import json
from .auth import hash_password
from .database import SessionLocal, init_db
from .db_models import Student, Topic, Question
from .paths import MODEL_DIR

FAKE_TOPIC_ID = "TEST-01"


def seed():
    init_db()
    db = SessionLocal()

    if db.query(Student).filter_by(student_id="TEST_STUDENT").first():
        db.close()
        return

    db.add(
        Student(
            student_id="TEST_STUDENT",
            name="Test Student",
            email="test_student@klassed.in",
            password_hash=hash_password("testpassword123"),
            onboarded=True,
        )
    )
    db.add(Topic(topic_id=FAKE_TOPIC_ID, subject="TestSubject", name="Fake Test Topic"))

    for difficulty in ["basic", "medium", "advanced"]:
        for i in range(3):
            db.add(
                Question(
                    question_id=f"TESTQ-{difficulty}-{i}",
                    topic_id=FAKE_TOPIC_ID,
                    difficulty=difficulty,
                    expected_time_seconds={"basic": 60, "medium": 90, "advanced": 120}[difficulty],
                    text=f"[FAKE] Sample {difficulty} question #{i}",
                    options=["A", "B", "C", "D"],
                    correct_answer="a",
                    wrong_option_tags={},
                    years_asked=[2026],
                )
            )

    db.commit()
    db.close()

    fake_importance_path = MODEL_DIR / "topic_importance.json"
    if not fake_importance_path.exists():
        with open(fake_importance_path, "w", encoding="utf-8") as f:
            json.dump({FAKE_TOPIC_ID: 100.0}, f, indent=2)


if __name__ == "__main__":
    seed()
