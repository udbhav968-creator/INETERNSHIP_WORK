from sqlalchemy.orm import Session
from .db_models import TestAttempt, Question, Topic


def build_analytics(db: Session, student_id: str) -> dict:
    rows = (
        db.query(TestAttempt, Question.topic_id, Topic.subject)
        .join(Question, TestAttempt.question_id == Question.question_id)
        .join(Topic, Question.topic_id == Topic.topic_id)
        .filter(TestAttempt.student_id == student_id)
        .order_by(TestAttempt.timestamp.desc())
        .all()
    )

    total = len(rows)
    correct = sum(1 for attempt, _, _ in rows if attempt.is_correct)
    accuracy = round(correct / total, 4) if total else 0.0
    topics_practiced = len({topic_id for _, topic_id, _ in rows})

    subject_stats: dict[str, dict] = {}
    for attempt, _, subject in rows:
        s = subject_stats.setdefault(subject, {"attempts": 0, "correct": 0})
        s["attempts"] += 1
        if attempt.is_correct:
            s["correct"] += 1

    subject_breakdown = [
        {
            "subject": subject,
            "attempts": stats["attempts"],
            "correct": stats["correct"],
            "accuracy": round(stats["correct"] / stats["attempts"], 4) if stats["attempts"] else 0.0,
        }
        for subject, stats in subject_stats.items()
    ]

    recent_attempts = [
        {
            "question_id": attempt.question_id,
            "topic_id": topic_id,
            "is_correct": attempt.is_correct,
            "time_taken_seconds": attempt.time_taken_seconds,
            "timestamp": attempt.timestamp.isoformat(),
        }
        for attempt, topic_id, _ in rows[:10]
    ]

    return {
        "total_attempts": total,
        "correct_attempts": correct,
        "accuracy": accuracy,
        "topics_practiced": topics_practiced,
        "subject_breakdown": subject_breakdown,
        "recent_attempts": recent_attempts,
    }
