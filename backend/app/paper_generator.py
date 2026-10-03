"""
Adaptive paper generator — docs/spec.md Section 9.

Topic proportions (importance) are fixed and never touched per-student.
Only which difficulty pool is sampled from, per topic, changes per student.
"""
import json
import random
from sqlalchemy.orm import Session
from .db_models import Question, StudentTopicState

from .paths import MODEL_DIR

with open(MODEL_DIR / "topic_importance.json") as f:
    _TOPIC_IMPORTANCE = json.load(f)


def generate_paper(db: Session, student_id: str, total_length: int = 45) -> list[Question]:
    paper: list[Question] = []

    for topic_id, importance in _TOPIC_IMPORTANCE.items():
        num_questions = round(importance / 100 * total_length)
        if num_questions == 0:
            continue

        state = (
            db.query(StudentTopicState)
            .filter_by(student_id=student_id, topic_id=topic_id)
            .first()
        )
        level = state.level if state else "basic"

        pool = (
            db.query(Question)
            .filter_by(topic_id=topic_id, difficulty=level)
            .all()
        )
        selected = random.sample(pool, min(num_questions, len(pool)))
        paper.extend(selected)

        # Pace-driven supplement: topic is correct but slow -> add timed
        # practice, doesn't change the topic's core question count above.
        if state and state.pace_flag == "slow":
            remaining_pool = [q for q in pool if q not in selected]
            extra = random.sample(remaining_pool, min(2, len(remaining_pool)))
            paper.extend(extra)

        # TODO: occasionally inject 1 probe question from adjacent difficulty
        # to test for promotion out of an "advanced" plateau or catch silent
        # regression — leaving as a stub until we have real usage data to
        # tune how often this should trigger.

    return paper
