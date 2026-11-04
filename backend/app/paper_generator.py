"""
Adaptive paper generator — docs/spec.md Section 9 + Part I Fix #2 + Part II LinUCB & DAG Routing.

Topic proportions (importance) are fixed and never touched per-student.
Difficulty pool selection and within-pool item selection use:
1. Adjacent-pool fallback cascade when a topic's primary difficulty pool has fewer items than requested (Part I Section 7 Fix #2).
2. LinUCB Contextual Multi-Armed Bandit & 3PL Fisher Information within the Zone of Proximal Development (Part II Section 4).
3. Periodic retention probes for advanced topics to guard against silent forgetting.
"""
import json
import random
from sqlalchemy.orm import Session
from .db_models import Question, StudentTopicState, Student
from .paths import MODEL_DIR
from model.linucb_bandit import build_context_vector, score_candidate_ucb
from model.knowledge_graph_dag import compute_prerequisite_readiness

with open(MODEL_DIR / "topic_importance.json", encoding="utf-8") as f:
    _TOPIC_IMPORTANCE = json.load(f)

ADJACENT_FALLBACK_ORDER = {
    "basic": ["medium", "advanced"],
    "medium": ["basic", "advanced"],
    "advanced": ["medium", "basic"],
}


def _select_with_bandit(
    pool: list[Question],
    k: int,
    student_id: str,
    student_theta: float,
    mastery_prob: float,
    topic_importance: float,
    prereq_readiness: float,
    pace_flag: str | None,
) -> list[Question]:
    if not pool or k <= 0:
        return []
    if len(pool) <= k:
        return list(pool)

    scored: list[tuple[float, Question]] = []
    for q in pool:
        x = build_context_vector(
            question_id=q.question_id,
            difficulty=q.difficulty,
            expected_time_seconds=q.expected_time_seconds or 90,
            student_theta=student_theta,
            topic_mastery_prob=mastery_prob,
            topic_importance=topic_importance,
            prereq_readiness=prereq_readiness,
            pace_flag=pace_flag,
        )
        ucb_score, _, _ = score_candidate_ucb(student_id, x)
        # Small random tie-breaker so repeated papers explore diverse high-UCB questions
        jitter = random.uniform(-0.015, 0.015)
        scored.append((ucb_score + jitter, q))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [q for _, q in scored[:k]]


def generate_paper(db: Session, student_id: str, total_length: int = 45) -> list[Question]:
    paper: list[Question] = []
    student = db.query(Student).filter_by(student_id=student_id).first()
    global_theta = float(getattr(student, "global_theta", 0.0) or 0.0) if student else 0.0

    all_states = db.query(StudentTopicState).filter_by(student_id=student_id).all()
    state_by_topic = {s.topic_id: s for s in all_states}
    mastery_map = {s.topic_id: float(getattr(s, "mastery_prob", 0.35) or 0.35) for s in all_states}

    for topic_id, importance in _TOPIC_IMPORTANCE.items():
        num_questions = round(importance / 100 * total_length)
        if num_questions == 0:
            continue

        state = state_by_topic.get(topic_id)
        level = state.level if state else "basic"
        topic_theta = float(getattr(state, "irt_theta", global_theta) or global_theta) if state else global_theta
        mastery_prob = float(getattr(state, "mastery_prob", 0.35) or 0.35) if state else 0.35
        pace_flag = state.pace_flag if state else None
        prereq_ready = compute_prerequisite_readiness(topic_id, mastery_map)

        pool = (
            db.query(Question)
            .filter_by(topic_id=topic_id, difficulty=level)
            .all()
        )
        selected = _select_with_bandit(
            pool,
            min(num_questions, len(pool)),
            student_id,
            topic_theta,
            mastery_prob,
            float(importance),
            prereq_ready,
            pace_flag,
        )

        # Part I Section 7 Fix #2: Adjacent difficulty pool fallback cascade when primary pool is sparse
        if len(selected) < num_questions:
            for fallback_diff in ADJACENT_FALLBACK_ORDER.get(level, ["medium", "basic"]):
                needed = num_questions - len(selected)
                if needed <= 0:
                    break
                adj_pool = (
                    db.query(Question)
                    .filter_by(topic_id=topic_id, difficulty=fallback_diff)
                    .all()
                )
                adj_candidates = [q for q in adj_pool if q not in selected]
                adj_selected = _select_with_bandit(
                    adj_candidates,
                    min(needed, len(adj_candidates)),
                    student_id,
                    topic_theta,
                    mastery_prob,
                    float(importance),
                    prereq_ready,
                    pace_flag,
                )
                selected.extend(adj_selected)

        paper.extend(selected)

        # Pace-driven supplement: topic is correct but slow -> add timed practice
        if state and state.pace_flag == "slow":
            remaining_pool = [q for q in pool if q not in selected]
            extra = _select_with_bandit(
                remaining_pool,
                min(2, len(remaining_pool)),
                student_id,
                topic_theta,
                mastery_prob,
                float(importance),
                prereq_ready,
                pace_flag,
            )
            paper.extend(extra)

        # Retention probe for advanced topics: confirm long-term retention
        if state and state.level == "advanced" and (state.attempts_counted or 0) >= 3:
            medium_pool = (
                db.query(Question)
                .filter_by(topic_id=topic_id, difficulty="medium")
                .all()
            )
            probe_candidates = [q for q in medium_pool if q not in paper]
            if probe_candidates and len(paper) < total_length + 5:
                paper.append(probe_candidates[0])

    return paper
