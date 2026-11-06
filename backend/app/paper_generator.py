import json
import random
from sqlalchemy.orm import Session, make_transient
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

_QUESTION_POOL_CACHE: dict[tuple[str, str], list[Question]] = {}
_CACHED_TOTAL_QUESTIONS: int = 0


def _get_question_pools(db: Session) -> dict[tuple[str, str], list[Question]]:
    global _QUESTION_POOL_CACHE, _CACHED_TOTAL_QUESTIONS
    total_in_db = db.query(Question).count()
    if not _QUESTION_POOL_CACHE or total_in_db != _CACHED_TOTAL_QUESTIONS:
        pools: dict[tuple[str, str], list[Question]] = {}
        all_questions = db.query(Question).all()
        for q in all_questions:
            db.expunge(q)
            make_transient(q)
            pools.setdefault((q.topic_id, q.difficulty), []).append(q)
        _QUESTION_POOL_CACHE = pools
        _CACHED_TOTAL_QUESTIONS = total_in_db
    return _QUESTION_POOL_CACHE


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
        jitter = random.uniform(-0.015, 0.015)
        scored.append((ucb_score + jitter, q))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [q for _, q in scored[:k]]


def generate_paper(
    db: Session,
    student_id: str,
    total_length: int = 45,
    topic_id: str | None = None,
) -> list[Question]:
    paper: list[Question] = []
    pools = _get_question_pools(db)

    student = db.query(Student).filter_by(student_id=student_id).first()
    global_theta = float(getattr(student, "global_theta", 0.0) or 0.0) if student else 0.0

    all_states = db.query(StudentTopicState).filter_by(student_id=student_id).all()
    state_by_topic = {s.topic_id: s for s in all_states}
    mastery_map = {s.topic_id: float(getattr(s, "mastery_prob", 0.35) or 0.35) for s in all_states}

    if topic_id:
        target_count = min(15, total_length)
        state = state_by_topic.get(topic_id)
        level = state.level if state else "basic"
        topic_theta = float(getattr(state, "irt_theta", global_theta) or global_theta) if state else global_theta
        mastery_prob = float(getattr(state, "mastery_prob", 0.35) or 0.35) if state else 0.35
        pace_flag = state.pace_flag if state else None
        prereq_ready = compute_prerequisite_readiness(topic_id, mastery_map)
        imp_val = float(_TOPIC_IMPORTANCE.get(topic_id, 1.5))

        pool = pools.get((topic_id, level), [])
        selected = _select_with_bandit(
            pool,
            min(target_count, len(pool)),
            student_id,
            topic_theta,
            mastery_prob,
            imp_val,
            prereq_ready,
            pace_flag,
        )
        if len(selected) < target_count:
            for fallback_diff in ADJACENT_FALLBACK_ORDER.get(level, ["medium", "basic"]):
                needed = target_count - len(selected)
                if needed <= 0:
                    break
                adj_pool = pools.get((topic_id, fallback_diff), [])
                adj_candidates = [q for q in adj_pool if q not in selected]
                selected.extend(
                    _select_with_bandit(
                        adj_candidates,
                        min(needed, len(adj_candidates)),
                        student_id,
                        topic_theta,
                        mastery_prob,
                        imp_val,
                        prereq_ready,
                        pace_flag,
                    )
                )
        return selected

    fractional_topics: list[tuple[float, str]] = []
    for topic_id, importance in _TOPIC_IMPORTANCE.items():
        num_questions = round(importance / 100 * total_length)
        if num_questions == 0:
            fractional_topics.append((float(importance), topic_id))
            continue

        state = state_by_topic.get(topic_id)
        level = state.level if state else "basic"
        topic_theta = float(getattr(state, "irt_theta", global_theta) or global_theta) if state else global_theta
        mastery_prob = float(getattr(state, "mastery_prob", 0.35) or 0.35) if state else 0.35
        pace_flag = state.pace_flag if state else None
        prereq_ready = compute_prerequisite_readiness(topic_id, mastery_map)

        pool = pools.get((topic_id, level), [])
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

        if len(selected) < num_questions:
            for fallback_diff in ADJACENT_FALLBACK_ORDER.get(level, ["medium", "basic"]):
                needed = num_questions - len(selected)
                if needed <= 0:
                    break
                adj_pool = pools.get((topic_id, fallback_diff), [])
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

        if state and state.level == "advanced" and (state.attempts_counted or 0) >= 3:
            medium_pool = pools.get((topic_id, "medium"), [])
            probe_candidates = [q for q in medium_pool if q not in paper]
            if probe_candidates and len(paper) < total_length + 5:
                paper.append(probe_candidates[0])

    if len(paper) < total_length and fractional_topics:
        remaining_slots = total_length - len(paper)
        weighted_candidates: list[tuple[float, str]] = []
        for imp, t_id in fractional_topics:
            readiness = compute_prerequisite_readiness(t_id, mastery_map)
            score = imp * readiness * random.uniform(0.75, 1.25)
            weighted_candidates.append((score, t_id))
        weighted_candidates.sort(key=lambda pair: pair[0], reverse=True)

        for _, t_id in weighted_candidates[:remaining_slots]:
            state = state_by_topic.get(t_id)
            level = state.level if state else "basic"
            topic_theta = float(getattr(state, "irt_theta", global_theta) or global_theta) if state else global_theta
            mastery_prob = float(getattr(state, "mastery_prob", 0.35) or 0.35) if state else 0.35
            pace_flag = state.pace_flag if state else None
            prereq_ready = compute_prerequisite_readiness(t_id, mastery_map)
            imp_val = float(_TOPIC_IMPORTANCE.get(t_id, 0.5))

            pool = pools.get((t_id, level), [])
            if not pool:
                for fb_diff in ADJACENT_FALLBACK_ORDER.get(level, ["medium", "basic"]):
                    pool = pools.get((t_id, fb_diff), [])
                    if pool:
                        break
            picked = _select_with_bandit(
                pool, 1, student_id, topic_theta, mastery_prob, imp_val, prereq_ready, pace_flag
            )
            paper.extend(picked)

    return paper
