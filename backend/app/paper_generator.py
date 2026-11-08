import json
import random
from sqlalchemy.orm import Session, make_transient
from .db_models import Question, StudentTopicState, Student, TestAttempt, Topic
from .paths import MODEL_DIR
from model.linucb_bandit import build_context_vector, score_candidate_ucb
from model.knowledge_graph_dag import compute_prerequisite_readiness
from model.question_similarity_audit import get_near_duplicate_siblings_map
from model.state_machine.bkt_model import get_chapter_bkt_prior

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


def _select_topic_questions(
    pools: dict[tuple[str, str], list[Question]],
    topic_id: str,
    level: str,
    target_count: int,
    attempted_all: set[str],
    mastered_correct: set[str],
    excluded_ids: set[str],
    student_id: str,
    topic_theta: float,
    mastery_prob: float,
    topic_importance: float,
    prereq_ready: float,
    pace_flag: str | None,
) -> list[Question]:
    if target_count <= 0:
        return []

    tier_order = [level] + list(ADJACENT_FALLBACK_ORDER.get(level, ["medium", "basic"]))
    selected: list[Question] = []

    for pass_mode in ("unseen", "missed", "mastered"):
        if len(selected) >= target_count:
            break
        for diff in tier_order:
            needed = target_count - len(selected)
            if needed <= 0:
                break
            raw_pool = pools.get((topic_id, diff), [])
            if pass_mode == "unseen":
                candidates = [
                    q
                    for q in raw_pool
                    if q.question_id not in excluded_ids and q.question_id not in attempted_all
                ]
            elif pass_mode == "missed":
                candidates = [
                    q
                    for q in raw_pool
                    if q.question_id not in excluded_ids
                    and q.question_id in attempted_all
                    and q.question_id not in mastered_correct
                ]
            else:
                candidates = [
                    q
                    for q in raw_pool
                    if q.question_id not in excluded_ids and q.question_id in mastered_correct
                ]

            if not candidates:
                continue

            siblings_map = get_near_duplicate_siblings_map()
            ranked_candidates = _select_with_bandit(
                candidates,
                len(candidates),
                student_id,
                topic_theta,
                mastery_prob,
                topic_importance,
                prereq_ready,
                pace_flag,
            )
            for q in ranked_candidates:
                if len(selected) >= target_count:
                    break
                if q.question_id in excluded_ids:
                    continue
                excluded_ids.add(q.question_id)
                for sib_id in siblings_map.get(q.question_id, []):
                    excluded_ids.add(sib_id)
                selected.append(q)

    return selected


def generate_paper(
    db: Session,
    student_id: str,
    total_length: int = 45,
    topic_id: str | None = None,
    mode: str = "mock_45",
) -> list[Question]:
    paper: list[Question] = []
    pools = _get_question_pools(db)

    student = db.query(Student).filter_by(student_id=student_id).first()
    global_theta = float(getattr(student, "global_theta", 0.0) or 0.0) if student else 0.0

    all_states = db.query(StudentTopicState).filter_by(student_id=student_id).all()
    state_by_topic = {s.topic_id: s for s in all_states}
    mastery_map = {
        tid: (float(state_by_topic[tid].mastery_prob) if tid in state_by_topic and getattr(state_by_topic[tid], "mastery_prob", None) is not None else get_chapter_bkt_prior(tid))
        for tid in _TOPIC_IMPORTANCE
    }

    attempt_rows = (
        db.query(TestAttempt.question_id, TestAttempt.is_correct)
        .filter(TestAttempt.student_id == student_id)
        .all()
    )
    attempted_all: set[str] = {qid for qid, _ in attempt_rows}
    mastered_correct: set[str] = {qid for qid, corr in attempt_rows if corr}
    excluded_ids: set[str] = set()

    if topic_id:
        target_count = min(15, total_length)
        state = state_by_topic.get(topic_id)
        mastery_prob = float(state.mastery_prob) if state and getattr(state, "mastery_prob", None) is not None else get_chapter_bkt_prior(topic_id)
        level = state.level if state and getattr(state, "level", None) else ("advanced" if mastery_prob >= 0.70 else ("medium" if mastery_prob >= 0.45 else "basic"))
        topic_theta = float(getattr(state, "irt_theta", global_theta) or global_theta) if state else global_theta
        pace_flag = state.pace_flag if state else None
        prereq_ready = compute_prerequisite_readiness(topic_id, mastery_map)
        imp_val = float(_TOPIC_IMPORTANCE.get(topic_id, 1.5))

        return _select_topic_questions(
            pools=pools,
            topic_id=topic_id,
            level=level,
            target_count=target_count,
            attempted_all=attempted_all,
            mastered_correct=mastered_correct,
            excluded_ids=excluded_ids,
            student_id=student_id,
            topic_theta=topic_theta,
            mastery_prob=mastery_prob,
            topic_importance=imp_val,
            prereq_ready=prereq_ready,
            pace_flag=pace_flag,
        )

    if mode == "neet_180" or total_length == 180:
        all_topics = db.query(Topic).all()
        topic_to_subj = {t.topic_id: t.subject for t in all_topics}
        subject_quotas = {"Physics": 45, "Chemistry": 45, "Biology": 90}
        for sub, target_sub_q in subject_quotas.items():
            sub_topics = [tid for tid, s in topic_to_subj.items() if s == sub]
            sub_imp_sum = sum(_TOPIC_IMPORTANCE.get(tid, 1.0) for tid in sub_topics) or 1.0
            sub_paper: list[Question] = []
            remaining = target_sub_q
            for tid in sorted(sub_topics, key=lambda t: _TOPIC_IMPORTANCE.get(t, 1.0), reverse=True):
                raw_target = round((_TOPIC_IMPORTANCE.get(tid, 1.0) / sub_imp_sum) * target_sub_q)
                q_count = min(remaining, max(1, raw_target))
                state = state_by_topic.get(tid)
                mastery_prob = float(state.mastery_prob) if state and getattr(state, "mastery_prob", None) is not None else get_chapter_bkt_prior(tid)
                level = state.level if state and getattr(state, "level", None) else ("advanced" if mastery_prob >= 0.70 else ("medium" if mastery_prob >= 0.45 else "basic"))
                topic_theta = float(getattr(state, "irt_theta", global_theta) or global_theta) if state else global_theta
                pace_flag = state.pace_flag if state else None
                prereq_ready = compute_prerequisite_readiness(tid, mastery_map)
                imp_val = float(_TOPIC_IMPORTANCE.get(tid, 1.0))
                picked = _select_topic_questions(
                    pools=pools,
                    topic_id=tid,
                    level=level,
                    target_count=q_count,
                    attempted_all=attempted_all,
                    mastered_correct=mastered_correct,
                    excluded_ids=excluded_ids,
                    student_id=student_id,
                    topic_theta=topic_theta,
                    mastery_prob=mastery_prob,
                    topic_importance=imp_val,
                    prereq_ready=prereq_ready,
                    pace_flag=pace_flag,
                )
                sub_paper.extend(picked)
                remaining = target_sub_q - len(sub_paper)
                if remaining <= 0:
                    break
            if remaining > 0:
                for tid in sub_topics:
                    state = state_by_topic.get(tid)
                    mastery_prob = float(state.mastery_prob) if state and getattr(state, "mastery_prob", None) is not None else get_chapter_bkt_prior(tid)
                    level = state.level if state and getattr(state, "level", None) else ("advanced" if mastery_prob >= 0.70 else ("medium" if mastery_prob >= 0.45 else "basic"))
                    topic_theta = float(getattr(state, "irt_theta", global_theta) or global_theta) if state else global_theta
                    pace_flag = state.pace_flag if state else None
                    prereq_ready = compute_prerequisite_readiness(tid, mastery_map)
                    imp_val = float(_TOPIC_IMPORTANCE.get(tid, 1.0))
                    picked = _select_topic_questions(
                        pools=pools,
                        topic_id=tid,
                        level=level,
                        target_count=remaining,
                        attempted_all=attempted_all,
                        mastered_correct=mastered_correct,
                        excluded_ids=excluded_ids,
                        student_id=student_id,
                        topic_theta=topic_theta,
                        mastery_prob=mastery_prob,
                        topic_importance=imp_val,
                        prereq_ready=prereq_ready,
                        pace_flag=pace_flag,
                    )
                    sub_paper.extend(picked)
                    remaining = target_sub_q - len(sub_paper)
                    if remaining <= 0:
                        break
            paper.extend(sub_paper)
        return paper

    fractional_topics: list[tuple[float, str]] = []
    for cur_topic_id, importance in _TOPIC_IMPORTANCE.items():
        num_questions = round(importance / 100 * total_length)
        if num_questions == 0:
            fractional_topics.append((float(importance), cur_topic_id))
            continue

        state = state_by_topic.get(cur_topic_id)
        mastery_prob = float(state.mastery_prob) if state and getattr(state, "mastery_prob", None) is not None else get_chapter_bkt_prior(cur_topic_id)
        level = state.level if state and getattr(state, "level", None) else ("advanced" if mastery_prob >= 0.70 else ("medium" if mastery_prob >= 0.45 else "basic"))
        topic_theta = float(getattr(state, "irt_theta", global_theta) or global_theta) if state else global_theta
        pace_flag = state.pace_flag if state else None
        prereq_ready = compute_prerequisite_readiness(cur_topic_id, mastery_map)

        selected = _select_topic_questions(
            pools=pools,
            topic_id=cur_topic_id,
            level=level,
            target_count=num_questions,
            attempted_all=attempted_all,
            mastered_correct=mastered_correct,
            excluded_ids=excluded_ids,
            student_id=student_id,
            topic_theta=topic_theta,
            mastery_prob=mastery_prob,
            topic_importance=float(importance),
            prereq_ready=prereq_ready,
            pace_flag=pace_flag,
        )
        paper.extend(selected)

        if state and state.pace_flag == "slow":
            extra = _select_topic_questions(
                pools=pools,
                topic_id=cur_topic_id,
                level=level,
                target_count=2,
                attempted_all=attempted_all,
                mastered_correct=mastered_correct,
                excluded_ids=excluded_ids,
                student_id=student_id,
                topic_theta=topic_theta,
                mastery_prob=mastery_prob,
                topic_importance=float(importance),
                prereq_ready=prereq_ready,
                pace_flag=pace_flag,
            )
            paper.extend(extra)

        if state and state.level == "advanced" and (state.attempts_counted or 0) >= 3:
            medium_pool = pools.get((cur_topic_id, "medium"), [])
            unseen_probes = [
                q
                for q in medium_pool
                if q.question_id not in excluded_ids and q.question_id not in attempted_all
            ]
            fallback_probes = [q for q in medium_pool if q.question_id not in excluded_ids]
            probe_candidates = unseen_probes if unseen_probes else fallback_probes
            if probe_candidates and len(paper) < total_length + 5:
                probe_q = probe_candidates[0]
                excluded_ids.add(probe_q.question_id)
                for sib_id in get_near_duplicate_siblings_map().get(probe_q.question_id, []):
                    excluded_ids.add(sib_id)
                paper.append(probe_q)

    if len(paper) < total_length and fractional_topics:
        remaining_slots = total_length - len(paper)
        weighted_candidates: list[tuple[float, str]] = []
        for imp, t_id in fractional_topics:
            readiness = compute_prerequisite_readiness(t_id, mastery_map)
            unseen_count = sum(
                1
                for d in ("basic", "medium", "advanced")
                for q in pools.get((t_id, d), [])
                if q.question_id not in attempted_all and q.question_id not in excluded_ids
            )
            freshness_boost = 1.25 if unseen_count > 0 else 0.35
            score = imp * readiness * freshness_boost * random.uniform(0.85, 1.15)
            weighted_candidates.append((score, t_id))
        weighted_candidates.sort(key=lambda pair: pair[0], reverse=True)

        for _, t_id in weighted_candidates[:remaining_slots]:
            state = state_by_topic.get(t_id)
            mastery_prob = float(state.mastery_prob) if state and getattr(state, "mastery_prob", None) is not None else get_chapter_bkt_prior(t_id)
            level = state.level if state and getattr(state, "level", None) else ("advanced" if mastery_prob >= 0.70 else ("medium" if mastery_prob >= 0.45 else "basic"))
            topic_theta = float(getattr(state, "irt_theta", global_theta) or global_theta) if state else global_theta
            pace_flag = state.pace_flag if state else None
            prereq_ready = compute_prerequisite_readiness(t_id, mastery_map)
            imp_val = float(_TOPIC_IMPORTANCE.get(t_id, 0.5))

            picked = _select_topic_questions(
                pools=pools,
                topic_id=t_id,
                level=level,
                target_count=1,
                attempted_all=attempted_all,
                mastered_correct=mastered_correct,
                excluded_ids=excluded_ids,
                student_id=student_id,
                topic_theta=topic_theta,
                mastery_prob=mastery_prob,
                topic_importance=imp_val,
                prereq_ready=prereq_ready,
                pace_flag=pace_flag,
            )
            paper.extend(picked)

    return paper
