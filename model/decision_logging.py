from __future__ import annotations

import os
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field

import numpy as np

POLICY_GREEDY = "linucb_greedy"
POLICY_EPSILON = "linucb_epsilon_greedy"
POLICY_UNIFORM = "uniform_random"
MAX_LOGGED_CANDIDATES = 60

_ACTIVE_LOG: ContextVar["DecisionLog | None"] = ContextVar("klassed_decision_log", default=None)


def exploration_epsilon() -> float:
    raw = os.environ.get("KLASSED_BANDIT_EXPLORATION_EPSILON", "0").strip()
    try:
        eps = float(raw)
    except ValueError:
        eps = 0.0
    return min(0.5, max(0.0, eps))


@dataclass
class RankedSlot:
    question_id: str
    position: int
    propensity: float
    score: float | None


@dataclass
class DecisionRecord:
    request_id: str
    student_id: str
    topic_id: str | None
    question_id: str
    slot: int
    policy: str
    arm: str
    epsilon: float
    propensity: float
    chosen_score: float | None
    candidate_ids: list[str]
    candidate_scores: list[float | None]
    candidate_count: int


@dataclass
class DecisionLog:
    request_id: str
    student_id: str
    arm: str
    records: list[DecisionRecord] = field(default_factory=list)


def rank_with_propensities(
    question_ids: list[str],
    scores: list[float] | None,
    policy: str,
    epsilon: float = 0.0,
    rng: np.random.Generator | None = None,
) -> list[RankedSlot]:
    n = len(question_ids)
    if n == 0:
        return []
    rng = rng or np.random.default_rng()
    if policy == POLICY_GREEDY and scores is not None:
        order = sorted(range(n), key=lambda i: (-scores[i], i))
        return [RankedSlot(question_ids[i], pos, 1.0, float(scores[i])) for pos, i in enumerate(order)]
    if policy == POLICY_UNIFORM or scores is None:
        order = list(rng.permutation(n))
        return [RankedSlot(question_ids[int(i)], pos, 1.0 / (n - pos), None if scores is None else float(scores[int(i)])) for pos, i in enumerate(order)]
    remaining = list(range(n))
    out: list[RankedSlot] = []
    position = 0
    while remaining:
        m = len(remaining)
        if policy == POLICY_UNIFORM or scores is None:
            pick = int(rng.integers(m))
            prop = 1.0 / m
        else:
            best_local = max(range(m), key=lambda j: (scores[remaining[j]], -remaining[j]))
            if policy == POLICY_EPSILON and epsilon > 0 and rng.random() < epsilon:
                pick = int(rng.integers(m))
            else:
                pick = best_local
            if policy == POLICY_EPSILON and epsilon > 0:
                prop = epsilon / m + ((1.0 - epsilon) if pick == best_local else 0.0)
            else:
                prop = 1.0 if pick == best_local else 0.0
        idx = remaining.pop(pick)
        out.append(RankedSlot(question_ids[idx], position, float(prop), None if scores is None else float(scores[idx])))
        position += 1
    return out


def start_decision_log(student_id: str, arm: str) -> DecisionLog:
    log = DecisionLog(request_id=uuid.uuid4().hex, student_id=str(student_id), arm=arm)
    _ACTIVE_LOG.set(log)
    return log


def active_decision_log() -> DecisionLog | None:
    return _ACTIVE_LOG.get()


def finish_decision_log() -> DecisionLog | None:
    log = _ACTIVE_LOG.get()
    _ACTIVE_LOG.set(None)
    return log


def record_choice(
    topic_id: str | None,
    slot: RankedSlot,
    policy: str,
    epsilon: float,
    candidates: list[RankedSlot],
) -> None:
    log = _ACTIVE_LOG.get()
    if log is None:
        return
    ordered = sorted(candidates, key=lambda c: (-(c.score if c.score is not None else 0.0), c.question_id))
    top = ordered[:MAX_LOGGED_CANDIDATES]
    log.records.append(
        DecisionRecord(
            request_id=log.request_id,
            student_id=log.student_id,
            topic_id=topic_id,
            question_id=slot.question_id,
            slot=len(log.records),
            policy=policy,
            arm=log.arm,
            epsilon=float(epsilon),
            propensity=float(slot.propensity),
            chosen_score=slot.score,
            candidate_ids=[c.question_id for c in top],
            candidate_scores=[None if c.score is None else round(float(c.score), 6) for c in top],
            candidate_count=len(candidates),
        )
    )
