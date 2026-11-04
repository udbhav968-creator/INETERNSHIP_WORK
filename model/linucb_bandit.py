"""
Contextual Multi-Armed Bandits (LinUCB) Question Router — Enterprise v2.

Optimizes active question routing within the student's Zone of Proximal
Development (ZPD, target P(correct) ~ 0.70) by balancing exploitation of weak
spots with exploration of borderline topics.
"""
import math
import numpy as np
from .irt_3pl_model import get_item_params, irt_3pl_probability, fisher_information

CONTEXT_DIM = 6
DEFAULT_ALPHA = 0.35
ZPD_TARGET_PROB = 0.70

_PRIOR_B = np.array([0.45, 0.30, 0.25, 0.20, 0.25, 0.15], dtype=np.float64)


class LinUCBStudentState:
    def __init__(self, alpha: float = DEFAULT_ALPHA):
        self.alpha = alpha
        self.A = np.eye(CONTEXT_DIM, dtype=np.float64)
        self.A_inv = np.eye(CONTEXT_DIM, dtype=np.float64)
        self.b = _PRIOR_B.copy()
        self.total_updates = 0

    def update(self, x: np.ndarray, reward: float) -> None:
        x_col = x.reshape(-1, 1)
        self.A += x_col @ x_col.T
        self.A_inv = np.linalg.inv(self.A)
        self.b += float(reward) * x
        self.total_updates += 1


_STUDENT_BANDITS: dict[str, LinUCBStudentState] = {}


def get_student_bandit(student_id: str) -> LinUCBStudentState:
    if student_id not in _STUDENT_BANDITS:
        _STUDENT_BANDITS[student_id] = LinUCBStudentState()
    return _STUDENT_BANDITS[student_id]


def build_context_vector(
    question_id: str,
    difficulty: str,
    expected_time_seconds: int,
    student_theta: float,
    topic_mastery_prob: float,
    topic_importance: float,
    prereq_readiness: float,
    pace_flag: str | None,
) -> np.ndarray:
    irt_params = get_item_params(question_id, difficulty, expected_time_seconds)
    pred_prob = irt_3pl_probability(student_theta, irt_params)
    info = fisher_information(student_theta, irt_params)

    zpd_alignment = max(0.0, 1.0 - abs(pred_prob - ZPD_TARGET_PROB) * 1.5)
    norm_info = min(1.0, info / 1.5)
    mastery_deficit = max(0.0, 1.0 - topic_mastery_prob)
    norm_importance = min(1.0, topic_importance / 5.5)
    readiness = max(0.1, min(1.0, prereq_readiness))
    pace_signal = 1.0 if pace_flag == "slow" else 0.25

    return np.array(
        [zpd_alignment, norm_info, mastery_deficit, norm_importance, readiness, pace_signal],
        dtype=np.float64,
    )


def score_candidate_ucb(student_id: str, x: np.ndarray) -> tuple[float, float, float]:
    bandit = get_student_bandit(student_id)
    w_hat = bandit.A_inv @ bandit.b
    exploitation = float(x @ w_hat)
    uncertainty = float(math.sqrt(max(1e-9, float(x @ bandit.A_inv @ x))))
    exploration = bandit.alpha * uncertainty
    return exploitation + exploration, exploitation, exploration


def record_bandit_feedback(
    student_id: str,
    question_id: str,
    difficulty: str,
    expected_time_seconds: int,
    student_theta: float,
    topic_mastery_prob: float,
    topic_importance: float,
    prereq_readiness: float,
    pace_flag: str | None,
    is_correct: bool,
    time_taken_seconds: int,
) -> float:
    x = build_context_vector(
        question_id=question_id,
        difficulty=difficulty,
        expected_time_seconds=expected_time_seconds,
        student_theta=student_theta,
        topic_mastery_prob=topic_mastery_prob,
        topic_importance=topic_importance,
        prereq_readiness=prereq_readiness,
        pace_flag=pace_flag,
    )

    irt_params = get_item_params(question_id, difficulty, expected_time_seconds)
    pred_prob = irt_3pl_probability(student_theta, irt_params)

    zpd_quality = max(0.0, 1.0 - abs(pred_prob - ZPD_TARGET_PROB))
    time_ratio = time_taken_seconds / max(1, expected_time_seconds)
    engagement_quality = 1.0 if 0.25 <= time_ratio <= 1.8 else 0.5
    outcome_bonus = 0.85 if is_correct else 0.65

    reward = round(0.45 * zpd_quality + 0.30 * engagement_quality + 0.25 * outcome_bonus, 4)
    get_student_bandit(student_id).update(x, reward)
    return reward
