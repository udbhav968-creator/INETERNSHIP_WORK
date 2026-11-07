import json
import math
import os
import tempfile
import threading
from pathlib import Path

import numpy as np

from .irt_3pl_model import fisher_information, get_item_params, irt_3pl_probability

CONTEXT_DIM = 6
DEFAULT_ALPHA = 0.35
ZPD_TARGET_PROB = 0.70

_BANDIT_LOCK = threading.RLock()
_MODEL_DIR = Path(__file__).resolve().parent
_PRIOR_INIT_PATH = _MODEL_DIR / "linucb_prior_init.json"


def _load_warm_start_prior() -> tuple[np.ndarray, np.ndarray]:
    if _PRIOR_INIT_PATH.exists():
        try:
            payload = json.loads(_PRIOR_INIT_PATH.read_text(encoding="utf-8"))
            a_mat = np.array(payload["A_init"], dtype=np.float64)
            b_vec = np.array(payload["b_init"], dtype=np.float64)
            if a_mat.shape == (CONTEXT_DIM, CONTEXT_DIM) and b_vec.shape == (CONTEXT_DIM,):
                return a_mat, b_vec
        except Exception:
            pass
    return (
        np.eye(CONTEXT_DIM, dtype=np.float64),
        np.array([0.45, 0.30, 0.25, 0.20, 0.25, 0.15], dtype=np.float64),
    )


_PRIOR_A, _PRIOR_B = _load_warm_start_prior()


def reload_linucb_warm_start_prior() -> tuple[np.ndarray, np.ndarray]:
    global _PRIOR_A, _PRIOR_B
    _PRIOR_A, _PRIOR_B = _load_warm_start_prior()
    return _PRIOR_A, _PRIOR_B


def _resolve_checkpoint_path() -> Path:
    if os.environ.get("VERCEL") or not os.access(_MODEL_DIR, os.W_OK):
        return Path(tempfile.gettempdir()) / "klassed_linucb_state.json"
    return _MODEL_DIR / "linucb_state.json"


_CHECKPOINT_PATH = _resolve_checkpoint_path()


class LinUCBStudentState:
    def __init__(self, alpha: float = DEFAULT_ALPHA):
        self.alpha = alpha
        self.A = _PRIOR_A.copy()
        self.A_inv = np.linalg.inv(self.A)
        self.b = _PRIOR_B.copy()
        self.total_updates = 0

    def update(self, x: np.ndarray, reward: float) -> None:
        x_col = x.reshape(-1, 1)
        self.A += x_col @ x_col.T
        self.A_inv = np.linalg.inv(self.A)
        self.b += float(reward) * x
        self.total_updates += 1

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "A": self.A.tolist(),
            "b": self.b.tolist(),
            "total_updates": self.total_updates,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LinUCBStudentState":
        inst = cls(alpha=float(data.get("alpha", DEFAULT_ALPHA)))
        inst.A = np.array(data["A"], dtype=np.float64)
        inst.A_inv = np.linalg.inv(inst.A)
        inst.b = np.array(data["b"], dtype=np.float64)
        inst.total_updates = int(data.get("total_updates", 0))
        return inst


_STUDENT_BANDITS: dict[str, LinUCBStudentState] = {}


def _load_bandit_checkpoint() -> None:
    src = _CHECKPOINT_PATH if _CHECKPOINT_PATH.exists() else (_MODEL_DIR / "linucb_state.json")
    if not src.exists():
        return
    try:
        with open(src, encoding="utf-8") as f:
            raw = json.load(f)
        for sid, payload in raw.items():
            if isinstance(payload, dict) and "A" in payload and "b" in payload:
                _STUDENT_BANDITS[sid] = LinUCBStudentState.from_dict(payload)
    except Exception:
        pass


def _save_bandit_checkpoint() -> None:
    try:
        payload = {sid: st.to_dict() for sid, st in _STUDENT_BANDITS.items()}
        with open(_CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


def reset_student_bandit(student_id: str, remove: bool = False) -> None:
    with _BANDIT_LOCK:
        if remove:
            _STUDENT_BANDITS.pop(student_id, None)
        else:
            _STUDENT_BANDITS[student_id] = LinUCBStudentState()
        _save_bandit_checkpoint()


_load_bandit_checkpoint()


def get_student_bandit(student_id: str) -> LinUCBStudentState:
    with _BANDIT_LOCK:
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
    prior_mastery_prob: float | None = None,
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

    mastery_gain_bonus = 0.0
    if prior_mastery_prob is not None:
        delta_m = max(0.0, float(topic_mastery_prob) - float(prior_mastery_prob))
        mastery_gain_bonus = min(0.20, delta_m * 1.5)

    reward = round(
        min(1.0, 0.40 * zpd_quality + 0.25 * engagement_quality + 0.25 * outcome_bonus + mastery_gain_bonus),
        4,
    )
    with _BANDIT_LOCK:
        get_student_bandit(student_id).update(x, reward)
        _save_bandit_checkpoint()
    return reward
