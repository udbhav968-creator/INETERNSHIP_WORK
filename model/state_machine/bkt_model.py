"""
Bayesian Knowledge Tracing (BKT) & SAINT+-Inspired Sequence Tracker — Enterprise v2.

Lives in `model/state_machine/` as specified in `docs/spec.md` Section 11.
Replaces rigid binary rolling windows with continuous mastery probability P(L_t)
and smoothed Bayesian credible intervals to eliminate tier boundary oscillation.
"""
import math
from dataclasses import dataclass


@dataclass
class BKTParams:
    p_init: float = 0.35      # P(L_0): EdNet / cohort-informed cold-start prior
    p_transit: float = 0.15   # P(T): learning transition rate per attempt
    p_guess: float = 0.22     # P(G): 4-option MCQ guess rate
    p_slip: float = 0.10      # P(S): careless slip rate despite mastery


DIFFICULTY_BKT_PRESETS: dict[str, BKTParams] = {
    "basic": BKTParams(p_init=0.40, p_transit=0.18, p_guess=0.25, p_slip=0.08),
    "medium": BKTParams(p_init=0.35, p_transit=0.15, p_guess=0.22, p_slip=0.10),
    "advanced": BKTParams(p_init=0.28, p_transit=0.12, p_guess=0.18, p_slip=0.12),
}

PROMOTE_TO_MEDIUM_THRESHOLD = 0.52
DEMOTE_TO_BASIC_THRESHOLD = 0.35
PROMOTE_TO_ADVANCED_THRESHOLD = 0.78
DEMOTE_TO_MEDIUM_THRESHOLD = 0.60


def modulate_params_with_telemetry(
    params: BKTParams,
    time_taken_seconds: int,
    expected_time_seconds: int,
    option_switches: int = 0,
    hesitation_ms: int = 0,
) -> tuple[float, float, float]:
    """
    SAINT+-inspired attention modulation over response time ratio, option-switching
    frequency, and hesitation latency.
    """
    p_t, p_g, p_s = params.p_transit, params.p_guess, params.p_slip
    time_ratio = time_taken_seconds / max(1, expected_time_seconds)

    if time_ratio < 0.15:
        p_g = min(0.45, p_g * 1.65)
        p_t = max(0.04, p_t * 0.55)
    elif 0.40 <= time_ratio <= 1.15:
        p_t = min(0.28, p_t * 1.15)

    if option_switches >= 2:
        p_g = min(0.40, p_g + 0.04 * min(option_switches, 4))
        p_s = min(0.22, p_s + 0.02 * min(option_switches, 4))

    if hesitation_ms > 45_000:
        p_t = max(0.05, p_t * 0.85)

    return p_t, p_g, p_s


def compute_bkt_update(
    prior_mastery: float,
    is_correct: bool,
    difficulty: str = "medium",
    time_taken_seconds: int = 75,
    expected_time_seconds: int = 90,
    option_switches: int = 0,
    hesitation_ms: int = 0,
    attempts_count: int = 1,
) -> tuple[float, float]:
    """
    Computes posterior mastery probability P(L_{t+1}) and 95% Bayesian credible
    interval half-width.
    """
    p_prev = max(0.01, min(0.99, prior_mastery if prior_mastery is not None else 0.35))
    params = DIFFICULTY_BKT_PRESETS.get(difficulty, DIFFICULTY_BKT_PRESETS["medium"])

    p_t, p_g, p_s = modulate_params_with_telemetry(
        params=params,
        time_taken_seconds=time_taken_seconds,
        expected_time_seconds=expected_time_seconds,
        option_switches=option_switches,
        hesitation_ms=hesitation_ms,
    )

    if is_correct:
        num = p_prev * (1.0 - p_s)
        den = num + (1.0 - p_prev) * p_g
    else:
        num = p_prev * p_s
        den = num + (1.0 - p_prev) * (1.0 - p_g)

    posterior = num / den if den > 0 else p_prev
    p_next = posterior + (1.0 - posterior) * p_t
    p_next = round(max(0.02, min(0.99, p_next)), 4)

    effective_n = max(1, attempts_count) + 4
    std_err = math.sqrt((p_next * (1.0 - p_next)) / effective_n)
    ci_95 = round(min(0.35, 1.96 * std_err), 4)

    return p_next, ci_95


def resolve_level_with_hysteresis(
    current_level: str, mastery_prob: float, ci_95: float
) -> str:
    """
    Applies smoothed Bayesian confidence intervals to prevent rapid tier oscillation.
    """
    level = current_level or "basic"
    conservative_lower = mastery_prob - 0.35 * ci_95

    if level == "basic":
        if conservative_lower >= PROMOTE_TO_MEDIUM_THRESHOLD or mastery_prob >= 0.58:
            return "medium"
        return "basic"

    if level == "medium":
        if conservative_lower >= PROMOTE_TO_ADVANCED_THRESHOLD or mastery_prob >= 0.82:
            return "advanced"
        if mastery_prob <= DEMOTE_TO_BASIC_THRESHOLD:
            return "basic"
        return "medium"

    if level == "advanced":
        if mastery_prob <= DEMOTE_TO_MEDIUM_THRESHOLD:
            return "medium"
        return "advanced"

    return "basic"
