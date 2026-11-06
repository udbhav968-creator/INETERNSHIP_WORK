import math
from dataclasses import dataclass


@dataclass
class BKTParams:
    p_init: float = 0.35
    p_transit: float = 0.15
    p_guess: float = 0.22
    p_slip: float = 0.10


DIFFICULTY_BKT_PRESETS: dict[str, BKTParams] = {
    "basic": BKTParams(p_init=0.40, p_transit=0.18, p_guess=0.25, p_slip=0.08),
    "medium": BKTParams(p_init=0.35, p_transit=0.15, p_guess=0.22, p_slip=0.10),
    "advanced": BKTParams(p_init=0.28, p_transit=0.12, p_guess=0.18, p_slip=0.12),
}

PROMOTE_TO_MEDIUM_THRESHOLD = 0.52
DEMOTE_TO_BASIC_THRESHOLD = 0.40
PROMOTE_TO_ADVANCED_THRESHOLD = 0.78
DEMOTE_TO_MEDIUM_THRESHOLD = 0.60


def modulate_params_with_telemetry(
    params: BKTParams,
    time_taken_seconds: int,
    expected_time_seconds: int,
    option_switches: int = 0,
    hesitation_ms: int = 0,
) -> tuple[float, float, float]:
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


def compute_ebbinghaus_forgetting_prior(
    prior_mastery: float,
    elapsed_hours: float = 0.0,
    streak: int = 0,
    base_prior: float = 0.35,
) -> float:
    p_prev = max(0.01, min(0.99, prior_mastery if prior_mastery is not None else base_prior))
    if elapsed_hours <= 0.5 or p_prev <= base_prior:
        return p_prev
    half_life_hours = 24.0 * (1.0 + 0.85 * max(0, int(streak))) * (1.0 + p_prev)
    retention_factor = math.pow(2.0, -float(elapsed_hours) / max(1.0, half_life_hours))
    decayed = base_prior + (p_prev - base_prior) * retention_factor
    return round(max(0.05, min(0.99, decayed)), 4)


def compute_retention_urgency(
    prior_mastery: float, elapsed_hours: float = 0.0, streak: int = 0, base_prior: float = 0.35
) -> float:
    decayed = compute_ebbinghaus_forgetting_prior(prior_mastery, elapsed_hours, streak, base_prior)
    peak = max(0.01, min(0.99, prior_mastery if prior_mastery is not None else base_prior))
    drop = max(0.0, peak - decayed)
    urgency = min(1.0, (drop / 0.30) * (float(elapsed_hours) / 48.0))
    return round(urgency, 4)


def compute_bkt_update(
    prior_mastery: float,
    is_correct: bool,
    difficulty: str = "medium",
    time_taken_seconds: int = 90,
    expected_time_seconds: int = 90,
    option_switches: int = 0,
    hesitation_ms: int = 0,
    attempts_count: int = 1,
    elapsed_hours: float = 0.0,
    streak: int = 0,
) -> tuple[float, float]:
    p_decayed = compute_ebbinghaus_forgetting_prior(prior_mastery, elapsed_hours, streak)
    p_t = max(0.05, min(0.95, p_decayed))

    base_params = DIFFICULTY_BKT_PRESETS.get(difficulty, DIFFICULTY_BKT_PRESETS["medium"])
    p_transit, p_guess, p_slip = modulate_params_with_telemetry(
        base_params,
        time_taken_seconds=time_taken_seconds,
        expected_time_seconds=expected_time_seconds,
        option_switches=option_switches,
        hesitation_ms=hesitation_ms,
    )

    if is_correct:
        numerator = p_t * (1.0 - p_slip)
        denominator = numerator + (1.0 - p_t) * p_guess
    else:
        numerator = p_t * p_slip
        denominator = numerator + (1.0 - p_t) * (1.0 - p_guess)

    p_updated = numerator / max(1e-6, denominator)

    effective_transit = p_transit if is_correct else (0.35 * p_transit)
    p_next = p_updated + (1.0 - p_updated) * effective_transit
    p_next = max(0.05, min(0.95, p_next))

    n = max(1, attempts_count)
    se = math.sqrt(p_next * (1.0 - p_next) / (n + 2.0))
    ci_half_width = 1.96 * se

    return round(p_next, 4), round(ci_half_width, 4)


def resolve_level_with_hysteresis(
    current_level: str,
    mastery_prob: float,
    rolling_results: list[bool],
) -> str:
    level = current_level
    results = list(rolling_results)
    m = mastery_prob

    if level == "basic":
        if sum(1 for r in results[-3:] if r) >= 2 and m >= PROMOTE_TO_MEDIUM_THRESHOLD:
            return "medium"
        return "basic"

    if level == "medium":
        if len(results) >= 3 and all(results[-3:]) and m >= PROMOTE_TO_ADVANCED_THRESHOLD:
            return "advanced"
        if sum(1 for r in results[-3:] if not r) >= 2 and m <= DEMOTE_TO_BASIC_THRESHOLD:
            return "basic"
        return "medium"

    if level == "advanced":
        if len(results) >= 2 and (not results[-1]) and (not results[-2]) and m <= DEMOTE_TO_MEDIUM_THRESHOLD:
            return "medium"
        return "advanced"

    return level
