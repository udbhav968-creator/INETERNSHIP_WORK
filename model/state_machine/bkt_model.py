import json
import math
from dataclasses import dataclass
from pathlib import Path

_BKT_JSON_PATH = Path(__file__).resolve().parent.parent / "bkt_chapter_parameters.json"


@dataclass
class BKTParams:
    p_init: float = 0.38
    p_transit: float = 0.15
    p_guess: float = 0.22
    p_slip: float = 0.10
    half_life_hours: float = 24.0


DIFFICULTY_BKT_PRESETS: dict[str, BKTParams] = {
    "basic": BKTParams(p_init=0.42, p_transit=0.18, p_guess=0.25, p_slip=0.08, half_life_hours=26.0),
    "medium": BKTParams(p_init=0.38, p_transit=0.15, p_guess=0.22, p_slip=0.10, half_life_hours=24.0),
    "advanced": BKTParams(p_init=0.29, p_transit=0.12, p_guess=0.18, p_slip=0.12, half_life_hours=20.0),
}

CHAPTER_BKT_REGISTRY: dict[str, dict] = {}
WEIGHTED_GLOBAL_PRIOR: float = 0.38


def reload_chapter_bkt_registry() -> dict[str, dict]:
    global CHAPTER_BKT_REGISTRY, WEIGHTED_GLOBAL_PRIOR
    if _BKT_JSON_PATH.exists():
        raw = json.loads(_BKT_JSON_PATH.read_text(encoding="utf-8"))
        CHAPTER_BKT_REGISTRY = raw.get("chapters", {})
        WEIGHTED_GLOBAL_PRIOR = float(raw.get("weighted_global_prior", 0.38))
    return CHAPTER_BKT_REGISTRY


reload_chapter_bkt_parameters = reload_chapter_bkt_registry
reload_chapter_bkt_registry()


def get_chapter_bkt_prior(topic_id: str | None = None) -> float:
    if topic_id and topic_id in CHAPTER_BKT_REGISTRY:
        return float(CHAPTER_BKT_REGISTRY[topic_id]["p_init"])
    return WEIGHTED_GLOBAL_PRIOR


def get_chapter_bkt_params(topic_id: str | None = None, difficulty: str = "medium") -> BKTParams:
    preset = DIFFICULTY_BKT_PRESETS.get(difficulty, DIFFICULTY_BKT_PRESETS["medium"])
    if not topic_id or topic_id not in CHAPTER_BKT_REGISTRY:
        return preset
    ch = CHAPTER_BKT_REGISTRY[topic_id]
    diff_g_mod = 0.025 if difficulty == "basic" else (-0.025 if difficulty == "advanced" else 0.0)
    diff_s_mod = -0.015 if difficulty == "basic" else (0.020 if difficulty == "advanced" else 0.0)
    diff_t_mod = 0.020 if difficulty == "basic" else (-0.020 if difficulty == "advanced" else 0.0)
    return BKTParams(
        p_init=float(ch["p_init"]),
        p_transit=max(0.06, min(0.28, float(ch["p_transit"]) + diff_t_mod)),
        p_guess=max(0.14, min(0.30, float(ch["p_guess"]) + diff_g_mod)),
        p_slip=max(0.05, min(0.20, float(ch["p_slip"]) + diff_s_mod)),
        half_life_hours=float(ch.get("half_life_base_hours", 24.0)),
    )


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
    base_prior: float | None = None,
    topic_id: str | None = None,
) -> float:
    resolved_base = base_prior if base_prior is not None else get_chapter_bkt_prior(topic_id)
    p_prev = max(0.01, min(0.99, prior_mastery if prior_mastery is not None else resolved_base))
    if elapsed_hours <= 0.5 or p_prev <= resolved_base:
        return p_prev
    ch_hl = (
        float(CHAPTER_BKT_REGISTRY[topic_id].get("half_life_base_hours", 24.0))
        if topic_id and topic_id in CHAPTER_BKT_REGISTRY
        else 24.0
    )
    half_life_hours = ch_hl * (1.0 + 0.85 * max(0, int(streak))) * (1.0 + p_prev)
    retention_factor = math.pow(2.0, -float(elapsed_hours) / max(1.0, half_life_hours))
    decayed = resolved_base + (p_prev - resolved_base) * retention_factor
    return round(max(0.05, min(0.99, decayed)), 4)


def compute_retention_urgency(
    prior_mastery: float,
    elapsed_hours: float = 0.0,
    streak: int = 0,
    base_prior: float | None = None,
    topic_id: str | None = None,
) -> float:
    resolved_base = base_prior if base_prior is not None else get_chapter_bkt_prior(topic_id)
    peak = max(0.01, min(0.99, prior_mastery if prior_mastery is not None else resolved_base))
    if elapsed_hours <= 0.5:
        return 0.0
    decayed = compute_ebbinghaus_forgetting_prior(peak, elapsed_hours, streak, resolved_base, topic_id=topic_id)
    decay_drop = max(0.0, peak - decayed)
    forgetting_urgency = decay_drop * 3.5
    remediation_urgency = (
        (0.45 - peak) * min(1.0, float(elapsed_hours) / 48.0) * 2.5 if peak < 0.45 else 0.0
    )
    return round(min(1.0, max(forgetting_urgency, remediation_urgency)), 4)


def compute_bkt_update(
    prior_mastery: float,
    is_correct: bool,
    difficulty: str = "medium",
    time_taken_seconds: int = 75,
    expected_time_seconds: int = 90,
    option_switches: int = 0,
    hesitation_ms: int = 0,
    attempts_count: int = 1,
    elapsed_hours: float = 0.0,
    streak: int = 0,
    topic_id: str | None = None,
) -> tuple[float, float]:
    ch_prior = get_chapter_bkt_prior(topic_id)
    p_prev = compute_ebbinghaus_forgetting_prior(
        prior_mastery=prior_mastery if prior_mastery is not None else ch_prior,
        elapsed_hours=elapsed_hours,
        streak=streak,
        base_prior=ch_prior,
        topic_id=topic_id,
    )
    base_params = get_chapter_bkt_params(topic_id=topic_id, difficulty=difficulty)
    p_transit, p_guess, p_slip = modulate_params_with_telemetry(
        base_params,
        time_taken_seconds=time_taken_seconds,
        expected_time_seconds=expected_time_seconds,
        option_switches=option_switches,
        hesitation_ms=hesitation_ms,
    )

    if is_correct:
        numerator = p_prev * (1.0 - p_slip)
        denominator = numerator + (1.0 - p_prev) * p_guess
    else:
        numerator = p_prev * p_slip
        denominator = numerator + (1.0 - p_prev) * (1.0 - p_guess)

    posterior = numerator / denominator if denominator > 0 else p_prev
    effective_transit = p_transit if is_correct else (0.35 * p_transit)
    p_next = posterior + (1.0 - posterior) * effective_transit
    p_next = round(max(0.05, min(0.95, p_next)), 4)

    effective_n = max(1, attempts_count) + 4
    std_err = math.sqrt(p_next * (1.0 - p_next) / effective_n)
    ci_95 = round(min(0.35, 1.96 * std_err), 4)

    return p_next, ci_95


def get_credible_bounds(mastery_prob: float, ci_half_width: float) -> tuple[float, float]:
    lower = round(max(0.0, mastery_prob - ci_half_width), 4)
    upper = round(min(1.0, mastery_prob + ci_half_width), 4)
    return lower, upper


def resolve_level_with_hysteresis(
    current_level: str | float,
    mastery_prob: float | str,
    ci_95: float = 0.25,
) -> str:
    if isinstance(current_level, (int, float)) and isinstance(mastery_prob, str):
        current_level, mastery_prob = mastery_prob, current_level
    level = current_level or "basic"
    conservative_lower = mastery_prob - 0.35 * float(ci_95)

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
