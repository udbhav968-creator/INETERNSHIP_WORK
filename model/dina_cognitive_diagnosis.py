from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent
DINA_CALIBRATED_PATH = MODEL_DIR / "dina_calibrated_parameters.json"

COGNITIVE_SKILLS: list[dict[str, str]] = [
    {
        "skill_id": "dimensional_unit_analysis",
        "name": "Dimensional & Unit Conversion",
        "description": "SI/CGS unit consistency, molar stoichiometry, order-of-magnitude scaling",
    },
    {
        "skill_id": "algebraic_numerical_computation",
        "name": "Multi-Step Numerical Computation",
        "description": "Ratio simplification, quadratic/system solving, decimal & fraction precision",
    },
    {
        "skill_id": "log_exp_calculus_reasoning",
        "name": "Logarithmic, Exponential & Rate Calculus",
        "description": "pH/Nernst logs, radioactive/1st-order decay, differentiation & integration",
    },
    {
        "skill_id": "sign_vector_convention",
        "name": "Sign, Vector & Convention Mastery",
        "description": "Thermodynamic work/heat signs, ray-optics conventions, vector/phasor directions",
    },
    {
        "skill_id": "mechanism_electronic_flow",
        "name": "Reaction Mechanism & Equilibrium Logic",
        "description": "Inductive/resonance effects, oxidation states, Le Chatelier & ionic shifts",
    },
    {
        "skill_id": "ncert_exception_trend_recall",
        "name": "NCERT Exceptions & Taxonomic Recall",
        "description": "Inorganic periodic exceptions, biological classification, NCERT factual anchors",
    },
    {
        "skill_id": "diagram_circuit_graph_reading",
        "name": "Circuit, Graph & Structural Interpretation",
        "description": "PV/VT graphs, resistor/capacitor networks, molecular & anatomical structures",
    },
    {
        "skill_id": "assertion_deductive_logic",
        "name": "Assertion–Reason & Multi-Statement Deduction",
        "description": "Negation traps ('not correct'), multi-statement elimination, causal reasoning",
    },
]

SKILL_IDS: list[str] = [s["skill_id"] for s in COGNITIVE_SKILLS]


def _load_calibrated_dina_skills() -> dict[str, dict[str, Any]]:
    if DINA_CALIBRATED_PATH.exists():
        try:
            payload = json.loads(DINA_CALIBRATED_PATH.read_text(encoding="utf-8"))
            skills = payload.get("skills", {})
            if isinstance(skills, dict) and skills:
                return skills
        except Exception:
            pass
    return {}


DINA_CALIBRATED_SKILLS: dict[str, dict[str, Any]] = _load_calibrated_dina_skills()


def reload_calibrated_dina_skills() -> dict[str, dict[str, Any]]:
    global DINA_CALIBRATED_SKILLS
    DINA_CALIBRATED_SKILLS = _load_calibrated_dina_skills()
    return DINA_CALIBRATED_SKILLS


def extract_item_q_vector(
    question_text: str,
    topic_id: str = "",
    difficulty: str = "medium",
) -> dict[str, int]:
    text_low = (question_text or "").lower()
    topic_low = (topic_id or "").lower()

    q_vec: dict[str, int] = {k: 0 for k in SKILL_IDS}

    if re.search(
        r"\b(unit|dimension|si\b|cgs|cm\b|mm\b|km/h|m/s|gram|mole|molarity|molality|normality|ppm|liter|litre|kelvin|°c|joule|erg|ev\b|angstrom)\b",
        text_low,
    ) or any(w in topic_low for w in ("units", "mole concept", "stoichiometry")):
        q_vec["dimensional_unit_analysis"] = 1

    has_numbers = len(re.findall(r"\d+(?:\.\d+)?", text_low)) >= 2
    if has_numbers or re.search(
        r"\b(calculate|find the value|ratio|percentage|fraction|mass of|velocity of|current in|force on|energy of|wavelength)\b",
        text_low,
    ):
        q_vec["algebraic_numerical_computation"] = 1

    if re.search(
        r"\b(log|ln\b|ph\b|pka|pkb|half-life|half life|t1/2|rate constant|first order|radioactive|decay|instantaneous|derivative|integrate|e\^)\b",
        text_low,
    ) or any(w in topic_low for w in ("kinetics", "electrochemistry", "nuclei", "kinematics", "alternating current")):
        q_vec["log_exp_calculus_reasoning"] = 1

    if re.search(
        r"\b(direction|vector|angle|focal length|concave|convex|mirror|lens|work done by|work done on|enthalpy|exothermic|endothermic|potential difference|emf|flux|torque|magnetic field)\b",
        text_low,
    ) or any(
        w in topic_low
        for w in ("optics", "thermodynamics", "electrostatics", "magnetic", "rotational", "laws of motion")
    ):
        q_vec["sign_vector_convention"] = 1

    if re.search(
        r"\b(reaction|reagent|product|mechanism|intermediate|carbocation|nucleophile|electrophile|oxidation|reduction|equilibrium|conjugate|hybridisation|hybridization|bond order|acidity|basicity)\b",
        text_low,
    ) or any(
        w in topic_low
        for w in (
            "organic",
            "hydrocarbons",
            "alcohols",
            "aldehydes",
            "amines",
            "haloalkanes",
            "equilibrium",
            "redox",
            "chemical bonding",
            "coordination",
        )
    ):
        q_vec["mechanism_electronic_flow"] = 1

    if re.search(
        r"\b(exception|which of the following|example of|belongs to|characteristic of|incorrect match|correct match|discovered by|called as|found in|enzyme|hormone|vitamin|disease|phylum|family|genus)\b",
        text_low,
    ) or any(
        w in topic_low
        for w in (
            "block",
            "periodic",
            "metallurgy",
            "kingdom",
            "classification",
            "morphology",
            "anatomy",
            "cell",
            "physiology",
            "reproduction",
            "genetics",
            "ecology",
            "biotechnology",
            "human",
        )
    ):
        q_vec["ncert_exception_trend_recall"] = 1

    if re.search(
        r"\b(graph|diagram|figure|circuit|network|bridge|slope|curve|area under|structure|isomer|chiral|iupac|cycle|pathway|ecg|punnett|pedigree)\b",
        text_low,
    ) or any(w in topic_low for w in ("current electricity", "semiconductor", "biomolecules")):
        q_vec["diagram_circuit_graph_reading"] = 1

    if re.search(
        r"\b(statement|assertion|reason|not correct|incorrect|false|wrong|both .* and|select the correct|choose the correct|how many of the following)\b",
        text_low,
    ) or difficulty == "advanced":
        q_vec["assertion_deductive_logic"] = 1

    active_count = sum(q_vec.values())
    if active_count == 0:
        q_vec["ncert_exception_trend_recall"] = 1
        q_vec["assertion_deductive_logic"] = 1
    elif active_count == 1:
        if q_vec["algebraic_numerical_computation"] == 1:
            q_vec["dimensional_unit_analysis"] = 1
        else:
            q_vec["assertion_deductive_logic"] = 1

    return q_vec


def get_dina_slip_and_guess(difficulty: str, skill_id: str | None = None) -> tuple[float, float]:
    diff = (difficulty or "medium").lower()
    cal = DINA_CALIBRATED_SKILLS.get(skill_id or "", {})
    base_s = float(cal.get("slip", 0.12))
    base_g = float(cal.get("guess", 0.22))
    if diff == "basic":
        return round(max(0.05, base_s - 0.035), 4), round(min(0.30, base_g + 0.035), 4)
    if diff == "advanced":
        return round(min(0.24, base_s + 0.040), 4), round(max(0.14, base_g - 0.035), 4)
    return round(base_s, 4), round(base_g, 4)


def compute_dina_and_rt_profile(attempt_records: list[dict[str, Any]]) -> dict[str, Any]:
    skill_posteriors: dict[str, float] = {
        k: float(DINA_CALIBRATED_SKILLS.get(k, {}).get("prior_mastery", 0.42)) for k in SKILL_IDS
    }
    skill_attempts: dict[str, int] = {k: 0 for k in SKILL_IDS}
    skill_correct: dict[str, int] = {k: 0 for k in SKILL_IDS}

    tau_mean = 0.0
    tau_precision = 1.0 / (0.35 ** 2)
    alpha_time_disc = 1.60

    behavior_counts: dict[str, int] = {
        "fluent_mastery": 0,
        "deep_derivation": 0,
        "careless_slip": 0,
        "rapid_guess": 0,
        "conceptual_struggle": 0,
    }
    time_ratios: list[float] = []

    for rec in attempt_records:
        q_text = str(rec.get("question_text", ""))
        topic_id = str(rec.get("topic_id", ""))
        diff = str(rec.get("difficulty", "medium"))
        exp_t = max(15.0, float(rec.get("expected_time_seconds") or 90.0))
        obs_t = max(1.5, float(rec.get("time_taken_seconds") or exp_t))
        is_corr = bool(rec.get("is_correct", False))

        q_vec = extract_item_q_vector(q_text, topic_id=topic_id, difficulty=diff)
        active_skills = [k for k, v in q_vec.items() if v == 1]

        beta_i = math.log(exp_t)
        ln_t = math.log(obs_t)
        expected_ln_t = beta_i - tau_mean
        z_time = alpha_time_disc * (ln_t - expected_ln_t)
        time_ratios.append(obs_t / exp_t)

        conjunctive_prior = 1.0
        for k in active_skills:
            conjunctive_prior *= skill_posteriors[k]

        is_rapid_guess = (obs_t < 0.22 * exp_t) and (not is_corr)
        if is_rapid_guess:
            behavior_counts["rapid_guess"] += 1
            rt_evidence_weight = 0.35
        elif (not is_corr) and (obs_t < 0.68 * exp_t) and (conjunctive_prior >= 0.48):
            behavior_counts["careless_slip"] += 1
            rt_evidence_weight = 0.70
        elif is_corr and z_time <= 0.30:
            behavior_counts["fluent_mastery"] += 1
            rt_evidence_weight = 1.12
        elif is_corr:
            behavior_counts["deep_derivation"] += 1
            rt_evidence_weight = 0.92
        else:
            behavior_counts["conceptual_struggle"] += 1
            rt_evidence_weight = 1.00

        if obs_t >= 3.0:
            obs_precision = (alpha_time_disc ** 2) * (0.5 if is_rapid_guess else 1.0)
            tau_mean = (tau_precision * tau_mean + obs_precision * (beta_i - ln_t)) / (
                tau_precision + obs_precision
            )
            tau_precision += obs_precision
            tau_mean = float(np.clip(tau_mean, -2.0, 2.0))

        old_posteriors = dict(skill_posteriors)

        for k in active_skills:
            s_i, g_i = get_dina_slip_and_guess(diff, skill_id=k)
            skill_attempts[k] += 1
            if is_corr:
                skill_correct[k] += 1

            other_mastery = 1.0
            for k_other in active_skills:
                if k_other != k:
                    other_mastery *= old_posteriors[k_other]
            other_mastery = max(0.35, other_mastery)

            p_y1_given_k1 = (1.0 - s_i) * other_mastery + g_i * (1.0 - other_mastery)
            p_y1_given_k0 = g_i

            if is_corr:
                lik_k1 = p_y1_given_k1
                lik_k0 = p_y1_given_k0
            else:
                lik_k1 = 1.0 - p_y1_given_k1
                lik_k0 = 1.0 - p_y1_given_k0

            prior_k = old_posteriors[k]
            denom = prior_k * lik_k1 + (1.0 - prior_k) * lik_k0
            raw_post = (prior_k * lik_k1 / denom) if denom > 1e-9 else prior_k

            updated_k = prior_k + rt_evidence_weight * (raw_post - prior_k)
            skill_posteriors[k] = float(np.clip(updated_k, 0.08, 0.98))

    skills_out: list[dict[str, Any]] = []
    for meta in COGNITIVE_SKILLS:
        k = meta["skill_id"]
        cal = DINA_CALIBRATED_SKILLS.get(k, {})
        n_att = skill_attempts[k]
        p_val = skill_posteriors[k]
        if n_att == 0:
            post_sd = float(cal.get("prior_sd", math.sqrt(p_val * (1.0 - p_val) / 7.0)))
            status = "prior_calibrated"
        else:
            post_sd = math.sqrt(p_val * (1.0 - p_val) / (n_att + 6.0))
            if p_val >= 0.72:
                status = "mastered"
            elif p_val >= 0.45:
                status = "developing"
            else:
                status = "bottleneck"

        skills_out.append(
            {
                "skill_id": k,
                "name": meta["name"],
                "description": meta["description"],
                "mastery_probability": round(p_val, 4),
                "posterior_sd": round(post_sd, 4),
                "attempts_count": n_att,
                "correct_count": skill_correct[k],
                "corpus_pyq_count": int(cal.get("corpus_pyq_count", 0)),
                "status": status,
            }
        )

    attempted_skills = [s for s in skills_out if s["attempts_count"] > 0]
    if attempted_skills:
        weakest = min(attempted_skills, key=lambda s: s["mastery_probability"])
        primary_bottleneck = {
            "skill_id": weakest["skill_id"],
            "name": weakest["name"],
            "mastery_probability": weakest["mastery_probability"],
            "diagnosis": (
                f"Weakest cognitive area: {weakest['name']} "
                f"({round(weakest['mastery_probability'] * 100)}% mastery across {weakest['attempts_count']} items) - {weakest['description']}."
            ),
        }
    else:
        weakest_prior = min(skills_out, key=lambda s: s["mastery_probability"])
        primary_bottleneck = {
            "skill_id": weakest_prior["skill_id"],
            "name": weakest_prior["name"],
            "mastery_probability": weakest_prior["mastery_probability"],
            "diagnosis": (
                f"Corpus prior bottleneck: {weakest_prior['name']} "
                f"({round(weakest_prior['mastery_probability'] * 100, 1)}% NEET baseline across {weakest_prior['corpus_pyq_count']} PYQs) - "
                f"complete an adaptive paper to personalize."
            ),
        }

    speed_z = tau_mean / 0.55
    speed_percentile = round(float(np.clip(50.0 * (1.0 + math.erf(speed_z / math.sqrt(2.0))), 1.0, 99.9)), 1)
    tau_se = round(1.0 / math.sqrt(tau_precision), 3)

    return {
        "model_type": "DINA_8_Skill_Q_Matrix_and_van_der_Linden_LogNormal_RT",
        "skills": skills_out,
        "primary_cognitive_bottleneck": primary_bottleneck,
        "speed_ability_profile": {
            "latent_speed_tau": round(tau_mean, 4),
            "latent_speed_se": tau_se,
            "speed_percentile": speed_percentile,
            "avg_time_ratio": round(float(np.mean(time_ratios)), 3) if time_ratios else 1.0,
            "behavior_classification": behavior_counts,
        },
    }
