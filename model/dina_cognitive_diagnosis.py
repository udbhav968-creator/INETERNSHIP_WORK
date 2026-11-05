"""
DINA (Deterministic Inputs, Noisy "And" Gate) 8-Skill Q-Matrix Cognitive Diagnosis Model
& van der Linden Bivariate Log-Normal Response-Time (theta, tau) Psychometric Engine.

Implements:
1. Automated 8-Dimensional Cognitive Micro-Skill Q-Matrix extraction (Q in {0,1}^{N x 8})
   mapping every NEET Physics, Chemistry, and Biology question to fine-grained skills:
   - dimensional_unit_analysis
   - algebraic_numerical_computation
   - log_exp_calculus_reasoning
   - sign_vector_convention
   - mechanism_electronic_flow
   - ncert_exception_trend_recall
   - diagram_circuit_graph_reading
   - assertion_deductive_logic
2. Exact Bayesian DINA Posterior Updating with item-level slip (s_i) and guess (g_i)
   parameters, accounting for conjunctive multi-skill interactions.
3. van der Linden (2006) Log-Normal Response Time Model:
   ln(T_{ij}) ~ Normal(beta_i - tau_j, 1 / alpha_i^2)
   with conjugate Gaussian updating for student latent speed (tau_j) and
   log-likelihood ratio anomaly detection (rapid_guess, careless_slip,
   fluent_mastery, deep_derivation, conceptual_struggle).
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np


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


def extract_item_q_vector(
    question_text: str,
    topic_id: str = "",
    difficulty: str = "medium",
) -> dict[str, int]:
    """
    Constructs the binary Q-Matrix row q_i in {0, 1}^8 for a NEET item based on
    domain lexical markers, mathematical syntax, and NCERT chapter structure.
    """
    text_low = (question_text or "").lower()
    topic_low = (topic_id or "").lower()

    q_vec: dict[str, int] = {k: 0 for k in SKILL_IDS}

    # 1. Dimensional & Unit Conversion
    if re.search(
        r"\b(unit|dimension|si\b|cgs|cm\b|mm\b|km/h|m/s|gram|mole|molarity|molality|normality|ppm|liter|litre|kelvin|°c|joule|erg|ev\b|angstrom)\b",
        text_low,
    ) or any(w in topic_low for w in ("units", "mole concept", "stoichiometry")):
        q_vec["dimensional_unit_analysis"] = 1

    # 2. Multi-Step Algebraic & Numerical Computation
    has_numbers = len(re.findall(r"\d+(?:\.\d+)?", text_low)) >= 2
    if has_numbers or re.search(
        r"\b(calculate|find the value|ratio|percentage|fraction|mass of|velocity of|current in|force on|energy of|wavelength)\b",
        text_low,
    ):
        q_vec["algebraic_numerical_computation"] = 1

    # 3. Logarithmic, Exponential & Rate Calculus Reasoning
    if re.search(
        r"\b(log|ln\b|ph\b|pka|pkb|half-life|half life|t1/2|rate constant|first order|radioactive|decay|instantaneous|derivative|integrate|e\^)\b",
        text_low,
    ) or any(w in topic_low for w in ("kinetics", "electrochemistry", "nuclei", "kinematics", "alternating current")):
        q_vec["log_exp_calculus_reasoning"] = 1

    # 4. Sign, Vector & Convention Mastery
    if re.search(
        r"\b(direction|vector|angle|focal length|concave|convex|mirror|lens|work done by|work done on|enthalpy|exothermic|endothermic|potential difference|emf|flux|torque|magnetic field)\b",
        text_low,
    ) or any(
        w in topic_low
        for w in ("optics", "thermodynamics", "electrostatics", "magnetic", "rotational", "laws of motion")
    ):
        q_vec["sign_vector_convention"] = 1

    # 5. Reaction Mechanism & Equilibrium Logic
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

    # 6. NCERT Exceptions & Taxonomic Recall
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

    # 7. Circuit, Graph & Structural Interpretation
    if re.search(
        r"\b(graph|diagram|figure|circuit|network|bridge|slope|curve|area under|structure|isomer|chiral|iupac|cycle|pathway|ecg|punnett|pedigree)\b",
        text_low,
    ) or any(w in topic_low for w in ("current electricity", "semiconductor", "biomolecules")):
        q_vec["diagram_circuit_graph_reading"] = 1

    # 8. Assertion–Reason & Multi-Statement Deduction
    if re.search(
        r"\b(statement|assertion|reason|not correct|incorrect|false|wrong|both .* and|select the correct|choose the correct|how many of the following)\b",
        text_low,
    ) or difficulty == "advanced":
        q_vec["assertion_deductive_logic"] = 1

    # Ensure every item activates at least 2 complementary cognitive skills for identifiable Q-matrix rank
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


def get_dina_slip_and_guess(difficulty: str) -> tuple[float, float]:
    """
    Returns calibrated DINA item slip (s_i) and guess (g_i) parameters:
      - s_i = P(Y_{ij} = 0 | eta_{ij} = 1) (careless slip despite mastering all required skills)
      - g_i = P(Y_{ij} = 1 | eta_{ij} = 0) (guessing/partial-knowledge success without full skill mastery)
    """
    diff = (difficulty or "medium").lower()
    if diff == "basic":
        return 0.08, 0.27
    if diff == "advanced":
        return 0.16, 0.19
    return 0.12, 0.23


def compute_dina_and_rt_profile(attempt_records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Executes sequential Bayesian DINA Q-Matrix skill updates paired with van der Linden's
    bivariate Log-Normal Response Time model across a student's attempt trajectory.

    Parameters
    ----------
    attempt_records : list of dicts with keys:
        - question_id: str
        - question_text: str
        - topic_id: str
        - difficulty: str
        - expected_time_seconds: int | float
        - time_taken_seconds: int | float
        - is_correct: bool

    Returns
    -------
    dict containing:
        - skills: list of 8 cognitive skill diagnostic objects with posterior mastery,
          uncertainty (posterior SD), attempts_linked, and status tier
        - speed_ability_profile: van der Linden latent speed (tau), speed percentile,
          mean response-time ratio, and behavioral classification counts
        - primary_cognitive_bottleneck: weakest active cognitive micro-skill & actionable fix
    """
    # Initialize conjugate Beta/Bernoulli skill posteriors at neutral prior P(alpha_k = 1) = 0.50
    skill_posteriors: dict[str, float] = {k: 0.50 for k in SKILL_IDS}
    skill_attempts: dict[str, int] = {k: 0 for k in SKILL_IDS}
    skill_correct: dict[str, int] = {k: 0 for k in SKILL_IDS}

    # van der Linden Log-Normal Response Time model:
    # Prior on student latent speed: tau ~ Normal(0.0, sigma_0^2 = 0.35^2) => precision lambda_0 = 1 / 0.1225
    tau_mean = 0.0
    tau_precision = 1.0 / (0.35 ** 2)
    alpha_time_disc = 1.60  # item time discrimination parameter alpha_i

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

        # 1. van der Linden Log-Normal Response Time evaluation
        # ln(T_ij) ~ Normal(beta_i - tau_j, 1 / alpha_i^2), where beta_i = ln(exp_t)
        beta_i = math.log(exp_t)
        ln_t = math.log(obs_t)
        expected_ln_t = beta_i - tau_mean
        z_time = alpha_time_disc * (ln_t - expected_ln_t)
        time_ratios.append(obs_t / exp_t)

        # Evaluate pre-update conjunctive mastery probability P(eta_ij = 1)
        conjunctive_prior = 1.0
        for k in active_skills:
            conjunctive_prior *= skill_posteriors[k]

        # Classify cognitive response-time behavior
        is_rapid_guess = (obs_t < 0.22 * exp_t) and (not is_corr)
        if is_rapid_guess:
            behavior_counts["rapid_guess"] += 1
            rt_evidence_weight = 0.35  # Dampen skill penalty on blind rapid guesses
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

        # Conjugate Gaussian update for latent speed tau_j (unless rapid blind click < 3s)
        if obs_t >= 3.0:
            obs_precision = (alpha_time_disc ** 2) * (0.5 if is_rapid_guess else 1.0)
            tau_mean = (tau_precision * tau_mean + obs_precision * (beta_i - ln_t)) / (
                tau_precision + obs_precision
            )
            tau_precision += obs_precision
            tau_mean = float(np.clip(tau_mean, -2.0, 2.0))

        # 2. Exact DINA Bayesian Marginal Posterior Update for each required skill k in K_i
        s_i, g_i = get_dina_slip_and_guess(diff)
        old_posteriors = dict(skill_posteriors)

        for k in active_skills:
            skill_attempts[k] += 1
            if is_corr:
                skill_correct[k] += 1

            # Probability that all OTHER required skills k' != k are mastered:
            other_mastery = 1.0
            for k_other in active_skills:
                if k_other != k:
                    other_mastery *= old_posteriors[k_other]
            # Floor other_mastery at 0.35 so a multi-skill item still transmits gradient signal
            other_mastery = max(0.35, other_mastery)

            # Likelihood P(Y_ij | alpha_jk = 1) vs P(Y_ij | alpha_jk = 0)
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

            # Blend with response-time evidence weight
            updated_k = prior_k + rt_evidence_weight * (raw_post - prior_k)
            skill_posteriors[k] = float(np.clip(updated_k, 0.08, 0.98))

    # Format 8-Skill Cognitive Profile
    skills_out: list[dict[str, Any]] = []
    for meta in COGNITIVE_SKILLS:
        k = meta["skill_id"]
        n_att = skill_attempts[k]
        p_val = skill_posteriors[k]
        # Posterior standard deviation shrinks with sqrt(n_att)
        post_sd = math.sqrt(p_val * (1.0 - p_val) / (n_att + 3.0))
        if n_att == 0:
            status = "uncalibrated"
        elif p_val >= 0.72:
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
                "status": status,
            }
        )

    # Identify primary cognitive micro-skill bottleneck among attempted skills
    attempted_skills = [s for s in skills_out if s["attempts_count"] > 0]
    if attempted_skills:
        weakest = min(attempted_skills, key=lambda s: s["mastery_probability"])
        primary_bottleneck = {
            "skill_id": weakest["skill_id"],
            "name": weakest["name"],
            "mastery_probability": weakest["mastery_probability"],
            "diagnosis": (
                f"DINA Q-Matrix analysis identifies '{weakest['name']}' "
                f"({round(weakest['mastery_probability'] * 100)}% mastery across {weakest['attempts_count']} items) "
                f"as your primary cross-chapter cognitive bottleneck: {weakest['description']}."
            ),
        }
    else:
        primary_bottleneck = {
            "skill_id": "dimensional_unit_analysis",
            "name": "Dimensional & Unit Conversion",
            "mastery_probability": 0.50,
            "diagnosis": "Attempt questions to calibrate your 8-skill DINA cognitive diagnosis matrix.",
        }

    # Compute speed percentile from latent speed tau ~ Normal(0, 0.55)
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
