from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from model.knowledge_graph_dag import DAG_NODES


def _allocate_180_neet_questions(topic_importance: dict[str, float]) -> list[tuple[str, str, int]]:
    subject_quotas = {"Physics": 45, "Chemistry": 45, "Biology": 90}
    by_subject: dict[str, list[tuple[str, float]]] = {"Physics": [], "Chemistry": [], "Biology": []}

    for topic_id, meta in DAG_NODES.items():
        subj = str(meta.get("subject", "Biology"))
        if subj not in by_subject:
            subj = "Biology"
        w = max(0.2, float(topic_importance.get(topic_id, 1.0)))
        by_subject[subj].append((topic_id, w))

    allocation: list[tuple[str, str, int]] = []
    for subj, target_q in subject_quotas.items():
        items = by_subject[subj]
        total_w = sum(w for _, w in items) or 1.0
        exact = [(t, (w / total_w) * target_q) for t, w in items]
        floors = {t: max(1, int(math.floor(val))) for t, val in exact}
        current_sum = sum(floors.values())

        remainders = sorted(exact, key=lambda x: x[1] - math.floor(x[1]), reverse=True)
        idx = 0
        while current_sum < target_q:
            t_name = remainders[idx % len(remainders)][0]
            floors[t_name] += 1
            current_sum += 1
            idx += 1

        by_smallest = sorted(exact, key=lambda x: x[1])
        idx = 0
        while current_sum > target_q and idx < len(by_smallest) * 2:
            t_name = by_smallest[idx % len(by_smallest)][0]
            if floors[t_name] > 1:
                floors[t_name] -= 1
                current_sum -= 1
            idx += 1

        for t_name, _ in items:
            allocation.append((t_name, subj, floors[t_name]))

    return allocation


def compute_graph_laplacian_leverage(
    mastery_map: dict[str, float],
    theta_map: dict[str, float],
    topic_importance: dict[str, float],
    global_theta: float = 0.0,
) -> list[dict[str, Any]]:
    topics = list(DAG_NODES.keys())
    n = len(topics)
    idx_map = {t: i for i, t in enumerate(topics)}

    adj = np.zeros((n, n), dtype=np.float64)
    for j_topic, meta in DAG_NODES.items():
        j = idx_map[j_topic]
        for prereq in meta.get("prerequisites", []):
            if prereq in idx_map:
                i = idx_map[prereq]
                adj[i, j] = 1.0

    beta = 0.38
    adj_2hop = adj @ adj
    heat_kernel = np.eye(n) + beta * adj + 0.5 * (beta ** 2) * adj_2hop

    allocation = _allocate_180_neet_questions(topic_importance)
    q_counts = np.zeros(n, dtype=np.float64)
    for t_name, _, q_cnt in allocation:
        if t_name in idx_map:
            q_counts[idx_map[t_name]] = float(q_cnt)

    direct_headroom_marks = np.zeros(n, dtype=np.float64)
    current_masteries = np.zeros(n, dtype=np.float64)

    for i, t_name in enumerate(topics):
        m_c = float(mastery_map.get(t_name, 0.35))
        th_c = float(theta_map.get(t_name, global_theta))
        current_masteries[i] = m_c

        p_curr = 0.22 + 0.78 / (1.0 + math.exp(-1.702 * 1.15 * (th_c - 0.10)))
        p_target = max(p_curr, 0.89)

        ev_curr = 5.0 * p_curr - 1.0
        ev_target = 5.0 * p_target - 1.0
        direct_headroom_marks[i] = max(0.0, q_counts[i] * (ev_target - ev_curr))

    bottleneck_factor = np.clip(0.85 - current_masteries, 0.05, 0.80)
    total_diffused_marks = np.zeros(n, dtype=np.float64)
    downstream_marks = np.zeros(n, dtype=np.float64)

    for i in range(n):
        direct_i = direct_headroom_marks[i]
        down_i = 0.0
        for j in range(n):
            if i != j and heat_kernel[i, j] > 0:
                down_i += heat_kernel[i, j] * direct_headroom_marks[j] * bottleneck_factor[i] * 0.45
        downstream_marks[i] = down_i
        total_diffused_marks[i] = direct_i + down_i

    ranked_indices = np.argsort(total_diffused_marks)[::-1]
    leverage_list: list[dict[str, Any]] = []
    for idx in ranked_indices[:6]:
        t_name = topics[int(idx)]
        meta = DAG_NODES[t_name]
        down_dependents = [
            topics[j] for j in range(n) if j != int(idx) and adj[int(idx), j] > 0
        ]
        leverage_list.append(
            {
                "topic_id": t_name,
                "subject": meta.get("subject", "Science"),
                "cluster": meta.get("cluster", "General"),
                "current_mastery": round(float(current_masteries[int(idx)]), 3),
                "direct_marks_gain": round(float(direct_headroom_marks[int(idx)]), 1),
                "downstream_diffused_marks_gain": round(float(downstream_marks[int(idx)]), 1),
                "total_expected_marks_gain": round(float(total_diffused_marks[int(idx)]), 1),
                "unlocks_downstream_count": len(down_dependents),
                "downstream_chapters": down_dependents[:4],
            }
        )

    return leverage_list


def run_monte_carlo_conformal_neet_simulation(
    student_id: str,
    global_theta: float,
    mastery_map: dict[str, float],
    theta_map: dict[str, float],
    ci_map: dict[str, float],
    topic_importance: dict[str, float],
    n_trials: int = 500,
) -> dict[str, Any]:
    allocation = _allocate_180_neet_questions(topic_importance)

    item_thetas = np.zeros(180, dtype=np.float64)
    item_ses = np.zeros(180, dtype=np.float64)
    item_masteries = np.zeros(180, dtype=np.float64)

    q_cursor = 0
    for topic_id, _, q_cnt in allocation:
        th_c = float(theta_map.get(topic_id, global_theta))
        m_c = float(mastery_map.get(topic_id, 0.35))
        ci_c = float(ci_map.get(topic_id, 0.38))
        se_c = max(0.12, min(0.55, ci_c * 0.75))

        end_cursor = min(180, q_cursor + q_cnt)
        item_thetas[q_cursor:end_cursor] = th_c
        item_ses[q_cursor:end_cursor] = se_c
        item_masteries[q_cursor:end_cursor] = m_c
        q_cursor = end_cursor
        if q_cursor >= 180:
            break

    state_fingerprint = f"{student_id}:{round(global_theta, 4)}:{len(theta_map)}:{round(sum(theta_map.values()), 4)}"
    seed_int = int(hashlib.sha256(state_fingerprint.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed_int)

    tier_b = np.concatenate(
        [
            np.linspace(-0.85, -0.25, 63),
            np.linspace(-0.20, 0.45, 81),
            np.linspace(0.50, 1.25, 36),
        ]
    )
    rng.shuffle(tier_b)
    item_a = rng.uniform(0.95, 1.45, size=180)
    item_c = rng.uniform(0.18, 0.24, size=180)

    sampled_thetas = rng.normal(
        loc=item_thetas[None, :],
        scale=item_ses[None, :],
        size=(n_trials, 180),
    )

    logits = 1.702 * item_a[None, :] * (sampled_thetas - tier_b[None, :])
    p_know = 1.0 / (1.0 + np.exp(-np.clip(logits, -15.0, 15.0)))
    p_3pl = item_c[None, :] + (1.0 - item_c[None, :]) * p_know

    trap_penalty = np.where(p_know < 0.26, 0.52 + 1.80 * p_know, 1.0)
    p_effective = np.clip(p_3pl * trap_penalty, 0.08, 0.995)

    u_draws = rng.random(size=(n_trials, 180))
    is_correct_mat = u_draws < p_effective

    marks_attempt_all = np.where(is_correct_mat, 4, -1)
    total_scores_attempt_all = np.clip(np.sum(marks_attempt_all, axis=1), 0, 720)

    should_skip_mask = p_effective < 0.23
    marks_optimal_skip = np.where(should_skip_mask, 0, marks_attempt_all)
    total_scores_optimal_skip = np.clip(np.sum(marks_optimal_skip, axis=1), 0, 720)

    alpha_90 = 0.10
    q_low_90 = max(0.01, math.floor((n_trials + 1) * (alpha_90 / 2.0)) / n_trials)
    q_high_90 = min(0.99, math.ceil((n_trials + 1) * (1.0 - alpha_90 / 2.0)) / n_trials)

    ci_90_low = int(round(float(np.quantile(total_scores_attempt_all, q_low_90))))
    ci_90_high = int(round(float(np.quantile(total_scores_attempt_all, q_high_90))))
    iqr_25 = int(round(float(np.quantile(total_scores_attempt_all, 0.25))))
    iqr_75 = int(round(float(np.quantile(total_scores_attempt_all, 0.75))))

    mean_score_all = int(round(float(np.mean(total_scores_attempt_all))))
    mean_score_skip = int(round(float(np.mean(total_scores_optimal_skip))))
    score_std = round(float(np.std(total_scores_attempt_all)), 1)

    marks_gained_by_skip = max(0, mean_score_skip - mean_score_all)
    avg_skip_count = int(round(float(np.mean(np.sum(should_skip_mask, axis=1)))))

    prob_qualifying_165 = round(float(np.mean(total_scores_optimal_skip >= 165) * 100.0), 1)
    prob_govt_550 = round(float(np.mean(total_scores_optimal_skip >= 550) * 100.0), 1)
    prob_aiims_650 = round(float(np.mean(total_scores_optimal_skip >= 650) * 100.0), 1)

    leverage_chapters = compute_graph_laplacian_leverage(
        mastery_map=mastery_map,
        theta_map=theta_map,
        topic_importance=topic_importance,
        global_theta=global_theta,
    )

    return {
        "model_type": "MonteCarlo_500_SplitConformal_and_GraphLaplacian_Diffusion",
        "n_trials": n_trials,
        "expected_score_attempt_all_720": mean_score_all,
        "conformal_90_interval": [ci_90_low, ci_90_high],
        "interquartile_50_interval": [iqr_25, iqr_75],
        "score_std_dev": score_std,
        "negative_marking_optimizer": {
            "ev_formula": "E[Marks] = 4*P(correct) - 1*(1 - P(correct)) = 5*P - 1",
            "optimal_skip_threshold_prob": 0.23,
            "optimal_skip_expected_score_720": mean_score_skip,
            "expected_marks_gained_by_skipping": marks_gained_by_skip,
            "recommended_skip_questions_per_180": avg_skip_count,
            "recommended_attempt_questions_per_180": 180 - avg_skip_count,
        },
        "cutoff_probabilities": {
            "qualifying_cutoff_165_percent": prob_qualifying_165,
            "state_govt_mbbs_550_percent": prob_govt_550,
            "top_aiims_650_percent": prob_aiims_650,
        },
        "high_yield_leverage_chapters": leverage_chapters,
    }
