from __future__ import annotations

import math
from typing import Any
import numpy as np
from model.knowledge_graph_dag import DAG_NODES

class StreamingOnlineCalibrationEngine:
    def __init__(
        self,
        base_learning_rate: float = 0.05,
        decay_gamma: float = 0.001,
        transfer_lambda: float = 0.08,
    ) -> None:
        self.eta_0 = base_learning_rate
        self.gamma = decay_gamma
        self.lam = transfer_lambda
        self.step_counts: dict[str, int] = {}
        self.topic_correlations: dict[tuple[str, str], float] = {}
        self.online_elbo_history: list[float] = []

    def update_item_parameters_online(
        self,
        question_id: str,
        student_theta: float,
        is_correct: bool,
        current_a: float,
        current_b: float,
        current_c: float = 0.20,
    ) -> tuple[float, float, float]:
        t = self.step_counts.get(question_id, 0) + 1
        self.step_counts[question_id] = t
        eta_t = self.eta_0 / (1.0 + self.gamma * t)

        logit = 1.702 * current_a * (student_theta - current_b)
        p_know = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, logit))))
        p_pred = current_c + (1.0 - current_c) * p_know

        y = 1.0 if is_correct else 0.0
        residual = y - p_pred

        delta_b = -eta_t * residual
        delta_a = eta_t * residual * 1.702 * (student_theta - current_b) * p_know * (1.0 - p_know) / max(0.001, p_pred)

        new_b = max(-2.5, min(3.0, current_b + delta_b))
        new_a = max(0.8, min(2.5, current_a + delta_a))
        new_c = current_c

        return (round(new_a, 4), round(new_b, 4), round(new_c, 4))

    def update_graph_edge_transfer(
        self,
        parent_topic: str,
        child_topic: str,
        parent_mastery: float,
        child_success: bool,
    ) -> float:
        pair_key = (parent_topic, child_topic)
        prior_weight = self.topic_correlations.get(pair_key, 0.38)
        observed_transfer = 1.0 if (parent_mastery >= 0.70 and child_success) else (0.1 if (parent_mastery >= 0.70 and not child_success) else 0.35)
        new_weight = (1.0 - self.lam) * prior_weight + self.lam * observed_transfer
        clamped_weight = max(0.05, min(0.95, round(new_weight, 4)))
        self.topic_correlations[pair_key] = clamped_weight
        return clamped_weight

    def record_online_elbo(self, likelihoods: list[float], priors: list[float]) -> float:
        log_l = sum(math.log(max(1e-9, l)) for l in likelihoods)
        log_p = sum(math.log(max(1e-9, p)) for p in priors)
        elbo = log_l + log_p
        self.online_elbo_history.append(round(elbo, 4))
        if len(self.online_elbo_history) > 100:
            self.online_elbo_history.pop(0)
        return round(elbo, 4)

    def get_online_summary(self) -> dict[str, Any]:
        return {
            "total_items_updated": len(self.step_counts),
            "total_edge_transfers_monitored": len(self.topic_correlations),
            "recent_mean_elbo": round(float(np.mean(self.online_elbo_history)), 4) if self.online_elbo_history else 0.0,
            "monitored_edges": [
                {"parent": p, "child": c, "transfer_weight": w}
                for (p, c), w in list(self.topic_correlations.items())[:10]
            ],
        }


    def update_batch_online(
        self,
        attempts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        updated_items = 0
        mean_abs_residual = 0.0
        for att in attempts:
            qid = str(att.get("question_id", ""))
            theta = float(att.get("student_theta", 0.0))
            is_corr = bool(att.get("is_correct", False))
            curr_a = float(att.get("current_a", 1.25))
            curr_b = float(att.get("current_b", 0.0))
            curr_c = float(att.get("current_c", 0.20))
            new_a, new_b, _ = self.update_item_parameters_online(
                qid, theta, is_corr, curr_a, curr_b, curr_c
            )
            updated_items += 1
            logit = max(-15.0, min(15.0, 1.702 * curr_a * (theta - curr_b)))
            p_pred = curr_c + (1.0 - curr_c) / (1.0 + math.exp(-logit))
            mean_abs_residual += abs((1.0 if is_corr else 0.0) - p_pred)
        return {
            "batch_size": len(attempts),
            "updated_items": updated_items,
            "mean_batch_residual": round(mean_abs_residual / max(1, updated_items), 4),
        }


STREAMING_CALIBRATION_ENGINE = StreamingOnlineCalibrationEngine()
