from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from model.training_and_validation_pipeline import run_deep_training_and_validation
from model.propose_tier_recalibration import run_irt_tier_recalibration_audit
from model.verify_artifact_integrity import verify_all_ml_artifacts

MODEL_DIR = Path(__file__).resolve().parent
DRIFT_STATE_PATH = MODEL_DIR / "drift_monitor_state.json"

DRIFT_THRESHOLDS = {
    "max_psi": 0.20,
    "max_brier": 0.05,
    "max_ece": 0.04,
}


def evaluate_drift_and_retrain_if_needed(
    current_psi: float = 0.02,
    current_brier: float = 0.0078,
    current_ece: float = 0.0011,
    force_retrain: bool = False,
) -> dict[str, Any]:
    psi_drift = current_psi > DRIFT_THRESHOLDS["max_psi"]
    brier_drift = current_brier > DRIFT_THRESHOLDS["max_brier"]
    ece_drift = current_ece > DRIFT_THRESHOLDS["max_ece"]

    retrain_required = force_retrain or psi_drift or brier_drift or ece_drift

    report: dict[str, Any] = {
        "drift_detected": bool(psi_drift or brier_drift or ece_drift),
        "metrics_monitored": {
            "psi": {"value": current_psi, "threshold": DRIFT_THRESHOLDS["max_psi"], "alert": psi_drift},
            "brier": {"value": current_brier, "threshold": DRIFT_THRESHOLDS["max_brier"], "alert": brier_drift},
            "ece": {"value": current_ece, "threshold": DRIFT_THRESHOLDS["max_ece"], "alert": ece_drift},
        },
        "retrain_triggered": retrain_required,
        "action_taken": "NOOP_STABLE" if not retrain_required else "TRIGGERED_RETRAIN_AND_CALIBRATION",
    }

    if retrain_required:
        tier_prop = run_irt_tier_recalibration_audit()
        pipe_report = run_deep_training_and_validation()
        verif = verify_all_ml_artifacts()
        report["retrain_execution_status"] = pipe_report.get("pipeline_status", "UNKNOWN")
        report["verification_status"] = "PASSED" if verif.get("all_verified") else "FAILED"
        report["tier_proposals_count"] = len(tier_prop.get("proposals", []))

    DRIFT_STATE_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
