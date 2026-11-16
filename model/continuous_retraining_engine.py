from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from model.training_and_validation_pipeline import run_deep_training_and_validation
from model.propose_tier_recalibration import run_irt_tier_recalibration_audit
from model.verify_artifact_integrity import verify_all_ml_artifacts

MODEL_DIR = Path(__file__).resolve().parent
DRIFT_STATE_PATH = MODEL_DIR / "drift_monitor_state.json"
VALIDATION_REPORT_PATH = MODEL_DIR / "validation_report.json"
MIN_LIVE_SAMPLE_SIZE = 30
PSI_ALERT_THRESHOLD = 0.20
RELATIVE_DEGRADATION_TOLERANCE = 0.25


def _load_validation_baseline() -> dict[str, float | None]:
    baseline: dict[str, float | None] = {"brier": None, "ece": None}
    if not VALIDATION_REPORT_PATH.exists():
        return baseline
    try:
        rep = json.loads(VALIDATION_REPORT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return baseline
    cv = rep.get("stratified_5fold_cross_validation", {}) or {}
    cal = rep.get("temperature_calibration_summary", {}) or {}
    if cv.get("oof_multiclass_brier_score") is not None:
        baseline["brier"] = float(cv["oof_multiclass_brier_score"])
    if cal.get("post_ece") is not None:
        baseline["ece"] = float(cal["post_ece"])
    return baseline


def _drift_thresholds() -> dict[str, float | None]:
    base = _load_validation_baseline()
    return {
        "max_psi": PSI_ALERT_THRESHOLD,
        "max_brier": None if base["brier"] is None else round(base["brier"] * (1.0 + RELATIVE_DEGRADATION_TOLERANCE), 5),
        "max_ece": None if base["ece"] is None else round(base["ece"] * (1.0 + RELATIVE_DEGRADATION_TOLERANCE), 5),
        "baseline_brier": base["brier"],
        "baseline_ece": base["ece"],
    }


DRIFT_THRESHOLDS = _drift_thresholds()


def _read_live_metrics() -> dict[str, Any]:
    from model.feature_store_and_drift import DRIFT_MONITOR
    health = DRIFT_MONITOR.evaluate_health()
    return {
        "psi": health.get("effective_psi"),
        "brier": health.get("brier_score"),
        "ece": health.get("expected_calibration_error"),
        "sample_size": int(health.get("sample_size", 0)),
    }


def _alert(value: float | None, threshold: float | None) -> bool:
    return value is not None and threshold is not None and float(value) > float(threshold)


def evaluate_drift_and_retrain_if_needed(
    current_psi: float | None = None,
    current_brier: float | None = None,
    current_ece: float | None = None,
    force_retrain: bool = False,
    sample_size: int | None = None,
) -> dict[str, Any]:
    explicit = any(v is not None for v in (current_psi, current_brier, current_ece))
    if explicit:
        metrics_source = "caller_supplied"
        n_obs = sample_size
    else:
        live = _read_live_metrics()
        metrics_source = "live_drift_monitor"
        n_obs = live["sample_size"]
        if n_obs > 0:
            current_psi, current_brier, current_ece = live["psi"], live["brier"], live["ece"]

    thresholds = _drift_thresholds()
    insufficient = n_obs is not None and n_obs < MIN_LIVE_SAMPLE_SIZE
    psi_drift = False if insufficient else _alert(current_psi, thresholds["max_psi"])
    brier_drift = False if insufficient else _alert(current_brier, thresholds["max_brier"])
    ece_drift = False if insufficient else _alert(current_ece, thresholds["max_ece"])
    drift = bool(psi_drift or brier_drift or ece_drift)
    retrain_required = bool(force_retrain or drift)

    if retrain_required:
        action = "TRIGGERED_RETRAIN_AND_CALIBRATION"
    elif insufficient:
        action = "INSUFFICIENT_DATA"
    else:
        action = "NOOP_STABLE"

    report: dict[str, Any] = {
        "drift_detected": drift,
        "metrics_source": metrics_source,
        "sample_size": n_obs,
        "min_sample_size": MIN_LIVE_SAMPLE_SIZE,
        "threshold_basis": f"validation_report baseline x (1 + {RELATIVE_DEGRADATION_TOLERANCE}); PSI {PSI_ALERT_THRESHOLD} is the conventional major-shift cutoff",
        "metrics_monitored": {
            "psi": {"value": current_psi, "threshold": thresholds["max_psi"], "alert": psi_drift},
            "brier": {"value": current_brier, "threshold": thresholds["max_brier"], "baseline": thresholds["baseline_brier"], "alert": brier_drift},
            "ece": {"value": current_ece, "threshold": thresholds["max_ece"], "baseline": thresholds["baseline_ece"], "alert": ece_drift},
        },
        "retrain_triggered": retrain_required,
        "action_taken": action,
    }

    if retrain_required:
        tier_prop = run_irt_tier_recalibration_audit()
        pipe_report = run_deep_training_and_validation()
        try:
            verif = verify_all_ml_artifacts()
            verified = verif.get("integrity_status") == "VERIFIED"
        except Exception as exc:
            verified = False
            report["verification_error"] = str(exc)
        report["retrain_execution_status"] = pipe_report.get("pipeline_status", "UNKNOWN")
        report["verification_status"] = "PASSED" if verified else "FAILED"
        report["tier_proposals_count"] = len(tier_prop.get("proposals", []))

    DRIFT_STATE_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
