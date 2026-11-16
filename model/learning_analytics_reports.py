from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .analytics_data import (
    load_actual_scores,
    load_decisions,
    load_full_attempts,
    load_score_snapshots,
    load_student_attributes,
)
from .anomaly_detection import detect_anomalies
from .empirical_item_times import estimate_item_times, save_item_times
from .experiments import analyse_selection_experiment
from .fairness_report import run_fairness_report
from .kt_challenger import compare_knowledge_tracing
from .offline_policy_evaluation import evaluate_logged_decisions
from .score_validation import pair_predictions_with_results, validate_score_predictions
from .student_response_training import OUTPUT_DIR, _bank_metadata, _prior_bkt, _prior_irt

REPORT_PATH = OUTPUT_DIR / "learning_analytics_report.json"


def _fitted_irt() -> dict:
    path = OUTPUT_DIR / "irt_item_parameters_fitted.json"
    if not path.exists():
        return {}
    items = json.loads(path.read_text(encoding="utf-8")).get("items", {})
    return {str(q): (float(p["a"]), float(p["b"]), float(p["c"])) for q, p in items.items()}


def run_learning_analytics(url: str | None = None, save: bool = True, output_path: Path = REPORT_PATH) -> dict:
    attempts = load_full_attempts(url)
    if not attempts.empty:
        meta = _bank_metadata()
        attempts["chapter"] = attempts["question_id"].map(lambda q: (meta.get(q) or {}).get("chapter")).fillna(attempts.get("topic_id"))
    decisions = load_decisions(url)
    snapshots = load_score_snapshots(url)
    actuals = load_actual_scores()
    attributes = load_student_attributes(url=url)
    arms = {}
    for d in decisions:
        arms.setdefault(str(d["student_id"]), d.get("arm"))
    item_params = {**_prior_irt(), **_fitted_irt()}
    pairs = pair_predictions_with_results(snapshots, actuals)
    item_times = estimate_item_times(attempts)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": {
            "attempts": int(len(attempts)),
            "students": int(attempts["student_id"].nunique()) if not attempts.empty else 0,
            "selection_decisions": len(decisions),
            "score_snapshots": int(len(snapshots)),
            "actual_neet_scores": int(len(actuals)),
        },
        "offline_policy_evaluation": evaluate_logged_decisions(decisions),
        "selection_ab_test": analyse_selection_experiment(attempts, arms) if not attempts.empty else {"status": "INSUFFICIENT_DATA"},
        "score_prediction_validation": validate_score_predictions(snapshots, actuals),
        "fairness": run_fairness_report(attempts, attributes, _prior_bkt(), pairs),
        "empirical_item_times": {k: v for k, v in item_times.items() if k != "items"},
        "anomalies": detect_anomalies(attempts, item_params),
        "knowledge_tracing_challenger": compare_knowledge_tracing(attempts.dropna(subset=["chapter"]) if not attempts.empty else attempts),
    }
    report["sections_with_results"] = sorted(
        k for k, v in report.items() if isinstance(v, dict) and v.get("status") not in (None, "INSUFFICIENT_DATA")
    )
    if save:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        if item_times.get("status") == "ESTIMATED":
            save_item_times(item_times)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Learning analytics on logged student data")
    ap.add_argument("--database-url", default=None)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    rep = run_learning_analytics(args.database_url, save=not args.no_save)
    print(json.dumps({k: (v.get("status") if isinstance(v, dict) else v) for k, v in rep.items() if k != "data"}, indent=2))
    print(json.dumps(rep["data"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
