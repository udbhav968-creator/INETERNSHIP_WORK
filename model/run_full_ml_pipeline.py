import argparse
import json
import sys
import time
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
RUN_REPORT = MODEL_DIR / "ml_pipeline_run.json"
REPRO_TOLERANCE = 0.005

REPRO_METRICS = {
    "validation_report.json": [
        ("random_forest_holdout_evaluation", "holdout_accuracy"),
        ("random_forest_holdout_evaluation", "macro_f1"),
        ("stratified_5fold_cross_validation", "mean_cv_accuracy"),
    ],
    "difficulty_model_v21_report.json": [
        ("v21_tfidf_logreg", "accuracy"),
        ("v21_tfidf_logreg", "macro_f1"),
    ],
    "ncert_rag_svd_meta.json": [
        ("heldout_chapter_retrieval", "top1_chapter_accuracy"),
        ("heldout_chapter_retrieval", "top3_chapter_accuracy"),
    ],
}


def _stage_data_gates() -> dict:
    from .data_integrity_gate import run_data_integrity_gate

    rep = run_data_integrity_gate()
    if rep.get("status") == "FAILED":
        raise RuntimeError(f"data integrity gate failed: {rep.get('blocking_failures')}")
    return {"status": rep.get("status"), "warnings": len(rep.get("warnings", []))}


def _stage_label_validity() -> dict:
    from .label_validity_audit import run_label_validity_audit

    rep = run_label_validity_audit()
    return {"heuristic_vs_human": rep["heuristic_rule_vs_human_verified_sample_tiers"]}


def _stage_production_training() -> dict:
    from .training_and_validation_pipeline import run_deep_training_and_validation

    rep = run_deep_training_and_validation(save_artifacts=True)
    return {"holdout": {k: rep["random_forest_holdout_evaluation"][k] for k in ("holdout_accuracy", "macro_f1", "testing_rows")}}


def _stage_difficulty_v21() -> dict:
    from .difficulty_model_v21_text import train_and_evaluate_v21

    rep = train_and_evaluate_v21(save=True)
    return {"v21": rep["v21_tfidf_logreg"], "decision": rep["decision"]["status"]}


def _stage_unified_side_by_side() -> dict:
    from .train_on_unified_bank import train_unified

    rep = train_unified(save_report=True)
    return {"rag": {k: v for k, v in rep["ncert_rag_index"].items() if k != "unified_metrics"}}


def _stage_harness() -> dict:
    from .evaluation_harness import run_evaluation_harness

    rep = run_evaluation_harness(save_report=True)
    return {"status": rep["harness_status"]}


def _stage_registries() -> dict:
    from .ml_extension_registry import build_extension_registry, verify_extension_registry
    from .training_and_validation_pipeline import REGISTRY_MANIFEST_FILE, build_model_registry_manifest
    from .verify_artifact_integrity import verify_all_ml_artifacts

    REGISTRY_MANIFEST_FILE.write_text(json.dumps(build_model_registry_manifest(), indent=2), encoding="utf-8")
    build_extension_registry()
    core = verify_all_ml_artifacts()
    ext = verify_extension_registry()
    if core.get("integrity_status") != "VERIFIED" or ext.get("status") != "VERIFIED":
        raise RuntimeError("artifact registry verification failed")
    return {"core": core.get("integrity_status"), "extension": ext}


def _stage_model_cards() -> dict:
    from .mlops_registry_and_cards import publish_model_cards

    return publish_model_cards()


STAGES = [
    ("data_gates", _stage_data_gates),
    ("label_validity", _stage_label_validity),
    ("production_training", _stage_production_training),
    ("difficulty_v21", _stage_difficulty_v21),
    ("unified_side_by_side", _stage_unified_side_by_side),
    ("evaluation_harness", _stage_harness),
    ("artifact_registries", _stage_registries),
    ("model_cards_and_promotion_gates", _stage_model_cards),
]


def _snapshot_metrics() -> dict:
    snap = {}
    for fname, keys in REPRO_METRICS.items():
        p = MODEL_DIR / fname
        if not p.exists():
            continue
        rep = json.loads(p.read_text(encoding="utf-8"))
        for section, key in keys:
            snap[f"{fname}:{section}.{key}"] = (rep.get(section) or {}).get(key)
    return snap


def run_pipeline(verify_reproducible: bool = False, only: list[str] | None = None) -> dict:
    before = _snapshot_metrics()
    stages = []
    t0 = time.perf_counter()
    for name, fn in STAGES:
        if only and name not in only:
            continue
        ts = time.perf_counter()
        try:
            out = fn()
            stages.append({"stage": name, "status": "PASSED", "seconds": round(time.perf_counter() - ts, 2), "output": out})
        except Exception as exc:
            stages.append({"stage": name, "status": "FAILED", "seconds": round(time.perf_counter() - ts, 2), "error": repr(exc)})
            break
    after = _snapshot_metrics()
    repro = None
    if verify_reproducible:
        diffs = {
            k: {"committed": before.get(k), "rerun": after.get(k)}
            for k in after
            if before.get(k) is None or after.get(k) is None or abs(float(before[k]) - float(after[k])) > REPRO_TOLERANCE
        }
        repro = {"tolerance": REPRO_TOLERANCE, "metrics_checked": len(after), "out_of_tolerance": diffs, "reproducible": not diffs}
    report = {
        "status": "PASSED" if all(s["status"] == "PASSED" for s in stages) and (repro is None or repro["reproducible"]) else "FAILED",
        "total_seconds": round(time.perf_counter() - t0, 2),
        "stages": stages,
        "reproducibility": repro,
        "metrics": after,
    }
    RUN_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-reproducible", action="store_true")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    rep = run_pipeline(verify_reproducible=args.verify_reproducible, only=args.only)
    print(json.dumps({k: rep[k] for k in ("status", "total_seconds", "reproducibility")}, indent=2))
    for s in rep["stages"]:
        print(f"[{s['status']}] {s['stage']} ({s['seconds']}s)")
    return 0 if rep["status"] == "PASSED" else 1


if __name__ == "__main__":
    sys.exit(main())
