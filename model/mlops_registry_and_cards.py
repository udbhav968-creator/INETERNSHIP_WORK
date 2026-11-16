import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
MODEL_CARDS_JSON = MODEL_DIR / "model_cards.json"
MODEL_CARDS_MD = PROJECT_ROOT / "docs" / "MODEL_CARDS.md"
PROMOTION_REPORT = MODEL_DIR / "promotion_gate_report.json"

DATASETS = {
    "pyq_bank": PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json",
    "pyq_features": PROJECT_ROOT / "data" / "training" / "question_features.csv",
    "sample_papers_verified": PROJECT_ROOT / "data" / "tagged_questions" / "neet_samples_001_to_101_verified.json",
    "ncert_knowledge_base": PROJECT_ROOT / "data" / "ncert_knowledge_base.json",
    "student_attempts": PROJECT_ROOT / "data" / "training" / "student_attempts.csv",
}

PROMOTION_MIN_MACRO_F1_GAIN = 0.02
PROMOTION_MIN_TOP1_GAIN = 0.02
PROMOTION_MAX_ECE = 0.05


def _load(name: str) -> dict:
    p = MODEL_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    data = path.read_bytes()
    if path.suffix.lower() in {".json", ".csv"}:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def _library_versions() -> dict:
    versions = {"python": platform.python_version()}
    for mod in ("numpy", "pandas", "sklearn", "scipy", "joblib"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None
    return versions


def _count_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.suffix == ".csv":
        return max(0, sum(1 for _ in path.open(encoding="utf-8")) - 1)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("chunks", data)
    return len(data)


def run_lineage() -> dict:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "libraries": _library_versions(),
        "datasets": {k: {"rows": _count_rows(p), "sha256": _sha256(p)} for k, p in DATASETS.items()},
    }


def _artifact(rel: str) -> dict:
    p = PROJECT_ROOT / rel
    return {"path": rel, "sha256": _sha256(p), "size_bytes": p.stat().st_size if p.exists() else None}


def build_model_cards() -> list[dict]:
    val = _load("validation_report.json")
    v21 = _load("difficulty_model_v21_report.json")
    leak = _load("difficulty_model_leakage_audit.json")
    rag = _load("ncert_rag_svd_meta.json")
    uni = _load("unified_training_report.json")
    harness = _load("harness_benchmark_report.json")
    aux = val.get("auxiliary_models_training", {})
    ridge = aux.get("irt_cold_start_ridge_regressor", {})
    hold = val.get("random_forest_holdout_evaluation", {})
    cv = val.get("stratified_5fold_cross_validation", {})
    cal = val.get("temperature_calibration_summary", {})
    tv = val.get("target_validity", {})
    clean = v21.get("near_duplicate_control", {})
    attempts = _count_rows(DATASETS["student_attempts"]) or 0

    return [
        {
            "model": "difficulty_rf_production",
            "status": "production",
            "task": "predict the PYQ difficulty tier",
            "artifacts": [_artifact("model/question_difficulty_model.joblib"), _artifact("model/rf_difficulty_forest.npz")],
            "training_data": "2,400 official PYQs, 11 non-leaking features",
            "labels": tv.get("label_source"),
            "evaluation": val.get("evaluation_protocol"),
            "metrics": {
                "holdout_accuracy": hold.get("holdout_accuracy"),
                "holdout_macro_f1": hold.get("macro_f1"),
                "cv_accuracy_train_split": cv.get("mean_cv_accuracy"),
                "oof_brier": cv.get("oof_multiclass_brier_score"),
                "holdout_ece": cal.get("post_ece"),
                "temperature_decision": cal.get("temperature_decision"),
            },
            "limitations": [
                tv.get("what_holdout_accuracy_measures"),
                f"heuristic vs human-verified tiers: kappa {((tv.get('heuristic_vs_human_verified_tiers') or {}).get('cohens_kappa'))}",
            ],
        },
        {
            "model": "difficulty_v21_text",
            "status": "candidate",
            "task": "predict the difficulty tier from question text",
            "artifacts": [_artifact("model/question_difficulty_model_v21_text_candidate.joblib")],
            "training_data": f"{v21.get('n_rows')} verified sample-paper questions from {v21.get('n_papers')} papers",
            "labels": v21.get("labels"),
            "evaluation": v21.get("evaluation"),
            "metrics": {
                "accuracy": (v21.get("v21_tfidf_logreg") or {}).get("accuracy"),
                "macro_f1": (v21.get("v21_tfidf_logreg") or {}).get("macro_f1"),
                "ece": (v21.get("v21_tfidf_logreg") or {}).get("ece_top_label"),
                "accuracy_duplicate_free_rows": (clean.get("v21_tfidf_logreg_on_clean_rows") or {}).get("accuracy"),
                "majority_baseline": v21.get("majority_baseline"),
            },
            "limitations": [
                "trained on sample papers; distribution may differ from real exam papers",
                "advanced tier is 8% of labels, so its recall is the weakest",
            ],
        },
        {
            "model": "difficulty_v2_structural",
            "status": "superseded_candidate",
            "task": "predict the difficulty tier from structural features",
            "artifacts": [_artifact("model/question_difficulty_model_v2_candidate.joblib")],
            "training_data": "verified sample-paper questions",
            "labels": "human-verified sample-paper tiers",
            "evaluation": "GroupKFold(5) by paper",
            "metrics": leak.get("sample_only_grouped_by_paper_5fold"),
            "limitations": ["superseded by v2.1"],
        },
        {
            "model": "ncert_rag_retriever",
            "status": "production",
            "task": "retrieve the NCERT chapter note for a question",
            "artifacts": [_artifact("model/ncert_rag_svd_index.npz"), _artifact("model/ncert_rag_svd_meta.json")],
            "training_data": rag.get("index_text"),
            "labels": "chapter of the question",
            "evaluation": (rag.get("heldout_chapter_retrieval") or {}).get("eval_set"),
            "metrics": rag.get("heldout_chapter_retrieval"),
            "limitations": ["lexical retrieval; no dense embeddings", "chapter retrieval is a proxy for answer usefulness"],
        },
        {
            "model": "ncert_rag_retriever_unified",
            "status": "candidate",
            "task": "same retriever indexed on PYQ + sample papers",
            "artifacts": [_artifact("model/ncert_rag_svd_index_unified.npz")],
            "training_data": (uni.get("ncert_rag_index") or {}).get("unified_index_excludes"),
            "labels": "chapter of the question",
            "evaluation": "held-out sample papers",
            "metrics": {
                "unified": (uni.get("ncert_rag_index") or {}).get("heldout_chapter_retrieval_unified"),
                "production_on_same_set": (uni.get("ncert_rag_index") or {}).get("heldout_chapter_retrieval_production"),
            },
            "limitations": ["small gain over production"],
        },
        {
            "model": "irt_cold_start_ridge",
            "status": "production_prior",
            "task": "predict 3PL a/b/c for a new question",
            "artifacts": [_artifact("model/irt_cold_start_ridge.npz")],
            "training_data": "2,400 PYQs",
            "labels": "a/b/c priors derived from tier and metadata, not fitted to responses",
            "evaluation": "5-fold CV",
            "metrics": {k: ridge.get(k) for k in ("a_5fold_r2", "b_5fold_r2", "c_5fold_r2")},
            "limitations": ["high R2 reflects that targets are a function of the inputs (tier, time)"],
        },
        {
            "model": "bkt_dina_linucb_priors",
            "status": "production_prior",
            "task": "student-state priors and item selection",
            "artifacts": [
                _artifact("model/bkt_chapter_parameters.json"),
                _artifact("model/dina_calibrated_parameters.json"),
                _artifact("model/linucb_prior_init.json"),
            ],
            "training_data": "question metadata only",
            "labels": "none (no student responses)",
            "evaluation": "simulation",
            "metrics": {
                "student_attempts_available": attempts,
                "linucb_cold_start_regret_reduction_vs_random_percent": (harness.get("linucb_cold_start_stochastic_ablation") or {}).get("regret_reduction_vs_random_percent"),
            },
            "limitations": ["cannot be validated until real attempts exist"],
        },
    ]


def evaluate_promotion_gates() -> dict:
    v21 = _load("difficulty_model_v21_report.json")
    uni = _load("unified_training_report.json")
    clean = v21.get("near_duplicate_control", {})
    v2c = clean.get("v2_structural_rf_on_clean_rows") or {}
    v21c = clean.get("v21_tfidf_logreg_on_clean_rows") or {}
    gain = (v21c.get("macro_f1") or 0) - (v2c.get("macro_f1") or 0)
    ece = (v21.get("v21_tfidf_logreg") or {}).get("ece_top_label")
    rag = uni.get("ncert_rag_index") or {}
    top1_gain = ((rag.get("heldout_chapter_retrieval_unified") or {}).get("top1_chapter_accuracy") or 0) - (
        (rag.get("heldout_chapter_retrieval_production") or {}).get("top1_chapter_accuracy") or 0
    )
    decisions = [
        {
            "candidate": "difficulty_v21_text",
            "compared_with": "difficulty_v2_structural on the same duplicate-free held-out papers",
            "macro_f1_gain": round(gain, 4),
            "ece": ece,
            "gates": {
                f"macro_f1_gain >= {PROMOTION_MIN_MACRO_F1_GAIN}": gain >= PROMOTION_MIN_MACRO_F1_GAIN,
                f"ece <= {PROMOTION_MAX_ECE}": ece is not None and ece <= PROMOTION_MAX_ECE,
            },
        },
        {
            "candidate": "ncert_rag_retriever_unified",
            "compared_with": "ncert_rag_retriever on the same held-out sample papers",
            "top1_gain": round(top1_gain, 4),
            "gates": {f"top1_gain >= {PROMOTION_MIN_TOP1_GAIN}": top1_gain >= PROMOTION_MIN_TOP1_GAIN},
        },
    ]
    for d in decisions:
        d["decision"] = "ELIGIBLE_PENDING_HUMAN_REVIEW" if all(d["gates"].values()) else "HOLD"
    return {"policy": "a candidate is promoted only after every gate passes and a reviewer signs off", "decisions": decisions}


def _md_value(v) -> str:
    if isinstance(v, dict):
        return "; ".join(f"{k}: {_md_value(x)}" for k, x in v.items())
    if isinstance(v, list):
        return "; ".join(_md_value(x) for x in v)
    if isinstance(v, float):
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return "—" if v is None else str(v)


def render_markdown(cards: list[dict], gates: dict, lineage: dict) -> str:
    lines = ["# Model Cards", "", f"Generated {lineage['generated_at_utc']} from commit `{lineage['git_commit']}` by `python -m model.mlops_registry_and_cards`.", ""]
    lines += ["## Summary", "", "| Model | Status | Key metrics |", "|---|---|---|"]
    for c in cards:
        lines.append(f"| {c['model']} | {c['status']} | {_md_value(c['metrics'])} |")
    for c in cards:
        lines += ["", f"## {c['model']}", "", "| Field | Value |", "|---|---|"]
        for key in ("status", "task", "training_data", "labels", "evaluation", "metrics", "limitations"):
            lines.append(f"| {key} | {_md_value(c.get(key))} |")
        lines.append(f"| artifacts | {'; '.join(a['path'] + ' (' + str(a['sha256'])[:12] + ')' for a in c['artifacts'])} |")
    lines += ["", "## Promotion gates", "", f"Policy: {gates['policy']}", "", "| Candidate | Compared with | Gates | Decision |", "|---|---|---|---|"]
    for d in gates["decisions"]:
        lines.append(f"| {d['candidate']} | {d['compared_with']} | {_md_value(d['gates'])} | {d['decision']} |")
    lines += ["", "## Lineage", "", "| Dataset | Rows | SHA-256 |", "|---|---|---|"]
    for k, v in lineage["datasets"].items():
        lines.append(f"| {k} | {v['rows']} | {str(v['sha256'])[:16]} |")
    lines += ["", f"Libraries: {_md_value(lineage['libraries'])}", ""]
    return "\n".join(lines)


def publish_model_cards() -> dict:
    cards = build_model_cards()
    gates = evaluate_promotion_gates()
    lineage = run_lineage()
    MODEL_CARDS_JSON.write_text(json.dumps({"lineage": lineage, "models": cards}, indent=2), encoding="utf-8")
    PROMOTION_REPORT.write_text(json.dumps(gates, indent=2), encoding="utf-8")
    MODEL_CARDS_MD.write_text(render_markdown(cards, gates, lineage), encoding="utf-8")
    return {"models": len(cards), "promotion_decisions": {d["candidate"]: d["decision"] for d in gates["decisions"]}}


if __name__ == "__main__":
    print(json.dumps(publish_model_cards(), indent=2))
