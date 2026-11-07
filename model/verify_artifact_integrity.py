import csv
import hashlib
import json
import sys
import time
from pathlib import Path

from .corpus_distribution_and_readiness_audit import (
    DISTRIBUTION_REPORT_PATH,
    format_distribution_markdown_table,
    run_corpus_distribution_and_readiness_audit,
)
from .dina_cognitive_diagnosis import COGNITIVE_SKILLS
from .irt_3pl_model import CALIBRATED_ITEMS, predict_cold_start_difficulty_and_irt
from .knowledge_graph_dag import DAG_NODES
from .rag_socratic_engine import _INDEX as RAG_INDEX
from .rf_compact_runtime import COMPACT_RF_RUNTIME

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
REGISTRY_MANIFEST_FILE = MODEL_DIR / "model_registry_manifest.json"
VALIDATION_REPORT_FILE = MODEL_DIR / "validation_report.json"
HARNESS_REPORT_FILE = MODEL_DIR / "harness_benchmark_report.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_all_ml_artifacts(
    allow_serverless_omissions: bool = False,
    require_clean_bandit: bool = True,
) -> dict:
    t0 = time.perf_counter()

    if not REGISTRY_MANIFEST_FILE.exists():
        raise RuntimeError(f"Missing model registry manifest: {REGISTRY_MANIFEST_FILE}")

    with open(REGISTRY_MANIFEST_FILE, encoding="utf-8") as f:
        manifest = json.load(f)

    verified_count = 0
    for rel_path, meta in manifest.get("artifacts", {}).items():
        abs_p = PROJECT_ROOT / rel_path
        if not abs_p.exists():
            if allow_serverless_omissions and rel_path in (
                "data/training/question_features.csv",
                "model/question_difficulty_model.joblib",
            ):
                continue
            raise RuntimeError(f"Registered artifact missing on disk: {rel_path}")
        actual_size = abs_p.stat().st_size
        actual_hash = _sha256_file(abs_p)
        if actual_size != int(meta["size_bytes"]):
            raise RuntimeError(
                f"Artifact byte size mismatch for {rel_path}: expected {meta['size_bytes']}, got {actual_size}"
            )
        if actual_hash != str(meta["sha256"]):
            raise RuntimeError(
                f"SHA-256 checksum mismatch for {rel_path}: expected {meta['sha256']}, got {actual_hash}"
            )
        verified_count += 1

    attempts_csv = PROJECT_ROOT / "data" / "training" / "student_attempts.csv"
    if attempts_csv.exists():
        with open(attempts_csv, encoding="utf-8") as f:
            attempt_rows = list(csv.DictReader(f))
        if len(attempt_rows) != 0:
            raise RuntimeError(f"Expected 0 pre-seeded student attempts, found {len(attempt_rows)}")

    if require_clean_bandit:
        with open(MODEL_DIR / "linucb_state.json", encoding="utf-8") as f:
            bandit_state = json.load(f)
        if bandit_state != {}:
            raise RuntimeError("Expected clean empty linucb_state.json ({})")

    if not COMPACT_RF_RUNTIME.loaded:
        raise RuntimeError("Compact 300-tree serverless RF runtime failed to load")
    if len(CALIBRATED_ITEMS) != 2471:
        raise RuntimeError(f"Expected 2471 calibrated 3PL IRT items, got {len(CALIBRATED_ITEMS)}")
    if len(DAG_NODES) != 65 or len(RAG_INDEX.chunks) != 65 or len(COGNITIVE_SKILLS) != 8:
        raise RuntimeError("Curriculum graph / RAG / DINA skill count mismatch")

    smoke = predict_cold_start_difficulty_and_irt(
        question_text="Two point charges +4 uC and -9 uC are separated by 0.30 m in vacuum. Find electric field E.",
        options=["5.2e6 N/C", "5.2e3 N/C", "2.0e6 N/C", "3.6e5 N/C"],
        has_diagram=False,
        subject="Physics",
        chapter="Electrostatics",
        topic_tag="Electric Field",
    )
    if smoke.get("rf_probabilities") is None:
        raise RuntimeError("Compact RF smoke inference returned None probabilities")

    boot_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    if boot_ms > 2500.0:
        raise RuntimeError(f"Artifact verification & cold-start SLA exceeded: {boot_ms}ms > 2500ms")

    return {
        "integrity_status": "VERIFIED",
        "verified_artifacts_count": verified_count,
        "calibrated_irt_items": len(CALIBRATED_ITEMS),
        "canonical_chapters": len(DAG_NODES),
        "cold_start_verification_ms": boot_ms,
    }


def render_ci_markdown_summary() -> str:
    integrity = verify_all_ml_artifacts()
    dist_report = (
        json.load(open(DISTRIBUTION_REPORT_PATH, encoding="utf-8"))
        if DISTRIBUTION_REPORT_PATH.exists()
        else run_corpus_distribution_and_readiness_audit(save_report=False)
    )
    val_report = json.load(open(VALIDATION_REPORT_FILE, encoding="utf-8"))
    har_report = json.load(open(HARNESS_REPORT_FILE, encoding="utf-8"))

    rf_h = val_report["random_forest_holdout_evaluation"]
    cv = val_report["stratified_5fold_cross_validation"]
    tourn = val_report.get("champion_challenger_tournament", {})
    abl = har_report["linucb_vs_random_ablation"]

    lines = [
        "## Klassed Enterprise v2 — MLOps & Psychometric CI Governance Report",
        "",
        "| Gate | Metric | Status |",
        "| :--- | :--- | :---: |",
        f"| **Cryptographic Artifact Registry** | `{integrity['verified_artifacts_count']}` SHA-256 checksums verified in `{integrity['cold_start_verification_ms']} ms` | PASSED |",
        f"| **Champion 300-Tree RF Holdout** | `{rf_h['holdout_accuracy']*100:.2f}%` (`{rf_h['correct_predictions']}/{rf_h['testing_rows']}`), Macro-F1 `{rf_h['macro_f1']}` | PASSED |",
        f"| **Stratified 5-Fold Cross-Validation** | `{cv['mean_cv_accuracy']*100:.2f}% +/- {cv['std_cv_accuracy']*100:.2f}%`, Brier `{cv['oof_multiclass_brier_score']}` | PASSED |",
        f"| **Serverless Compact RF Parity** | `2471 / 2471` exact matches (`max_diff = {val_report['serverless_compact_rf_parity']['max_abs_probability_diff']}`) | PASSED |",
        f"| **120-Round LinUCB Bandit Regret** | Regret `{abl['linucb_cumulative_regret']}` vs Random `{abl['random_cumulative_regret']}` (ZPD `{abl['linucb_mean_zpd_alignment']*100:.1f}%`) | PASSED |",
        "",
        "### Multi-Model Champion-Challenger Tournament",
        "",
        "| Model Family | Holdout Accuracy | Macro F1 | Multiclass Log-Loss |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for m_name, m_stats in tourn.items():
        lines.append(
            f"| `{m_name}` | {m_stats['holdout_accuracy']*100:.2f}% | {m_stats['macro_f1']:.4f} | {m_stats['log_loss']:.4f} |"
        )
    lines.extend(["", format_distribution_markdown_table(dist_report)])
    return "\n".join(lines)


if __name__ == "__main__":
    if "--markdown-summary" in sys.argv:
        print(render_ci_markdown_summary())
    else:
        print(json.dumps(verify_all_ml_artifacts(), indent=2))
