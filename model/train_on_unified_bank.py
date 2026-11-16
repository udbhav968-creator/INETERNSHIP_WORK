import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import train_psychometric_and_rag_models as tp
from .irt_3pl_model import CALIBRATED_ITEMS
from .sample_paper_ingestion import load_accepted_sample_rows

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
PYQ_PATH = PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json"
REPORT_PATH = MODEL_DIR / "unified_training_report.json"

UNIFIED_PATHS = {
    "BKT_CHAPTER_JSON": MODEL_DIR / "bkt_chapter_parameters_unified.json",
    "DINA_CALIBRATED_JSON": MODEL_DIR / "dina_calibrated_parameters_unified.json",
    "RAG_SVD_NPZ": MODEL_DIR / "ncert_rag_svd_index_unified.npz",
    "RAG_SVD_META_JSON": MODEL_DIR / "ncert_rag_svd_meta_unified.json",
    "LINUCB_PRIOR_JSON": MODEL_DIR / "linucb_prior_init_unified.json",
    "CONFORMAL_LAPLACIAN_JSON": MODEL_DIR / "conformal_laplacian_model_unified.json",
}
PRODUCTION_PATHS = {k: getattr(tp, k) for k in UNIFIED_PATHS}
SEED = 20261116


def unified_rows_and_params() -> tuple[list[dict], dict[str, dict], int]:
    pyq = json.loads(PYQ_PATH.read_text(encoding="utf-8"))
    sample, params = load_accepted_sample_rows()
    items = {k: dict(v) for k, v in CALIBRATED_ITEMS.items()}
    for qid, p in params.items():
        items[qid] = {"a": p["a"], "b": p["b"], "c": p["c"]}
    rows = pyq + sample
    random.Random(SEED).shuffle(rows)
    return rows, items, len(sample)


def _rag_chapter_accuracy(npz_path: Path, meta_path: Path, questions: list[dict]) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    raw_vocab = meta["vocab"]
    vocab = {t: int(i) for t, i in raw_vocab.items()} if isinstance(raw_vocab, dict) else {t: i for i, t in enumerate(raw_vocab)}
    kb_raw = json.loads(tp.NCERT_KB_PATH.read_text(encoding="utf-8"))
    chunks = kb_raw.get("chunks", kb_raw) if isinstance(kb_raw, dict) else kb_raw
    return tp.evaluate_rag_chapter_retrieval(
        data["idf"], data["tfidf_matrix"], data["svd_vt"], data["doc_lsa"], vocab, [c["chapter"] for c in chunks], questions
    )


def _set_paths(paths: dict[str, Path]) -> None:
    for k, v in paths.items():
        setattr(tp, k, v)


def train_unified(save_report: bool = True) -> dict:
    rows, items, n_sample = unified_rows_and_params()
    topic_imp = json.loads(tp.IMPORTANCE_PATH.read_text(encoding="utf-8")) if tp.IMPORTANCE_PATH.exists() else {}
    sample_papers = sorted({str(q.get("paper_source")) for q in rows if q.get("source_type") == "sample_paper"})
    heldout_papers = set(random.Random(SEED + 1).sample(sample_papers, max(1, len(sample_papers) // 5)))
    rag_eval = [q for q in rows if q.get("source_type") == "sample_paper" and str(q.get("paper_source")) in heldout_papers]
    rag_index_rows = [q for q in rows if not (q.get("source_type") == "sample_paper" and str(q.get("paper_source")) in heldout_papers)]

    _set_paths(UNIFIED_PATHS)
    try:
        bkt = tp.train_chapter_wise_bkt_parameters(rows, items, topic_imp)
        dina = tp.train_dina_8skill_q_matrix_parameters(rows, items)
        rag = tp.train_ncert_and_pyq_svd_rag_index(rag_index_rows, heldout_eval_rows=rag_eval)
        bandit = tp.train_linucb_warm_start_and_conformal_laplacian(rows, items, bkt, topic_imp)
    finally:
        _set_paths(PRODUCTION_PATHS)

    prod_bkt = json.loads(PRODUCTION_PATHS["BKT_CHAPTER_JSON"].read_text(encoding="utf-8"))
    prod_dina = json.loads(PRODUCTION_PATHS["DINA_CALIBRATED_JSON"].read_text(encoding="utf-8"))
    bkt_delta = [
        abs(bkt["chapters"][ch]["p_init"] - prod_bkt["chapters"][ch]["p_init"])
        for ch in bkt["chapters"] if ch in prod_bkt["chapters"]
    ]
    items_per_chapter = defaultdict(lambda: [0, 0])
    for q in rows:
        items_per_chapter[q["chapter"]][0 if q.get("source_type", "pyq") == "pyq" else 1] += 1

    report = {
        "training_corpus": {"total_items": len(rows), "pyq": len(rows) - n_sample, "sample_paper": n_sample},
        "bkt_chapter_priors": {
            "chapters": bkt["total_chapters"],
            "weighted_global_prior_production": prod_bkt["weighted_global_prior"],
            "weighted_global_prior_unified": bkt["weighted_global_prior"],
            "mean_abs_p_init_change": round(float(np.mean(bkt_delta)), 4),
            "max_abs_p_init_change": round(float(np.max(bkt_delta)), 4),
            "min_items_per_chapter_production": min(v[0] for v in items_per_chapter.values()),
            "min_items_per_chapter_unified": min(sum(v) for v in items_per_chapter.values()),
        },
        "dina_8_skill_priors": {
            "items_production": prod_dina["total_corpus_items"],
            "items_unified": dina["total_corpus_items"],
            "items_per_skill_unified": {k: v.get("corpus_pyq_count") for k, v in dina["skills"].items()},
            "prior_mastery_production": {k: v["prior_mastery"] for k, v in prod_dina["skills"].items()},
            "prior_mastery_unified": {k: v["prior_mastery"] for k, v in dina["skills"].items()},
        },
        "ncert_rag_index": {
            "unified_metrics": rag,
            "heldout_papers": len(heldout_papers),
            "heldout_chapter_retrieval_production": _rag_chapter_accuracy(
                PRODUCTION_PATHS["RAG_SVD_NPZ"], PRODUCTION_PATHS["RAG_SVD_META_JSON"], rag_eval
            ),
            "heldout_chapter_retrieval_unified": rag["heldout_chapter_retrieval"],
            "unified_index_excludes": "every question from the held-out sample papers",
        },
        "linucb_and_laplacian": bandit,
        "artifacts": {k: v.name for k, v in UNIFIED_PATHS.items()},
        "status": "trained side-by-side; production artifacts unchanged until promoted",
        "not_retrained": {
            "irt_cold_start_ridge": "sample-paper IRT parameters were produced by this ridge; retraining on them would be circular",
            "topic_importance": "sample papers carry no exam year",
            "student_response_models": "0 student attempts exist",
        },
    }
    if save_report:
        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    r = train_unified()
    print(json.dumps({k: r[k] for k in ("training_corpus", "bkt_chapter_priors")}, indent=2))
    print(json.dumps({k: v for k, v in r["ncert_rag_index"].items() if k != "unified_metrics"}, indent=2))
    print({k: (r["dina_8_skill_priors"][k]) for k in ("items_production", "items_unified")})
