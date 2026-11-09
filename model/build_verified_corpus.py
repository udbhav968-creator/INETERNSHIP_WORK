from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from .remediate_and_calibrate_corpus import (
    CLEAN_OUT_PATH,
    DAG_PATH,
    RAW_FLAWLESS_PATH,
    compute_item_feature_dict,
    write_derived_datasets,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUARANTINE_DIR = PROJECT_ROOT / "data" / "quarantine"
UNVERIFIED_OUT_PATH = QUARANTINE_DIR / "neet_unverified_augmented.json"
SUMMARY_PATH = PROJECT_ROOT / "data" / "taxonomy" / "chapter_wise_neet_pyq_summary.json"

OFFICIAL_SOURCE_RE = re.compile(r"^NEET \d{4}( Re-NEET)?$")
OFFICIAL_ID_RE = re.compile(r"^(RE)?NEET\d{4}_(PHY|CHEM|BIO)_\d{3}$")

DIFFICULTY_SOURCE = "chapter_text_complexity_quantile_heuristic"
TIME_SOURCE = "synthetic_tier_formula"


def is_official_pyq(q: dict[str, Any]) -> bool:
    return bool(
        OFFICIAL_SOURCE_RE.match(str(q.get("paper_source", "")).strip())
        and OFFICIAL_ID_RE.match(str(q.get("question_id", "")).strip())
    )


def build_verified_corpus() -> dict[str, Any]:
    questions = json.loads(CLEAN_OUT_PATH.read_text(encoding="utf-8"))
    raw_by_id = {q["question_id"]: q for q in json.loads(RAW_FLAWLESS_PATH.read_text(encoding="utf-8"))}

    verified: list[dict[str, Any]] = []
    unverified: list[dict[str, Any]] = []
    for q in questions:
        if is_official_pyq(q):
            verified.append(q)
        else:
            item = dict(q)
            item["quarantine_reason"] = "no_verifiable_official_exam_provenance"
            unverified.append(item)

    for q in verified:
        raw = raw_by_id.get(q["question_id"], {})
        q.setdefault("original_difficulty_level", raw.get("difficulty_level"))
        q.setdefault("original_ideal_time_sec", raw.get("ideal_time_sec"))
        q["difficulty_source"] = DIFFICULTY_SOURCE
        q["ideal_time_source"] = TIME_SOURCE

    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    if unverified or not UNVERIFIED_OUT_PATH.exists():
        UNVERIFIED_OUT_PATH.write_text(
            json.dumps(unverified, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    verified.sort(key=lambda x: x["question_id"])
    CLEAN_OUT_PATH.write_text(json.dumps(verified, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    dag_nodes = json.loads(DAG_PATH.read_text(encoding="utf-8"))["nodes"]
    valid_chapters = set(dag_nodes.keys())
    feats_by_id: dict[str, dict[str, Any]] = {}
    items_by_chapter: dict[str, list[dict[str, Any]]] = {ch: [] for ch in valid_chapters}
    for q in verified:
        fc = compute_item_feature_dict(q)
        fc["difficulty_level"] = q["difficulty_level"]
        feats_by_id[q["question_id"]] = fc
        items_by_chapter.setdefault(q["chapter"], []).append(q)

    prior = json.loads(SUMMARY_PATH.read_text(encoding="utf-8")) if SUMMARY_PATH.exists() else {}
    summary = write_derived_datasets(
        verified,
        feats_by_id,
        items_by_chapter,
        dag_nodes,
        valid_chapters,
        {
            "quarantined": int(prior.get("total_quarantined_items", 0)),
            "repaired_leaked": int(prior.get("repaired_leaked_statement_items", 0)),
            "explicit_fixed": int(prior.get("explicit_ocr_repaired_items", 0)),
            "reclassified_chapters": int(prior.get("reclassified_chapter_items", 0)),
            "exact_dups_removed": int(prior.get("exact_duplicates_removed", 0)),
            "tier_recalibrated_count": int(prior.get("tier_recalibrated_items", 0)),
        },
    )
    return {
        "verified_official_pyqs": len(verified),
        "quarantined_unverified": len(unverified),
        "unverified_path": str(UNVERIFIED_OUT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "summary_total_verified_pyqs": summary["total_verified_pyqs"],
    }


if __name__ == "__main__":
    print(json.dumps(build_verified_corpus(), indent=2))
    sys.exit(0)
