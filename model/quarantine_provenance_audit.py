from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
QUARANTINE_PATH = PROJECT_ROOT / "data" / "quarantine" / "neet_unverified_augmented.json"
CLEAN_QUESTIONS_PATH = PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json"

VALID_SUBJECTS = {"Physics", "Chemistry", "Biology", "Botany", "Zoology"}
VALID_OPTIONS = {"A", "B", "C", "D", "a", "b", "c", "d"}


def audit_quarantine_expansion_readiness(quarantine_path: Path | None = None) -> dict[str, Any]:
    target = quarantine_path or QUARANTINE_PATH
    if not target.exists():
        return {
            "status": "ERROR",
            "detail": f"Quarantine file not found at {target}",
            "total_quarantined": 0,
            "ready_for_sme_review": 0,
            "structurally_defective": 0,
            "sme_candidates": [],
            "defective_items": [],
        }

    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "ERROR",
            "detail": f"Failed to parse quarantine JSON: {exc}",
            "total_quarantined": 0,
            "ready_for_sme_review": 0,
            "structurally_defective": 0,
            "sme_candidates": [],
            "defective_items": [],
        }

    sme_candidates = []
    defective_items = []
    subject_counts: dict[str, int] = {}
    difficulty_counts: dict[str, int] = {}
    chapter_counts: dict[str, int] = {}

    for item in data:
        qid = str(item.get("question_id", ""))
        subj = str(item.get("subject", "")).strip()
        ch = str(item.get("chapter", "")).strip()
        diff = str(item.get("difficulty_level", "")).strip().lower()
        q_text = str(item.get("question_text", "")).strip()
        opt_a = str(item.get("option_a", "")).strip()
        opt_b = str(item.get("option_b", "")).strip()
        opt_c = str(item.get("option_c", "")).strip()
        opt_d = str(item.get("option_d", "")).strip()
        corr = str(item.get("correct_option", "")).strip()
        expl = str(item.get("explanation", "")).strip()

        defect_reasons = []
        if not qid:
            defect_reasons.append("MISSING_QUESTION_ID")
        if subj not in VALID_SUBJECTS:
            defect_reasons.append(f"INVALID_SUBJECT:{subj}")
        if not ch:
            defect_reasons.append("MISSING_CHAPTER")
        if len(q_text) < 10:
            defect_reasons.append("TRUNCATED_QUESTION_TEXT")
        if not (opt_a and opt_b and opt_c and opt_d):
            defect_reasons.append("MISSING_OPTIONS")
        distinct_opts = {opt_a.lower(), opt_b.lower(), opt_c.lower(), opt_d.lower()}
        if len(distinct_opts) < 4:
            defect_reasons.append("DUPLICATE_DISTRACTOR_OPTIONS")
        if corr not in VALID_OPTIONS:
            defect_reasons.append(f"INVALID_CORRECT_OPTION:{corr}")
        if len(expl) < 5:
            defect_reasons.append("TRUNCATED_EXPLANATION")

        if defect_reasons:
            defective_items.append({"question_id": qid, "reasons": defect_reasons})
        else:
            sme_candidates.append(qid)
            subject_counts[subj] = subject_counts.get(subj, 0) + 1
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
            chapter_counts[ch] = chapter_counts.get(ch, 0) + 1

    total = len(data)
    ready_count = len(sme_candidates)
    defect_count = len(defective_items)
    readiness_pct = round((ready_count / total * 100.0), 2) if total > 0 else 0.0

    return {
        "status": "SUCCESS",
        "total_quarantined_items": total,
        "ready_for_sme_review_count": ready_count,
        "structurally_defective_count": defect_count,
        "structural_readiness_rate_pct": readiness_pct,
        "subject_distribution": subject_counts,
        "difficulty_distribution": difficulty_counts,
        "unique_chapters_covered": len(chapter_counts),
        "provenance_policy": "QUARANTINED_ITEMS_REQUIRE_OFFICIAL_NTA_SESSION_SOURCE_OR_DOUBLE_BLIND_SME_SIGNOFF_BEFORE_MERGE",
        "safe_to_merge_unconditionally": False,
        "sme_candidates_preview": sme_candidates[:10],
        "defective_items": defective_items[:10],
    }


def get_quarantine_item_by_id(question_id: str, quarantine_path: Path | None = None) -> dict[str, Any] | None:
    target = quarantine_path or QUARANTINE_PATH
    if not target.exists():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        for item in data:
            if item.get("question_id") == question_id:
                return item
    except Exception:
        return None
    return None
