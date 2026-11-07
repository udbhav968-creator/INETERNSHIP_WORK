import json
import sys
from pathlib import Path

from .irt_3pl_model import CALIBRATED_ITEMS, get_item_params
from .knowledge_graph_dag import DAG_NODES

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
CLEAN_QUESTIONS_PATH = PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json"
RECALIBRATION_REPORT_PATH = MODEL_DIR / "tier_recalibration_proposal.json"


def _resolve_canonical_chapter(q: dict) -> str:
    ch = str(q.get("chapter") or q.get("topic_id") or "").strip()
    if ch in DAG_NODES:
        return ch
    ttag = str(q.get("topic_tag") or "").strip()
    if ttag in DAG_NODES:
        return ttag
    return ch


def run_irt_tier_recalibration_audit(save_report: bool = True) -> dict:
    with open(CLEAN_QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    topic_tier_items: dict[str, dict[str, list[dict]]] = {
        tid: {"basic": [], "medium": [], "advanced": []} for tid in DAG_NODES
    }

    strict_basic_candidates: list[dict] = []
    strict_advanced_candidates: list[dict] = []

    for q in questions:
        qid = str(q.get("question_id") or "")
        tid = _resolve_canonical_chapter(q)
        if tid not in topic_tier_items:
            continue
        subj = str(DAG_NODES[tid].get("subject") or q.get("subject") or "Physics")
        diff = str(q.get("difficulty") or q.get("difficulty_level") or "medium").lower()
        if diff not in ("basic", "medium", "advanced"):
            diff = "medium"

        ideal_t = int(q.get("ideal_time_sec") or q.get("expected_time_seconds") or 90)
        qtype = str(q.get("question_type") or "Single_Choice_MCQ")
        has_diag = bool(q.get("has_diagram", False))
        qtext = str(q.get("question_text") or q.get("text") or "")
        word_cnt = len(qtext.split())

        if qid in CALIBRATED_ITEMS:
            raw_p = CALIBRATED_ITEMS[qid]
            a_val = float(raw_p["a"] if isinstance(raw_p, dict) else raw_p.a)
            b_val = float(raw_p["b"] if isinstance(raw_p, dict) else raw_p.b)
            c_val = float(raw_p["c"] if isinstance(raw_p, dict) else raw_p.c)
        else:
            p = get_item_params(qid, diff, ideal_t)
            a_val, b_val, c_val = float(p.a), float(p.b), float(p.c)

        composite_ease = (-b_val) + (90.0 - float(ideal_t)) / 60.0 - (0.25 if has_diag else 0.0) - (word_cnt / 120.0)
        entry = {
            "question_id": qid,
            "topic_id": tid,
            "subject": subj,
            "current_tier": diff,
            "irt_a": round(a_val, 4),
            "irt_b": round(b_val, 4),
            "irt_c": round(c_val, 4),
            "ideal_time_sec": ideal_t,
            "word_count": word_cnt,
            "question_type": qtype,
            "has_diagram": has_diag,
            "foundational_score": round(composite_ease, 4),
        }
        topic_tier_items[tid][diff].append(entry)

        if (
            diff == "medium"
            and not has_diag
            and ideal_t <= 75
            and b_val <= 0.12
            and word_cnt <= 55
        ):
            strict_basic_candidates.append({**entry, "proposed_tier": "basic"})
        elif diff == "medium" and (b_val >= 0.72 or ideal_t >= 125 or (has_diag and b_val >= 0.45)):
            strict_advanced_candidates.append({**entry, "proposed_tier": "advanced"})

    missing_basic_chapters = [tid for tid, buckets in topic_tier_items.items() if len(buckets["basic"]) == 0]
    missing_advanced_chapters = [tid for tid, buckets in topic_tier_items.items() if len(buckets["advanced"]) == 0]

    basic_gap_resolutions: list[dict] = []
    resolved_basic_count = 0
    resolved_physics_basic = 0

    for tid in sorted(missing_basic_chapters):
        med_pool = sorted(
            topic_tier_items[tid]["medium"],
            key=lambda r: (-r["foundational_score"], r["irt_b"], r["ideal_time_sec"]),
        )
        anchors = med_pool[:3]
        if anchors:
            resolved_basic_count += 1
            if DAG_NODES[tid].get("subject") == "Physics":
                resolved_physics_basic += 1
        basic_gap_resolutions.append(
            {
                "topic_id": tid,
                "subject": str(DAG_NODES[tid].get("subject", "Physics")),
                "current_basic_count": 0,
                "current_medium_count": len(med_pool),
                "proposed_basic_anchors_count": len(anchors),
                "top_anchor_question_ids": [a["question_id"] for a in anchors],
                "lowest_irt_b_in_chapter": anchors[0]["irt_b"] if anchors else 0.0,
                "fastest_ideal_time_sec": min((a["ideal_time_sec"] for a in anchors), default=90),
            }
        )

    advanced_gap_resolutions: list[dict] = []
    resolved_advanced_count = 0
    for tid in sorted(missing_advanced_chapters):
        med_pool = sorted(
            topic_tier_items[tid]["medium"],
            key=lambda r: (-r["irt_b"], -r["ideal_time_sec"], -r["word_count"]),
        )
        anchors = med_pool[:2]
        if anchors:
            resolved_advanced_count += 1
        advanced_gap_resolutions.append(
            {
                "topic_id": tid,
                "subject": str(DAG_NODES[tid].get("subject", "Biology")),
                "current_advanced_count": 0,
                "current_medium_count": len(med_pool),
                "proposed_advanced_anchors_count": len(anchors),
                "top_anchor_question_ids": [a["question_id"] for a in anchors],
                "highest_irt_b_in_chapter": anchors[0]["irt_b"] if anchors else 0.0,
            }
        )

    strict_basic_candidates.sort(key=lambda r: (-r["foundational_score"], r["irt_b"]))
    strict_advanced_candidates.sort(key=lambda r: (-r["irt_b"], -r["ideal_time_sec"]))

    report = {
        "audit_mode": "NON_DESTRUCTIVE_PROPOSAL_ONLY",
        "source_corpus_mutated": False,
        "total_audited_questions": len(questions),
        "gap_resolution_summary": {
            "chapters_missing_basic_before": len(missing_basic_chapters),
            "chapters_resolvable_via_irt_anchors": resolved_basic_count,
            "physics_chapters_missing_basic_before": sum(
                1 for t in missing_basic_chapters if DAG_NODES[t].get("subject") == "Physics"
            ),
            "physics_chapters_resolvable_via_irt_anchors": resolved_physics_basic,
            "chapters_missing_advanced_before": len(missing_advanced_chapters),
            "advanced_chapters_resolvable_via_irt_anchors": resolved_advanced_count,
            "strict_global_basic_candidates_count": len(strict_basic_candidates),
            "strict_global_advanced_candidates_count": len(strict_advanced_candidates),
        },
        "top_basic_recalibration_candidates": strict_basic_candidates[:20],
        "top_advanced_recalibration_candidates": strict_advanced_candidates[:15],
        "per_chapter_basic_gap_anchors": basic_gap_resolutions,
        "per_chapter_advanced_gap_anchors": advanced_gap_resolutions,
    }

    if save_report:
        with open(RECALIBRATION_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    rep = run_irt_tier_recalibration_audit(save_report=True)
    if "--summary" in sys.argv:
        print(json.dumps(rep["gap_resolution_summary"], indent=2))
    else:
        print(json.dumps(rep["gap_resolution_summary"], indent=2))
