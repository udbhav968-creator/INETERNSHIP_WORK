from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
CLEAN_QS_PATH = PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json"
PROPOSAL_PATH = MODEL_DIR / "pyq_difficulty_relabel_proposal.json"
EXPLANATION_REPORT = MODEL_DIR / "explanation_quality_report.json"
REVIEW_DIR = PROJECT_ROOT / "data" / "review"
TIER_SHEET = REVIEW_DIR / "difficulty_tier_review.csv"
EXPLANATION_SHEET = REVIEW_DIR / "explanation_rewrite_list.csv"
DECISION_PATH = MODEL_DIR / "difficulty_review_decision.json"
KEY_PATH = MODEL_DIR / "difficulty_review_key.json"
SAMPLE_SIZE = 300
SEED = 20261116
MIN_REVIEWED = 200
MIN_KAPPA_FOR_PROMOTION = 0.40
TIERS = ("basic", "medium", "advanced")
TIER_COLUMNS = ("review_id", "question_id", "subject", "chapter", "question_text", "option_a", "option_b", "option_c", "option_d", "correct_option", "reviewer_tier", "chapter_tag_ok", "correct_chapter", "reviewer_name", "notes")


def build_tier_review_sheet(sample_size: int = SAMPLE_SIZE, path: Path = TIER_SHEET, key_path: Path = KEY_PATH) -> dict:
    qs = {q["question_id"]: q for q in json.loads(CLEAN_QS_PATH.read_text(encoding="utf-8"))}
    items = json.loads(PROPOSAL_PATH.read_text(encoding="utf-8"))["items"]
    rng = np.random.default_rng(SEED)
    disagree = [i for i in items if i["current_tier"] != i["proposed_tier"]]
    agree = [i for i in items if i["current_tier"] == i["proposed_tier"]]
    n_dis = min(len(disagree), int(sample_size * 0.75))
    chosen = list(rng.choice(disagree, n_dis, replace=False)) + list(rng.choice(agree, min(len(agree), sample_size - n_dis), replace=False))
    order = rng.permutation(len(chosen))
    path.parent.mkdir(parents=True, exist_ok=True)
    key_rows = []
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=TIER_COLUMNS)
        w.writeheader()
        for n, idx in enumerate(order, 1):
            it = chosen[int(idx)]
            q = qs[it["question_id"]]
            w.writerow({"review_id": f"R{n:04d}", "question_id": q["question_id"], "subject": q["subject"], "chapter": q["chapter"], "question_text": q["question_text"], **{f"option_{k}": q[f"option_{k}"] for k in "abcd"}, "correct_option": q["correct_option"], "reviewer_tier": "", "chapter_tag_ok": "", "correct_chapter": "", "reviewer_name": "", "notes": ""})
            key_rows.append({"review_id": f"R{n:04d}", "question_id": q["question_id"], "current_tier": it["current_tier"], "proposed_tier": it["proposed_tier"]})
    key_path.write_text(json.dumps({"note": "kept out of the reviewer sheet so reviewers label blind", "rows": key_rows}, indent=2), encoding="utf-8")
    return {"sheet": str(path), "rows": len(chosen), "where_models_disagree": n_dis, "key": str(key_path)}


def build_explanation_sheet(path: Path = EXPLANATION_SHEET) -> dict:
    if not EXPLANATION_REPORT.exists():
        return {"rows": 0}
    rep = json.loads(EXPLANATION_REPORT.read_text(encoding="utf-8"))
    qs = {q["question_id"]: q for q in json.loads(CLEAN_QS_PATH.read_text(encoding="utf-8"))}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=("question_id", "subject", "chapter", "issues", "question_text", "correct_option", "current_explanation", "new_explanation", "reviewer_name"))
        w.writeheader()
        for row in rep["failing_questions"]:
            q = qs[row["question_id"]]
            w.writerow({"question_id": q["question_id"], "subject": q["subject"], "chapter": q["chapter"], "issues": "; ".join(row["issues"]), "question_text": q["question_text"], "correct_option": q["correct_option"], "current_explanation": q["explanation"], "new_explanation": "", "reviewer_name": ""})
    return {"sheet": str(path), "rows": len(rep["failing_questions"])}


def score_tier_review(path: Path = TIER_SHEET, save: bool = True, key_path: Path = KEY_PATH) -> dict:
    key = {r["review_id"]: r for r in json.loads(key_path.read_text(encoding="utf-8"))["rows"]}
    rows = []
    chapter_checked = chapter_wrong = 0
    corrections = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            t = str(r.get("reviewer_tier", "")).strip().lower()
            if t in TIERS and r["review_id"] in key:
                rows.append((t, key[r["review_id"]]["current_tier"], key[r["review_id"]]["proposed_tier"]))
            ok = str(r.get("chapter_tag_ok", "")).strip().lower()
            if ok in ("yes", "no", "y", "n"):
                chapter_checked += 1
                if ok in ("no", "n"):
                    chapter_wrong += 1
                    corrections.append({"question_id": r["question_id"], "tagged": r["chapter"], "correct": r.get("correct_chapter", "")})
    report = {
        "reviewed": len(rows),
        "min_reviewed": MIN_REVIEWED,
        "min_kappa_for_promotion": MIN_KAPPA_FOR_PROMOTION,
        "chapter_tags_checked": chapter_checked,
        "chapter_tag_error_rate": round(chapter_wrong / chapter_checked, 4) if chapter_checked else None,
        "chapter_corrections": corrections,
    }
    if len(rows) < MIN_REVIEWED:
        report["decision"] = "NOT_ENOUGH_REVIEWS"
    else:
        human, cur, prop = zip(*rows)
        k_cur = float(cohen_kappa_score(human, cur))
        k_prop = float(cohen_kappa_score(human, prop))
        report.update({
            "kappa_reviewer_vs_current_tiers": round(k_cur, 4),
            "kappa_reviewer_vs_v21_tiers": round(k_prop, 4),
            "accuracy_current": round(float(np.mean([h == c for h, c in zip(human, cur)])), 4),
            "accuracy_v21": round(float(np.mean([h == p for h, p in zip(human, prop)])), 4),
        })
        report["decision"] = "PROMOTE_V21" if (k_prop >= MIN_KAPPA_FOR_PROMOTION and k_prop > k_cur) else "DO_NOT_PROMOTE"
    if save:
        DECISION_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Content-team review packs and scoring")
    ap.add_argument("action", choices=("build", "score"))
    args = ap.parse_args()
    if args.action == "build":
        print(json.dumps({"difficulty": build_tier_review_sheet(), "explanations": build_explanation_sheet()}, indent=2))
    else:
        print(json.dumps(score_tier_review(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
