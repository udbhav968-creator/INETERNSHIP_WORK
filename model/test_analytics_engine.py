import math
from collections import defaultdict

import numpy as np

from .assessment_blueprint import MARKS_CORRECT, MARKS_WRONG, SUBJECT_ORDER, load_forms
from .irt_3pl_model import theta_to_percentile
from .student_profile_initializer import _bank_by_id, _eap, _p_correct, THETA_GRID

ANALYTICS_VERSION = "test-analytics-v1"
SKIP_THRESHOLD = (-MARKS_WRONG) / (MARKS_CORRECT - MARKS_WRONG)
UNEXPECTED_MISS_P = 0.70
STRONG_SOLVE_P = 0.35
RUSHED_RATIO = 0.40
SLOW_RATIO = 1.50


def _equated_marks_curve(bank: dict[str, dict]) -> np.ndarray:
    ref = [bank[qid] for qid in load_forms()["forms"][0]["question_ids"] if qid in bank]
    total = np.zeros_like(THETA_GRID)
    for q in ref:
        p = _p_correct(THETA_GRID, q)
        total += MARKS_CORRECT * p + MARKS_WRONG * (1.0 - p)
    return total


def _bucket() -> dict:
    return {"total": 0, "attempted": 0, "correct": 0, "incorrect": 0, "unattempted": 0, "marks": 0, "time_seconds": 0.0}


def _finish(b: dict) -> dict:
    b["accuracy"] = round(b["correct"] / b["attempted"], 4) if b["attempted"] else 0.0
    b["max_marks"] = b["total"] * MARKS_CORRECT
    b["time_seconds"] = round(b["time_seconds"], 1)
    return b


def analyze_test(
    paper_question_ids: list[str],
    responses: list[dict],
    test_type: str = "mock_180",
    source: str = "unified",
) -> dict:
    bank = _bank_by_id(source)
    resp = {r["question_id"]: r for r in responses if r.get("question_id")}
    obs = []
    for qid in paper_question_ids:
        r = resp.get(qid)
        q = bank.get(qid)
        if q is not None and r is not None and r.get("is_correct") is not None:
            obs.append((q, 1 if r["is_correct"] else 0))
    theta, sem, _ = _eap(obs, 0.0, 1.0)
    curve = _equated_marks_curve(bank)

    overall = _bucket()
    by_subject = defaultdict(_bucket)
    by_chapter = defaultdict(_bucket)
    by_tier = defaultdict(_bucket)
    questions = []
    negative_marks_lost = 0
    avoidable_negative = 0
    for pos, qid in enumerate(paper_question_ids, start=1):
        q = bank.get(qid)
        if q is None:
            continue
        r = resp.get(qid)
        attempted = r is not None and r.get("is_correct") is not None
        y = bool(r["is_correct"]) if attempted else None
        p = float(_p_correct(np.array([theta]), q)[0])
        t = float(r.get("time_taken_seconds")) if attempted and r.get("time_taken_seconds") is not None else None
        ratio = round(t / max(1.0, float(q["ideal_time_sec"])), 3) if t is not None else None
        mark = 0 if not attempted else (MARKS_CORRECT if y else MARKS_WRONG)
        for b in (overall, by_subject[q["subject"]], by_chapter[q["unit"]], by_tier[q["tier"]]):
            b["total"] += 1
            b["marks"] += mark
            b["time_seconds"] += t or 0.0
            if not attempted:
                b["unattempted"] += 1
            else:
                b["attempted"] += 1
                b["correct" if y else "incorrect"] += 1
        flags = []
        if attempted and not y:
            negative_marks_lost += -MARKS_WRONG
            if p < SKIP_THRESHOLD:
                flags.append("should_have_skipped")
                avoidable_negative += -MARKS_WRONG
            if p >= UNEXPECTED_MISS_P:
                flags.append("unexpected_miss")
        if attempted and y and p <= STRONG_SOLVE_P:
            flags.append("strong_solve")
        if not attempted and p >= 0.5:
            flags.append("skipped_likely_correct")
        if ratio is not None and ratio < RUSHED_RATIO and attempted and not y:
            flags.append("rushed")
        if ratio is not None and ratio > SLOW_RATIO:
            flags.append("time_sink")
        questions.append(
            {
                "position": pos,
                "question_id": qid,
                "subject": q["subject"],
                "chapter": q["chapter"],
                "difficulty": q["tier"],
                "attempted": attempted,
                "is_correct": y,
                "marks": mark,
                "expected_p_correct": round(p, 4),
                "time_taken_seconds": t,
                "time_ratio": ratio,
                "flags": flags,
            }
        )

    chapters = {k: _finish(v) for k, v in by_chapter.items()}
    weakest = sorted(
        (
            {"chapter": k.split("::", 1)[1], "subject": k.split("::", 1)[0], "marks": v["marks"], "max_marks": v["max_marks"], "accuracy": v["accuracy"], "lost_marks": v["max_marks"] - v["marks"]}
            for k, v in chapters.items()
        ),
        key=lambda c: (-c["lost_marks"], c["accuracy"]),
    )
    flag_counts = defaultdict(int)
    for qrow in questions:
        for f in qrow["flags"]:
            flag_counts[f] += 1
    return {
        "analytics_version": ANALYTICS_VERSION,
        "test_type": test_type,
        "summary": _finish(overall),
        "theta": round(theta, 4),
        "theta_sem": round(sem, 4),
        "theta_percentile": theta_to_percentile(theta),
        "equated_marks": round(float(np.interp(theta, THETA_GRID, curve)), 1),
        "by_subject": {s: _finish(by_subject[s]) for s in SUBJECT_ORDER if s in by_subject},
        "by_difficulty": {t: _finish(v) for t, v in by_tier.items()},
        "by_chapter": chapters,
        "weakest_chapters": weakest[:8],
        "negative_marking": {
            "marks_lost_to_wrong_answers": negative_marks_lost,
            "avoidable_marks_lost": avoidable_negative,
            "skip_rule": f"skip when estimated P(correct) < {SKIP_THRESHOLD:.2f} (expected value {MARKS_CORRECT}P {MARKS_WRONG:+d}(1-P) < 0)",
        },
        "question_flag_counts": dict(flag_counts),
        "questions": questions,
    }


def score_trend(tests: list[dict], source: str = "unified") -> dict:
    bank = _bank_by_id(source)
    curve = _equated_marks_curve(bank)
    pts = []
    for i, t in enumerate(tests):
        th = float(t["theta"])
        pts.append(
            {
                "index": i,
                "test_type": t.get("test_type"),
                "timestamp": t.get("timestamp"),
                "raw_marks": t.get("marks"),
                "theta": round(th, 4),
                "theta_sem": t.get("theta_sem"),
                "equated_marks": round(float(np.interp(th, THETA_GRID, curve)), 1),
            }
        )
    if len(pts) >= 2:
        x = np.arange(len(pts), dtype=float)
        w = np.array([1.0 / max(0.01, float(p["theta_sem"] or 0.2)) ** 2 for p in pts])
        th = np.array([p["theta"] for p in pts])
        xm = np.sum(w * x) / np.sum(w)
        ym = np.sum(w * th) / np.sum(w)
        slope = float(np.sum(w * (x - xm) * (th - ym)) / max(1e-9, np.sum(w * (x - xm) ** 2)))
        slope_se = float(math.sqrt(1.0 / max(1e-9, np.sum(w * (x - xm) ** 2))))
        first, last = pts[0], pts[-1]
        diff = last["theta"] - first["theta"]
        diff_se = math.sqrt(float(first["theta_sem"] or 0.2) ** 2 + float(last["theta_sem"] or 0.2) ** 2)
        trend = {
            "theta_slope_per_test": round(slope, 4),
            "theta_slope_se": round(slope_se, 4),
            "equated_marks_change_first_to_last": round(last["equated_marks"] - first["equated_marks"], 1),
            "change_is_significant": bool(abs(diff) > 1.96 * diff_se),
            "direction": "improving" if slope > slope_se else ("declining" if slope < -slope_se else "stable"),
        }
    else:
        trend = {"theta_slope_per_test": None, "direction": "insufficient_tests"}
    best = max(pts, key=lambda p: p["equated_marks"]) if pts else None
    return {
        "analytics_version": ANALYTICS_VERSION,
        "points": pts,
        "trend": trend,
        "best_equated_marks": best["equated_marks"] if best else None,
        "average_equated_marks": round(float(np.mean([p["equated_marks"] for p in pts])), 1) if pts else None,
    }
