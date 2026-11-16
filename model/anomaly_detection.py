from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import binom

D_SCALING = 1.702
RAPID_GUESS_FRACTION = 0.10
RAPID_GUESS_FLOOR_SECONDS = 5.0
LZ_FLAG = -2.33
RAPID_SHARE_FLAG = 0.30
COPY_MIN_SHARED_ITEMS = 30
COPY_MIN_SHARED_WRONG = 8
COPY_P_FLAG = 1e-6
MIN_ITEMS_FOR_PERSON_FIT = 30
MAX_PAIRS = 2_000_000


def _p3pl(a, b, c, theta):
    return c + (1 - c) / (1 + np.exp(-D_SCALING * a * (theta - b)))


def _theta_mle(y: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    grid = np.linspace(-4, 4, 161)
    p = _p3pl(a[None, :], b[None, :], c[None, :], grid[:, None])
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll = (y * np.log(p) + (1 - y) * np.log(1 - p)).sum(axis=1) - 0.5 * grid**2
    return float(grid[int(np.argmax(ll))])


def person_fit_lz(y: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[float, float]:
    theta = _theta_mle(y, a, b, c)
    p = np.clip(_p3pl(a, b, c, theta), 1e-6, 1 - 1e-6)
    l0 = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    e = float(np.sum(p * np.log(p) + (1 - p) * np.log(1 - p)))
    v = float(np.sum(p * (1 - p) * np.log(p / (1 - p)) ** 2))
    return ((l0 - e) / math.sqrt(v) if v > 0 else 0.0), theta


def rapid_guess_thresholds(attempts: pd.DataFrame) -> dict[str, float]:
    out = {}
    for qid, g in attempts.groupby("question_id"):
        ref = float(g["expected_time_seconds"].dropna().iloc[0]) if "expected_time_seconds" in g and g["expected_time_seconds"].notna().any() else float(g["time_taken_seconds"].median())
        out[str(qid)] = max(RAPID_GUESS_FLOOR_SECONDS, RAPID_GUESS_FRACTION * ref)
    return out


def answer_copy_index(attempts: pd.DataFrame, max_pairs: int = MAX_PAIRS) -> list[dict]:
    if "selected_option" not in attempts or "correct_answer" not in attempts:
        return []
    df = attempts.dropna(subset=["selected_option"]).copy()
    df["wrong"] = df["selected_option"].astype(str).str.lower() != df["correct_answer"].astype(str).str.lower()
    wrong = df[df["wrong"]]
    if wrong.empty:
        return []
    option_share = wrong.groupby(["question_id", "selected_option"]).size() / wrong.groupby("question_id").size()
    by_student = {sid: g.set_index("question_id") for sid, g in df.groupby("student_id")}
    flagged = []
    pairs_checked = 0
    for paper, g in df.groupby("paper_number"):
        students = sorted(g["student_id"].unique())
        for s1, s2 in combinations(students, 2):
            pairs_checked += 1
            if pairs_checked > max_pairs:
                break
            a, b = by_student[s1], by_student[s2]
            shared = a.index.intersection(b.index)
            if len(shared) < COPY_MIN_SHARED_ITEMS:
                continue
            both_wrong = [q for q in shared if bool(a.at[q, "wrong"]) and bool(b.at[q, "wrong"])]
            if len(both_wrong) < COPY_MIN_SHARED_WRONG:
                continue
            same = [q for q in both_wrong if str(a.at[q, "selected_option"]).lower() == str(b.at[q, "selected_option"]).lower()]
            probs = [float(option_share.get((q, a.at[q, "selected_option"]), 0.34)) for q in both_wrong]
            mean_p = float(np.mean(probs))
            p_value = float(binom.sf(len(same) - 1, len(both_wrong), mean_p))
            if p_value < COPY_P_FLAG:
                flagged.append({"student_a": s1, "student_b": s2, "paper_number": int(paper), "shared_items": int(len(shared)), "both_wrong": len(both_wrong), "identical_wrong_options": len(same), "chance_match_rate": round(mean_p, 3), "p_value": p_value})
    return flagged


def detect_anomalies(attempts: pd.DataFrame, item_params: dict[str, tuple[float, float, float]]) -> dict:
    report = {
        "what": "patterns that suggest guessing without reading, answer copying, or answers inconsistent with a student's level",
        "methods": {
            "person_fit": f"lz* statistic on the 3PL model (flag below {LZ_FLAG}, about the lowest 1%)",
            "rapid_guessing": f"answer faster than {int(RAPID_GUESS_FRACTION * 100)}% of the item's expected time (min {RAPID_GUESS_FLOOR_SECONDS:g}s); flag a student when over {int(RAPID_SHARE_FLAG * 100)}% of answers are rapid",
            "answer_copying": f"pairs on the same paper whose identical wrong options are very unlikely by chance (binomial p < {COPY_P_FLAG:g})",
        },
        "use": "flags are for human review only; never penalise a student automatically",
    }
    if attempts is None or attempts.empty:
        report["status"] = "INSUFFICIENT_DATA"
        return report
    thresholds = rapid_guess_thresholds(attempts)
    attempts = attempts.copy()
    attempts["rapid"] = [float(t) < thresholds.get(str(q), RAPID_GUESS_FLOOR_SECONDS) for q, t in zip(attempts["question_id"], attempts["time_taken_seconds"])]
    students = []
    for sid, g in attempts.groupby("student_id"):
        entry = {"student_id": str(sid), "responses": int(len(g)), "rapid_share": round(float(g["rapid"].mean()), 3)}
        known = g[g["question_id"].isin(item_params.keys())]
        if len(known) >= MIN_ITEMS_FOR_PERSON_FIT:
            prm = np.array([item_params[q] for q in known["question_id"]])
            lz, theta = person_fit_lz(known["is_correct"].to_numpy(dtype=float), prm[:, 0], prm[:, 1], prm[:, 2])
            entry["lz"] = round(lz, 3)
            entry["theta"] = round(theta, 3)
        flags = []
        if entry.get("lz") is not None and entry["lz"] < LZ_FLAG:
            flags.append("person_misfit")
        if entry["rapid_share"] > RAPID_SHARE_FLAG:
            flags.append("rapid_guessing")
        entry["flags"] = flags
        students.append(entry)
    copies = answer_copy_index(attempts)
    copy_ids = {c["student_a"] for c in copies} | {c["student_b"] for c in copies}
    for e in students:
        if e["student_id"] in copy_ids:
            e["flags"].append("possible_answer_copying")
    flagged = [e for e in students if e["flags"]]
    report.update(
        {
            "status": "ANALYSED",
            "students": len(students),
            "students_flagged": len(flagged),
            "flag_counts": {k: sum(k in e["flags"] for e in students) for k in ("person_misfit", "rapid_guessing", "possible_answer_copying")},
            "copy_pairs": copies[:200],
            "flagged_students": flagged[:500],
        }
    )
    return report
