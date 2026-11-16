from __future__ import annotations

import pandas as pd

from .student_response_training import _metrics, bkt_one_step_predictions

MIN_STUDENTS_PER_GROUP = 50
MAX_AUC_GAP = 0.05
MAX_ECE = 0.05
MAX_COVERAGE_GAP = 0.10
DEFAULT_BKT = {"p_init": 0.4, "p_transit": 0.15, "p_guess": 0.2, "p_slip": 0.1}
CANDIDATE_ATTRIBUTES = ("class_level", "language", "region", "state", "device", "gender", "board", "school_type")


def per_response_bkt_predictions(attempts: pd.DataFrame, chapter_params: dict | None = None) -> pd.DataFrame:
    chapter_params = chapter_params or {}
    out = []
    for (ch, sid), g in attempts.groupby(["chapter", "student_id"], sort=False):
        y = g["is_correct"].to_numpy(dtype=float)
        p = bkt_one_step_predictions([y], chapter_params.get(ch, DEFAULT_BKT))
        out.append(pd.DataFrame({"student_id": sid, "y": y, "p": p}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["student_id", "y", "p"])


def group_report(preds: pd.DataFrame, attributes: pd.DataFrame, attribute: str, score_pairs: pd.DataFrame | None = None, min_students: int = MIN_STUDENTS_PER_GROUP) -> dict:
    merged = preds.merge(attributes[["student_id", attribute]], on="student_id", how="left")
    merged[attribute] = merged[attribute].fillna("not_stated").astype(str)
    groups = {}
    for value, g in merged.groupby(attribute):
        n_students = int(g["student_id"].nunique())
        entry = {"students": n_students, "responses": int(len(g)), "observed_accuracy": round(float(g["y"].mean()), 4)}
        if n_students >= min_students:
            entry.update({f"model_{k}": v for k, v in _metrics(g["y"].to_numpy(), g["p"].to_numpy()).items() if k != "n"})
        else:
            entry["note"] = f"fewer than {min_students} students: not evaluated"
        if score_pairs is not None and not score_pairs.empty:
            sp = score_pairs.merge(attributes[["student_id", attribute]], on="student_id", how="left")
            sp = sp[sp[attribute].fillna("not_stated").astype(str) == value]
            if len(sp) >= min_students:
                inside = (sp["actual"] >= sp["low_90"]) & (sp["actual"] <= sp["high_90"])
                entry["score_mae"] = round(float((sp["predicted"] - sp["actual"]).abs().mean()), 2)
                entry["score_coverage_90"] = round(float(inside.mean()), 4)
        groups[str(value)] = entry
    evaluated = {k: v for k, v in groups.items() if v.get("model_auc") is not None}
    flags = []
    if len(evaluated) >= 2:
        aucs = [v["model_auc"] for v in evaluated.values()]
        if max(aucs) - min(aucs) > MAX_AUC_GAP:
            flags.append(f"AUC gap {max(aucs) - min(aucs):.3f} > {MAX_AUC_GAP}")
        for k, v in evaluated.items():
            if v.get("model_ece") is not None and v["model_ece"] > MAX_ECE:
                flags.append(f"{k}: ECE {v['model_ece']} > {MAX_ECE}")
        covs = [v["score_coverage_90"] for v in groups.values() if "score_coverage_90" in v]
        if len(covs) >= 2 and max(covs) - min(covs) > MAX_COVERAGE_GAP:
            flags.append(f"score-range coverage gap {max(covs) - min(covs):.3f} > {MAX_COVERAGE_GAP}")
    return {"attribute": attribute, "groups": groups, "groups_evaluated": len(evaluated), "flags": flags}


def run_fairness_report(attempts: pd.DataFrame, attributes: pd.DataFrame, chapter_params: dict | None = None, score_pairs: pd.DataFrame | None = None, min_students: int = MIN_STUDENTS_PER_GROUP) -> dict:
    report = {
        "what": "next-answer prediction quality (BKT) and score-range accuracy, split by student group",
        "thresholds": {"max_auc_gap": MAX_AUC_GAP, "max_ece": MAX_ECE, "max_coverage_gap": MAX_COVERAGE_GAP, "min_students_per_group": min_students},
        "attributes_supplied": [],
        "how_to_supply_attributes": "data/training/student_attributes.csv (student_id plus any of language, region, state, device, gender, board, school_type), collected with consent; class_level is read from the students table",
    }
    if attempts is None or attempts.empty or attributes is None or attributes.empty:
        report["status"] = "INSUFFICIENT_DATA"
        return report
    present = [a for a in CANDIDATE_ATTRIBUTES if a in attributes.columns and attributes[a].notna().any()]
    report["attributes_supplied"] = present
    preds = per_response_bkt_predictions(attempts, chapter_params)
    reports = [group_report(preds, attributes, a, score_pairs, min_students) for a in present]
    report["by_attribute"] = reports
    if not any(r["groups_evaluated"] >= 2 for r in reports):
        report["status"] = "INSUFFICIENT_DATA"
    else:
        report["status"] = "GAPS_FLAGGED" if any(r["flags"] for r in reports) else "NO_GAPS_ABOVE_THRESHOLDS"
    report["flags"] = [f"{r['attribute']}: {f}" for r in reports for f in r["flags"]]
    return report

