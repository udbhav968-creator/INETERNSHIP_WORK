from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .student_response_training import (
    MIN_SEQUENCES_PER_CHAPTER,
    _metrics,
    bkt_one_step_predictions,
    fit_bkt_em,
    split_students,
)

MIN_STUDENTS = 200
MIN_RESPONSES = 20000
GATE_MIN_LOGLOSS_GAIN = 0.01
GATE_MAX_AUC_DROP = 0.005
DEEP_KT_MIN_RESPONSES = 1_000_000
DEFAULT_BKT = {"p_init": 0.4, "p_transit": 0.15, "p_guess": 0.2, "p_slip": 0.1}


def pfa_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["student_id", "chapter", "is_correct"]].copy()
    grp = out.groupby(["student_id", "chapter"], sort=False)["is_correct"]
    prior_correct = grp.cumsum() - out["is_correct"]
    prior_total = grp.cumcount()
    out["log_success"] = np.log1p(prior_correct.to_numpy(dtype=float))
    out["log_failure"] = np.log1p((prior_total - prior_correct).to_numpy(dtype=float))
    return out


def fit_pfa(train: pd.DataFrame, min_rows: int = 200) -> dict:
    feats = pfa_features(train)
    models = {}
    for ch, g in feats.groupby("chapter"):
        if len(g) < min_rows or g["is_correct"].nunique() < 2:
            continue
        m = LogisticRegression(C=1.0, max_iter=500)
        m.fit(g[["log_success", "log_failure"]].to_numpy(), g["is_correct"].to_numpy())
        models[ch] = m
    glob = LogisticRegression(C=1.0, max_iter=500)
    glob.fit(feats[["log_success", "log_failure"]].to_numpy(), feats["is_correct"].to_numpy())
    return {"chapters": models, "global": glob}


def predict_pfa(model: dict, df: pd.DataFrame) -> np.ndarray:
    feats = pfa_features(df)
    p = np.empty(len(feats))
    X = feats[["log_success", "log_failure"]].to_numpy()
    for ch, idx in feats.groupby("chapter").indices.items():
        m = model["chapters"].get(ch, model["global"])
        p[idx] = m.predict_proba(X[idx])[:, 1]
    return p


def predict_bkt(params: dict, df: pd.DataFrame) -> np.ndarray:
    p = np.empty(len(df))
    for (ch, _), idx in df.groupby(["chapter", "student_id"], sort=False).indices.items():
        y = df["is_correct"].to_numpy(dtype=float)[idx]
        p[idx] = bkt_one_step_predictions([y], params.get(ch, DEFAULT_BKT))
    return p


def compare_knowledge_tracing(df: pd.DataFrame, min_students: int = MIN_STUDENTS, min_responses: int = MIN_RESPONSES) -> dict:
    report = {
        "champion": "BKT (per-chapter EM)",
        "challenger": "PFA (per-chapter logistic regression on prior successes and failures)",
        "protocol": "students split 80/20 by student; one-step-ahead prediction on held-out students",
        "gate": {"min_logloss_gain": GATE_MIN_LOGLOSS_GAIN, "max_auc_drop": GATE_MAX_AUC_DROP},
        "deep_models": f"SAKT/AKT are not trained: they need about {DEEP_KT_MIN_RESPONSES:,} responses to beat these baselines reliably",
    }
    if df is None or df.empty or df["student_id"].nunique() < min_students or len(df) < min_responses:
        report["status"] = "INSUFFICIENT_DATA"
        report["students"] = 0 if df is None or df.empty else int(df["student_id"].nunique())
        report["responses"] = 0 if df is None else int(len(df))
        return report
    df = df.reset_index(drop=True)
    tr_ids, te_ids = split_students(df)
    train = df[df["student_id"].isin(tr_ids)].reset_index(drop=True)
    test = df[df["student_id"].isin(te_ids)].reset_index(drop=True)
    bkt_params = {}
    for ch, g in train.groupby("chapter"):
        seqs = [s["is_correct"].to_numpy(dtype=float) for _, s in g.groupby("student_id", sort=False)]
        if len(seqs) >= MIN_SEQUENCES_PER_CHAPTER:
            bkt_params[ch] = fit_bkt_em(seqs)
    pfa = fit_pfa(train)
    y = test["is_correct"].to_numpy(dtype=float)
    m_bkt = _metrics(y, predict_bkt(bkt_params, test))
    m_pfa = _metrics(y, predict_pfa(pfa, test))
    gain = (m_bkt["log_loss"] - m_pfa["log_loss"]) / m_bkt["log_loss"]
    auc_drop = (m_bkt["auc"] or 0) - (m_pfa["auc"] or 0)
    passes = gain >= GATE_MIN_LOGLOSS_GAIN and auc_drop <= GATE_MAX_AUC_DROP and m_pfa["ece"] <= 0.05
    report.update(
        {
            "students": int(df["student_id"].nunique()),
            "responses": int(len(df)),
            "heldout_students": len(te_ids),
            "bkt": m_bkt,
            "pfa": m_pfa,
            "logloss_gain": round(float(gain), 4),
            "auc_drop": round(float(auc_drop), 4),
            "status": "CHALLENGER_PASSES_GATE" if passes else "KEEP_BKT",
        }
    )
    return report
