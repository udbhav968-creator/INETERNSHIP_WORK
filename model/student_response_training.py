import argparse
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
ATTEMPTS_CSV = PROJECT_ROOT / "data" / "training" / "student_attempts.csv"
SQLITE_DB = PROJECT_ROOT / "backend" / "klassed.db"
PYQ_PATH = PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json"
SAMPLE_PATH = PROJECT_ROOT / "data" / "tagged_questions" / "neet_samples_001_to_101_verified.json"
PRIOR_IRT_PATH = MODEL_DIR / "irt_item_parameters.json"
PRIOR_SAMPLE_IRT_PATH = MODEL_DIR / "sample_paper_irt_parameters.json"
PRIOR_BKT_PATH = MODEL_DIR / "bkt_chapter_parameters.json"
V21_PROPOSAL_PATH = MODEL_DIR / "pyq_difficulty_relabel_proposal.json"
OUTPUT_DIR = MODEL_DIR / "student_models"
REPORT_NAME = "student_response_training_report.json"
FITTED_IRT_NAME = "irt_item_parameters_fitted.json"
FITTED_BKT_NAME = "bkt_chapter_parameters_fitted.json"
BACKUP_DIR = MODEL_DIR / "backups"

MIN_STUDENTS = 200
MIN_TOTAL_RESPONSES = 20000
MIN_RESPONSES_PER_ITEM = 30
MIN_SEQUENCES_PER_CHAPTER = 30
HOLDOUT_STUDENT_FRACTION = 0.20
OBSERVED_PREFIX_FRACTION = 0.50
SPLIT_SEED = 20261116

D_SCALING = 1.702
QUAD_POINTS = 21
IRT_EM_ITERATIONS = 60
IRT_M_STEPS = 25
IRT_LEARNING_RATE = 0.05
C_MAX = 0.35
PRIOR_LOG_A_SD = 0.5
PRIOR_B_SD = 2.0
PRIOR_C_ALPHA = 5.0
PRIOR_C_BETA = 17.0

BKT_EM_ITERATIONS = 40
BKT_MAX_GUESS = 0.40
BKT_MAX_SLIP = 0.40

GATE_MIN_LOGLOSS_GAIN = 0.01
GATE_MAX_AUC_DROP = 0.005
GATE_MAX_ECE = 0.05

TIER_ORDER = {"basic": 0, "medium": 1, "advanced": 2}


def _bank_metadata() -> dict[str, dict]:
    meta: dict[str, dict] = {}
    if PYQ_PATH.exists():
        for q in json.loads(PYQ_PATH.read_text(encoding="utf-8")):
            meta[str(q["question_id"])] = {"subject": q.get("subject"), "chapter": q.get("chapter"), "tier": str(q.get("difficulty_level", "")).lower(), "source": "pyq"}
    if SAMPLE_PATH.exists():
        raw = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
        for q in raw if isinstance(raw, list) else raw.get("questions", []):
            qid = str(q.get("question_id") or q.get("id"))
            meta.setdefault(qid, {"subject": q.get("subject"), "chapter": q.get("chapter"), "tier": str(q.get("difficulty_level", "")).lower(), "source": "sample_paper"})
    return meta


def _prior_irt() -> dict[str, tuple[float, float, float]]:
    out: dict[str, tuple[float, float, float]] = {}
    if PRIOR_SAMPLE_IRT_PATH.exists():
        for qid, p in json.loads(PRIOR_SAMPLE_IRT_PATH.read_text(encoding="utf-8")).get("items", {}).items():
            out[str(qid)] = (float(p["a"]), float(p["b"]), float(p["c"]))
    if PRIOR_IRT_PATH.exists():
        for qid, p in json.loads(PRIOR_IRT_PATH.read_text(encoding="utf-8")).items():
            out[str(qid)] = (float(p["a"]), float(p["b"]), float(p["c"]))
    return out


def _prior_bkt() -> dict[str, dict]:
    if not PRIOR_BKT_PATH.exists():
        return {}
    return json.loads(PRIOR_BKT_PATH.read_text(encoding="utf-8")).get("chapters", {})


def _read_database() -> pd.DataFrame:
    url = os.environ.get("DATABASE_URL") or os.environ.get("KLASSED_DB_URL")
    if url and url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    if url and url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    if not url and SQLITE_DB.exists():
        url = f"sqlite:///{SQLITE_DB.as_posix()}"
    if not url:
        return pd.DataFrame()
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(url)
        with engine.connect() as conn:
            return pd.read_sql(
                text("SELECT student_id, question_id, paper_number, is_correct, time_taken_seconds, timestamp FROM test_attempts"),
                conn,
            )
    except Exception:
        return pd.DataFrame()


def load_attempts(csv_path: Path | None = None) -> tuple[pd.DataFrame, str]:
    path = csv_path or ATTEMPTS_CSV
    df = pd.DataFrame()
    source = "none"
    if path.exists():
        df = pd.read_csv(path)
        source = str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)
    if df.empty and csv_path is None:
        df = _read_database()
        source = "database:test_attempts" if not df.empty else source
    if df.empty:
        return pd.DataFrame(columns=["student_id", "question_id", "paper_number", "is_correct", "timestamp", "chapter", "subject"]), source
    df = df.dropna(subset=["student_id", "question_id", "is_correct"]).copy()
    df["student_id"] = df["student_id"].astype(str)
    df["question_id"] = df["question_id"].astype(str)
    df["is_correct"] = df["is_correct"].map(lambda v: 1 if str(v).strip().lower() in {"1", "true", "t", "yes"} else 0).astype(int)
    df["timestamp"] = pd.to_datetime(df.get("timestamp"), errors="coerce", utc=True)
    if "paper_number" not in df:
        df["paper_number"] = 0
    meta = _bank_metadata()
    df["chapter"] = df["question_id"].map(lambda q: (meta.get(q) or {}).get("chapter"))
    df["subject"] = df["question_id"].map(lambda q: (meta.get(q) or {}).get("subject"))
    df = df.dropna(subset=["chapter"])
    df = df.drop_duplicates(subset=["student_id", "question_id", "timestamp"])
    df["_order"] = np.arange(len(df))
    df = df.sort_values(["student_id", "timestamp", "_order"]).drop(columns="_order").reset_index(drop=True)
    return df, source


def data_readiness(df: pd.DataFrame) -> dict:
    n_students = int(df["student_id"].nunique()) if len(df) else 0
    first = df.drop_duplicates(["student_id", "question_id"]) if len(df) else df
    per_item = first.groupby("question_id").size() if len(first) else pd.Series(dtype=int)
    items_ready = int((per_item >= MIN_RESPONSES_PER_ITEM).sum())
    checks = {
        f"students >= {MIN_STUDENTS}": n_students >= MIN_STUDENTS,
        f"responses >= {MIN_TOTAL_RESPONSES}": len(df) >= MIN_TOTAL_RESPONSES,
        f"items with >= {MIN_RESPONSES_PER_ITEM} responses >= 50": items_ready >= 50,
    }
    return {
        "students": n_students,
        "responses": int(len(df)),
        "unique_items": int(per_item.size),
        "items_with_enough_responses": items_ready,
        "checks": checks,
        "ready": all(checks.values()),
    }


def split_students(df: pd.DataFrame, seed: int = SPLIT_SEED) -> tuple[set, set]:
    students = np.array(sorted(df["student_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(students)
    n_test = max(1, int(round(len(students) * HOLDOUT_STUDENT_FRACTION)))
    return set(students[n_test:]), set(students[:n_test])


def _quadrature() -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.hermite_e.hermegauss(QUAD_POINTS)
    return x, w / w.sum()


def _p3pl(a, b, c, theta):
    z = D_SCALING * a * (theta - b)
    return c + (1.0 - c) / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def fit_irt_3pl_mml(train: pd.DataFrame, init: dict[str, tuple[float, float, float]]) -> dict:
    first = train.drop_duplicates(["student_id", "question_id"])
    counts = first.groupby("question_id").size()
    items = sorted(counts[counts >= MIN_RESPONSES_PER_ITEM].index)
    data = first[first["question_id"].isin(set(items))]
    if not items or data.empty:
        return {"items": {}, "iterations": 0}
    item_idx = {q: i for i, q in enumerate(items)}
    stud = sorted(data["student_id"].unique())
    stud_idx = {s: i for i, s in enumerate(stud)}
    ii = data["question_id"].map(item_idx).to_numpy()
    ss = data["student_id"].map(stud_idx).to_numpy()
    yy = data["is_correct"].to_numpy(dtype=float)
    theta_q, w_q = _quadrature()

    p_val = data.groupby("question_id")["is_correct"].mean().reindex(items).to_numpy()
    log_a = np.log(np.array([init.get(q, (1.0, 0.0, 0.2))[0] for q in items]).clip(0.3, 3.0))
    b = -np.log(np.clip(p_val, 0.02, 0.98) / (1 - np.clip(p_val, 0.02, 0.98))) / D_SCALING
    c0 = np.full(len(items), 0.2)
    g = np.log(c0 / (C_MAX - c0))
    m = {k: np.zeros(len(items)) for k in ("a", "b", "g")}
    v = {k: np.zeros(len(items)) for k in ("a", "b", "g")}
    step = 0
    history = []
    for _ in range(IRT_EM_ITERATIONS):
        a = np.exp(log_a)
        c = C_MAX / (1.0 + np.exp(-g))
        P = _p3pl(a[ii, None], b[ii, None], c[ii, None], theta_q[None, :])
        ll = yy[:, None] * np.log(P) + (1 - yy[:, None]) * np.log(1 - P)
        stud_ll = np.zeros((len(stud), QUAD_POINTS))
        np.add.at(stud_ll, ss, ll)
        post = stud_ll + np.log(w_q)[None, :]
        mx = post.max(axis=1, keepdims=True)
        marginal = mx[:, 0] + np.log(np.exp(post - mx).sum(axis=1))
        post = np.exp(post - mx)
        post /= post.sum(axis=1, keepdims=True)
        history.append(float(marginal.sum()))
        wr = post[ss]
        n_iq = np.zeros((len(items), QUAD_POINTS))
        r_iq = np.zeros((len(items), QUAD_POINTS))
        np.add.at(n_iq, ii, wr)
        np.add.at(r_iq, ii, wr * yy[:, None])
        for _ in range(IRT_M_STEPS):
            a = np.exp(log_a)
            sg = 1.0 / (1.0 + np.exp(-g))
            c = C_MAX * sg
            z = D_SCALING * a[:, None] * (theta_q[None, :] - b[:, None])
            s = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            P = np.clip(c[:, None] + (1 - c[:, None]) * s, 1e-6, 1 - 1e-6)
            dP = r_iq / P - (n_iq - r_iq) / (1 - P)
            ds = (1 - c[:, None]) * s * (1 - s)
            grad_a = (dP * ds * D_SCALING * (theta_q[None, :] - b[:, None])).sum(1) * a - log_a / PRIOR_LOG_A_SD**2
            grad_b = (dP * ds * (-D_SCALING * a[:, None])).sum(1) - b / PRIOR_B_SD**2
            dc = (dP * (1 - s)).sum(1) + (PRIOR_C_ALPHA - 1) / c - (PRIOR_C_BETA - 1) / (1 - c)
            grad_g = dc * C_MAX * sg * (1 - sg)
            step += 1
            for key, grad, ref in (("a", grad_a, "log_a"), ("b", grad_b, "b"), ("g", grad_g, "g")):
                m[key] = 0.9 * m[key] + 0.1 * grad
                v[key] = 0.999 * v[key] + 0.001 * grad**2
                upd = IRT_LEARNING_RATE * (m[key] / (1 - 0.9**step)) / (np.sqrt(v[key] / (1 - 0.999**step)) + 1e-8)
                if ref == "log_a":
                    log_a = np.clip(log_a + upd, math.log(0.2), math.log(4.0))
                elif ref == "b":
                    b = np.clip(b + upd, -4.0, 4.0)
                else:
                    g = np.clip(g + upd, -6.0, 6.0)
    a = np.exp(log_a)
    c = C_MAX / (1.0 + np.exp(-g))
    n_resp = data.groupby("question_id").size().reindex(items).to_numpy()
    params = {
        q: {"a": round(float(a[k]), 4), "b": round(float(b[k]), 4), "c": round(float(c[k]), 4), "n_responses": int(n_resp[k]), "p_value": round(float(p_val[k]), 4)}
        for q, k in item_idx.items()
    }
    return {"items": params, "iterations": IRT_EM_ITERATIONS, "marginal_loglik_trace": [round(h, 2) for h in history[:: max(1, len(history) // 10)]]}


def link_prior_to_fitted(fitted: dict, prior: dict) -> dict:
    common = [q for q in fitted if q in prior]
    if len(common) < 10:
        return {"A": 1.0, "B": 0.0, "common_items": len(common)}
    bf = np.array([fitted[q]["b"] for q in common])
    bp = np.array([prior[q][1] for q in common])
    A = float(np.std(bf) / max(1e-6, np.std(bp)))
    B = float(np.mean(bf) - A * np.mean(bp))
    return {"A": round(A, 4), "B": round(B, 4), "common_items": len(common)}


def _params_for(qid: str, fitted: dict | None, prior: dict, link: dict) -> tuple[float, float, float]:
    if fitted and qid in fitted:
        p = fitted[qid]
        return p["a"], p["b"], p["c"]
    a, b, c = prior.get(qid, (1.0, 0.0, 0.2))
    return a / link["A"], link["A"] * b + link["B"], c


def _eap(items: list[str], ys: np.ndarray, fitted: dict | None, prior: dict, link: dict) -> tuple[float, float]:
    theta_q, w_q = _quadrature()
    if not items:
        return 0.0, 1.0
    abc = np.array([_params_for(q, fitted, prior, link) for q in items])
    P = _p3pl(abc[:, 0:1], abc[:, 1:2], abc[:, 2:3], theta_q[None, :])
    ll = (ys[:, None] * np.log(P) + (1 - ys[:, None]) * np.log(1 - P)).sum(0) + np.log(w_q)
    w = np.exp(ll - ll.max())
    w /= w.sum()
    mean = float((w * theta_q).sum())
    return mean, float(math.sqrt(max(1e-9, (w * (theta_q - mean) ** 2).sum())))


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    conf_bins = np.linspace(0, 1, 11)
    ece = 0.0
    for lo, hi in zip(conf_bins[:-1], conf_bins[1:]):
        mk = (p > lo) & (p <= hi)
        if mk.any():
            ece += mk.mean() * abs(p[mk].mean() - y[mk].mean())
    return {
        "n": int(len(y)),
        "auc": round(float(roc_auc_score(y, p)), 4) if len(set(y.tolist())) > 1 else None,
        "log_loss": round(float(log_loss(y, p, labels=[0, 1])), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "ece": round(float(ece), 4),
    }


def _split_observed_future(test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs, fut = [], []
    for _, g in test.groupby("student_id", sort=False):
        k = max(1, int(len(g) * OBSERVED_PREFIX_FRACTION))
        obs.append(g.iloc[:k])
        fut.append(g.iloc[k:])
    return pd.concat(obs), pd.concat(fut) if fut else test.iloc[0:0]


def evaluate_irt(train: pd.DataFrame, test: pd.DataFrame, fitted: dict, prior: dict, link: dict) -> dict:
    observed, future = _split_observed_future(test)
    future = future.drop_duplicates(["student_id", "question_id"])
    y = future["is_correct"].to_numpy(dtype=float)
    if len(y) == 0:
        return {"available": False}
    global_rate = float(train["is_correct"].mean())
    item_rate = train.groupby("question_id")["is_correct"].agg(["sum", "count"])
    smooth = ((item_rate["sum"] + 2 * global_rate) / (item_rate["count"] + 2)).to_dict()
    preds = {"global_rate": [], "item_p_value": [], "prior_irt": [], "fitted_irt": []}
    obs_by_student = {s: g for s, g in observed.groupby("student_id")}
    theta_cache: dict = {}
    for row in future.itertuples(index=False):
        g = obs_by_student.get(row.student_id)
        qs = g["question_id"].tolist() if g is not None else []
        ys = g["is_correct"].to_numpy(dtype=float) if g is not None else np.zeros(0)
        if row.student_id not in theta_cache:
            theta_cache[row.student_id] = (
                _eap(qs, ys, None, prior, {"A": 1.0, "B": 0.0})[0],
                _eap(qs, ys, fitted, prior, link)[0],
            )
        t_prior, t_fit = theta_cache[row.student_id]
        pa, pb, pc = prior.get(row.question_id, (1.0, 0.0, 0.2))
        fa, fb, fc = _params_for(row.question_id, fitted, prior, link)
        preds["global_rate"].append(global_rate)
        preds["item_p_value"].append(smooth.get(row.question_id, global_rate))
        preds["prior_irt"].append(float(_p3pl(pa, pb, pc, t_prior)))
        preds["fitted_irt"].append(float(_p3pl(fa, fb, fc, t_fit)))
    return {
        "available": True,
        "protocol": f"held-out students; first {int(OBSERVED_PREFIX_FRACTION * 100)}% of each student's answers estimate ability, the rest are predicted",
        "held_out_students": int(test["student_id"].nunique()),
        "predicted_responses": int(len(y)),
        "models": {k: _metrics(y, np.array(v)) for k, v in preds.items()},
    }


def _pad_sequences(seqs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    L = max(len(s) for s in seqs)
    Y = np.zeros((len(seqs), L))
    M = np.zeros((len(seqs), L), dtype=bool)
    for i, s in enumerate(seqs):
        Y[i, : len(s)] = s
        M[i, : len(s)] = True
    return Y, M


def fit_bkt_em(seqs: list[np.ndarray], init: dict | None = None) -> dict:
    Y, M = _pad_sequences(seqs)
    n, L = Y.shape
    L0 = float((init or {}).get("p_init", 0.4))
    T = float((init or {}).get("p_transit", 0.15))
    G = float(min(BKT_MAX_GUESS, (init or {}).get("p_guess", 0.2)))
    S = float(min(BKT_MAX_SLIP, (init or {}).get("p_slip", 0.1)))
    for _ in range(BKT_EM_ITERATIONS):
        e1 = np.where(Y == 1, 1 - S, S)
        e0 = np.where(Y == 1, G, 1 - G)
        e1 = np.where(M, e1, 1.0)
        e0 = np.where(M, e0, 1.0)
        f0 = np.zeros((n, L))
        f1 = np.zeros((n, L))
        scale = np.ones((n, L))
        a0, a1 = (1 - L0) * e0[:, 0], L0 * e1[:, 0]
        sc = a0 + a1
        f0[:, 0], f1[:, 0], scale[:, 0] = a0 / sc, a1 / sc, sc
        for t in range(1, L):
            p0 = f0[:, t - 1] * (1 - T)
            p1 = f1[:, t - 1] + f0[:, t - 1] * T
            a0, a1 = p0 * e0[:, t], p1 * e1[:, t]
            sc = a0 + a1
            f0[:, t], f1[:, t], scale[:, t] = a0 / sc, a1 / sc, sc
        b0 = np.ones((n, L))
        b1 = np.ones((n, L))
        for t in range(L - 2, -1, -1):
            nb0 = (1 - T) * e0[:, t + 1] * b0[:, t + 1] + T * e1[:, t + 1] * b1[:, t + 1]
            nb1 = e1[:, t + 1] * b1[:, t + 1]
            b0[:, t] = np.where(M[:, t + 1], nb0 / scale[:, t + 1], 1.0)
            b1[:, t] = np.where(M[:, t + 1], nb1 / scale[:, t + 1], 1.0)
        g0 = f0 * b0
        g1 = f1 * b1
        tot = g0 + g1
        g0, g1 = g0 / tot, g1 / tot
        xi = np.zeros((n, max(1, L - 1)))
        for t in range(L - 1):
            xi[:, t] = np.where(M[:, t + 1], f0[:, t] * T * e1[:, t + 1] * b1[:, t + 1] / scale[:, t + 1], 0.0)
        valid_prev = M[:, :-1] & M[:, 1:] if L > 1 else np.zeros((n, 0), dtype=bool)
        L0 = float(np.clip(g1[:, 0].mean(), 0.01, 0.99))
        denom_T = (g0[:, :-1] * valid_prev).sum() if L > 1 else 0.0
        T = float(np.clip(xi[:, : L - 1][valid_prev].sum() / denom_T, 0.001, 0.6)) if denom_T > 0 else T
        G = float(np.clip((g0 * Y * M).sum() / max(1e-9, (g0 * M).sum()), 0.01, BKT_MAX_GUESS))
        S = float(np.clip((g1 * (1 - Y) * M).sum() / max(1e-9, (g1 * M).sum()), 0.01, BKT_MAX_SLIP))
    return {"p_init": round(L0, 4), "p_transit": round(T, 4), "p_guess": round(G, 4), "p_slip": round(S, 4), "sequences": n}


def bkt_one_step_predictions(seqs: list[np.ndarray], prm: dict) -> np.ndarray:
    out = []
    L0, T, G, S = prm["p_init"], prm["p_transit"], prm["p_guess"], prm["p_slip"]
    for s in seqs:
        Lk = L0
        for y in s:
            p = Lk * (1 - S) + (1 - Lk) * G
            out.append(p)
            post = Lk * (1 - S) / p if y == 1 else Lk * S / (1 - p)
            Lk = post + (1 - post) * T
    return np.array(out)


def _chapter_sequences(df: pd.DataFrame) -> dict[str, list[np.ndarray]]:
    out: dict[str, list[np.ndarray]] = {}
    for (ch, _), g in df.groupby(["chapter", "student_id"], sort=False):
        out.setdefault(ch, []).append(g["is_correct"].to_numpy(dtype=float))
    return out


def fit_and_evaluate_bkt(train: pd.DataFrame, test: pd.DataFrame, prior: dict) -> dict:
    tr = _chapter_sequences(train)
    te = _chapter_sequences(test)
    fitted: dict[str, dict] = {}
    for ch, seqs in tr.items():
        if len(seqs) >= MIN_SEQUENCES_PER_CHAPTER:
            fitted[ch] = fit_bkt_em(seqs, prior.get(ch))
    y_all, p_prior, p_fit = [], [], []
    default = {"p_init": 0.4, "p_transit": 0.15, "p_guess": 0.2, "p_slip": 0.1}
    for ch, seqs in te.items():
        pr = {k: float(prior.get(ch, default).get(k, default[k])) for k in default}
        ft = fitted.get(ch, pr)
        y_all.append(np.concatenate(seqs))
        p_prior.append(bkt_one_step_predictions(seqs, pr))
        p_fit.append(bkt_one_step_predictions(seqs, ft))
    if not y_all:
        return {"fitted_chapters": fitted, "evaluation": {"available": False}}
    y = np.concatenate(y_all)
    return {
        "fitted_chapters": fitted,
        "evaluation": {
            "available": True,
            "protocol": "held-out students; each answer predicted from the same chapter's earlier answers (one-step-ahead), plain chapter parameters without telemetry modulation",
            "chapters_fitted": len(fitted),
            "predicted_responses": int(len(y)),
            "models": {"prior_bkt": _metrics(y, np.concatenate(p_prior)), "fitted_bkt": _metrics(y, np.concatenate(p_fit))},
        },
    }


def validate_difficulty_labels(fitted: dict) -> dict:
    meta = _bank_metadata()
    v21 = {}
    if V21_PROPOSAL_PATH.exists():
        v21 = {r["question_id"]: r["proposed_tier"] for r in json.loads(V21_PROPOSAL_PATH.read_text(encoding="utf-8")).get("items", [])}
    rows = []
    for q, p in fitted.items():
        m = meta.get(q, {})
        rows.append({"b": p["b"], "p_value": p["p_value"], "tier": TIER_ORDER.get(m.get("tier", "")), "source": m.get("source"), "v21": TIER_ORDER.get(v21.get(q, ""))})
    df = pd.DataFrame(rows)
    out: dict = {"items": int(len(df))}
    for name, sub, col in (
        ("pyq_heuristic_tier_vs_fitted_b", df[df["source"] == "pyq"], "tier"),
        ("human_verified_tier_vs_fitted_b", df[df["source"] == "sample_paper"], "tier"),
        ("v21_predicted_tier_vs_fitted_b_on_pyq", df[df["source"] == "pyq"], "v21"),
    ):
        sub = sub.dropna(subset=[col])
        if len(sub) >= 20 and sub[col].nunique() > 1:
            rho, pv = spearmanr(sub[col], sub["b"])
            out[name] = {"n": int(len(sub)), "spearman_rho": round(float(rho), 4), "p_value": float(pv)}
        else:
            out[name] = {"n": int(len(sub)), "spearman_rho": None}
    return out


def evaluate_score_prediction(df: pd.DataFrame, fitted: dict, prior: dict, link: dict, n_draws: int = 500) -> dict:
    rng = np.random.default_rng(SPLIT_SEED)
    rows = []
    for sid, g in df.groupby("student_id"):
        papers = [p for p, gp in g.groupby("paper_number") if len(gp) >= 150]
        for k in range(len(papers) - 1):
            prev = g[g["paper_number"] == papers[k]]
            nxt = g[g["paper_number"] == papers[k + 1]]
            mean, sd = _eap(prev["question_id"].tolist(), prev["is_correct"].to_numpy(dtype=float), fitted, prior, link)
            abc = np.array([_params_for(q, fitted, prior, link) for q in nxt["question_id"]])
            thetas = rng.normal(mean, sd, size=n_draws)
            P = _p3pl(abc[None, :, 0], abc[None, :, 1], abc[None, :, 2], thetas[:, None])
            sims = (np.where(rng.random(P.shape) < P, 4, -1)).sum(1)
            actual = int((np.where(nxt["is_correct"].to_numpy() == 1, 4, -1)).sum())
            lo, hi = np.quantile(sims, [0.05, 0.95])
            rows.append((float(sims.mean()), actual, lo <= actual <= hi))
    if not rows:
        return {"available": False, "reason": "needs students with two or more papers of 150+ answers"}
    pred = np.array([r[0] for r in rows])
    act = np.array([r[1] for r in rows])
    return {
        "available": True,
        "pairs": len(rows),
        "mae_marks": round(float(np.abs(pred - act).mean()), 2),
        "interval_90_coverage": round(float(np.mean([r[2] for r in rows])), 4),
        "note": "coverage near 0.90 means the score interval is honest; real NEET results are still needed for exam-level calibration",
    }


def _gate(prior_m: dict, fit_m: dict) -> dict:
    gain = (prior_m["log_loss"] - fit_m["log_loss"]) / max(1e-9, prior_m["log_loss"])
    auc_ok = prior_m["auc"] is None or fit_m["auc"] is None or fit_m["auc"] >= prior_m["auc"] - GATE_MAX_AUC_DROP
    gates = {
        f"log_loss gain >= {GATE_MIN_LOGLOSS_GAIN:.0%}": gain >= GATE_MIN_LOGLOSS_GAIN,
        f"auc drop <= {GATE_MAX_AUC_DROP}": bool(auc_ok),
        f"ece <= {GATE_MAX_ECE}": fit_m["ece"] <= GATE_MAX_ECE,
    }
    return {"log_loss_gain": round(gain, 4), "gates": gates, "decision": "ELIGIBLE_PENDING_HUMAN_REVIEW" if all(gates.values()) else "HOLD"}


def run_student_response_training(csv_path: Path | None = None, output_dir: Path | None = None, label: str = "real") -> dict:
    out_dir = output_dir or OUTPUT_DIR
    df, source = load_attempts(csv_path)
    ready = data_readiness(df)
    report: dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_label": label,
        "source": source,
        "readiness": ready,
    }
    if not ready["ready"]:
        report["status"] = "SKIPPED_INSUFFICIENT_DATA"
        report["next_step"] = "collect initial-assessment attempts until every readiness check is true, then re-run"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / REPORT_NAME).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    train_ids, test_ids = split_students(df)
    train = df[df["student_id"].isin(train_ids)]
    test = df[df["student_id"].isin(test_ids)]
    prior = _prior_irt()
    irt = fit_irt_3pl_mml(train, prior)
    fitted = irt["items"]
    link = link_prior_to_fitted(fitted, prior)
    irt_eval = evaluate_irt(train, test, fitted, prior, link)
    bkt = fit_and_evaluate_bkt(train, test, _prior_bkt())
    report.update(
        {
            "status": "TRAINED",
            "split": {"train_students": len(train_ids), "test_students": len(test_ids), "seed": SPLIT_SEED},
            "irt_3pl": {
                "method": f"marginal maximum likelihood EM, {QUAD_POINTS}-point quadrature, N(0,1) ability, priors log a~N(0,{PRIOR_LOG_A_SD}^2), b~N(0,{PRIOR_B_SD}^2), c~Beta({PRIOR_C_ALPHA:g},{PRIOR_C_BETA:g})",
                "items_fitted": len(fitted),
                "marginal_loglik_trace": irt["marginal_loglik_trace"],
                "prior_to_fitted_linking": link,
                "evaluation": irt_eval,
                "promotion": _gate(irt_eval["models"]["prior_irt"], irt_eval["models"]["fitted_irt"]) if irt_eval.get("available") else None,
            },
            "bkt": {
                "method": "Baum-Welch EM per chapter, no forgetting, guess and slip capped at 0.40",
                "evaluation": bkt["evaluation"],
                "promotion": _gate(bkt["evaluation"]["models"]["prior_bkt"], bkt["evaluation"]["models"]["fitted_bkt"]) if bkt["evaluation"].get("available") else None,
            },
            "difficulty_label_validation": validate_difficulty_labels(fitted),
            "score_prediction_check": evaluate_score_prediction(test, fitted, prior, link),
            "not_trained_here": {
                "dina": "needs a validated Q-matrix before slip/guess can be re-estimated",
                "linucb": "offline evaluation needs logged selection propensities; log them before re-fitting the bandit",
            },
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / FITTED_IRT_NAME).write_text(json.dumps({"linking": link, "items": fitted}, indent=1), encoding="utf-8")
    prior_bkt_doc = json.loads(PRIOR_BKT_PATH.read_text(encoding="utf-8")) if PRIOR_BKT_PATH.exists() else {"chapters": {}}
    merged = {ch: dict(v) for ch, v in prior_bkt_doc.get("chapters", {}).items()}
    for ch, p in bkt["fitted_chapters"].items():
        merged.setdefault(ch, {}).update({k: p[k] for k in ("p_init", "p_transit", "p_guess", "p_slip")})
        merged[ch]["fitted_sequences"] = p["sequences"]
    fitted_bkt_doc = dict(prior_bkt_doc)
    fitted_bkt_doc.update({"model_type": "EM_Fitted_65Chapter_BKT", "chapters": merged, "fitted_chapters": len(bkt["fitted_chapters"])})
    (out_dir / FITTED_BKT_NAME).write_text(json.dumps(fitted_bkt_doc, indent=1), encoding="utf-8")
    (out_dir / REPORT_NAME).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def simulate_attempts(n_students: int = 500, n_items: int = 150, per_student: int = 120, seed: int = 7) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    meta = _bank_metadata()
    pool = sorted(q for q, m in meta.items() if m.get("source") == "pyq")
    chosen = rng.choice(pool, size=min(n_items, len(pool)), replace=False)
    truth = {
        q: {"a": float(np.exp(rng.normal(0.1, 0.3))), "b": float(rng.normal(0.0, 1.0)), "c": float(rng.beta(5, 17))}
        for q in chosen
    }
    rows = []
    t0 = pd.Timestamp("2027-01-01", tz="UTC")
    for s in range(n_students):
        theta = rng.normal()
        qs = rng.choice(chosen, size=min(per_student, len(chosen)), replace=False)
        for k, q in enumerate(qs):
            p = _p3pl(truth[q]["a"], truth[q]["b"], truth[q]["c"], theta)
            rows.append({"student_id": f"SIM{s:05d}", "question_id": q, "paper_number": 1 + k // 60, "is_correct": int(rng.random() < p), "timestamp": t0 + pd.Timedelta(minutes=k + 1000 * s)})
    return pd.DataFrame(rows), truth


def run_simulation_self_test(output_dir: Path) -> dict:
    sim, truth = simulate_attempts()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv = output_dir / "simulated_attempts.csv"
    sim.to_csv(csv, index=False)
    global MIN_STUDENTS, MIN_TOTAL_RESPONSES
    saved = (MIN_STUDENTS, MIN_TOTAL_RESPONSES)
    MIN_STUDENTS, MIN_TOTAL_RESPONSES = 100, 10000
    try:
        rep = run_student_response_training(csv_path=csv, output_dir=output_dir, label="SIMULATED_PIPELINE_SELF_TEST")
    finally:
        MIN_STUDENTS, MIN_TOTAL_RESPONSES = saved
    fitted = json.loads((output_dir / FITTED_IRT_NAME).read_text(encoding="utf-8"))["items"]
    common = [q for q in truth if q in fitted]
    rec = {
        k: round(float(np.corrcoef([truth[q][k] for q in common], [fitted[q][k] for q in common])[0, 1]), 4)
        for k in ("a", "b", "c")
    }
    return {
        "label": "SIMULATED data, used only to verify the estimation code; not a model result",
        "items_recovered": len(common),
        "parameter_recovery_correlation": rec,
        "held_out_metrics": rep["irt_3pl"]["evaluation"]["models"],
        "status": rep["status"],
    }


def promote(component: str) -> dict:
    rep_path = OUTPUT_DIR / REPORT_NAME
    if not rep_path.exists():
        raise RuntimeError("no student training report; run training first")
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    if rep.get("data_label") != "real" or rep.get("status") != "TRAINED":
        raise RuntimeError("only models trained on real attempts can be promoted")
    section = {"irt": "irt_3pl", "bkt": "bkt"}[component]
    decision = (rep[section].get("promotion") or {}).get("decision")
    if decision != "ELIGIBLE_PENDING_HUMAN_REVIEW":
        raise RuntimeError(f"{component} did not pass its gates (decision={decision})")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = BACKUP_DIR / stamp
    backup.mkdir(parents=True, exist_ok=True)
    if component == "irt":
        shutil.copy2(PRIOR_IRT_PATH, backup / PRIOR_IRT_PATH.name)
        fitted_doc = json.loads((OUTPUT_DIR / FITTED_IRT_NAME).read_text(encoding="utf-8"))
        link, fitted = fitted_doc["linking"], fitted_doc["items"]
        current = json.loads(PRIOR_IRT_PATH.read_text(encoding="utf-8"))
        merged = {}
        for qid, p in current.items():
            if qid in fitted:
                merged[qid] = {"a": fitted[qid]["a"], "b": fitted[qid]["b"], "c": fitted[qid]["c"]}
            else:
                merged[qid] = {"a": round(p["a"] / link["A"], 4), "b": round(link["A"] * p["b"] + link["B"], 4), "c": p["c"]}
        PRIOR_IRT_PATH.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    else:
        shutil.copy2(PRIOR_BKT_PATH, backup / PRIOR_BKT_PATH.name)
        shutil.copy2(OUTPUT_DIR / FITTED_BKT_NAME, PRIOR_BKT_PATH)
    from .training_and_validation_pipeline import REGISTRY_MANIFEST_FILE, build_model_registry_manifest

    REGISTRY_MANIFEST_FILE.write_text(json.dumps(build_model_registry_manifest(), indent=2), encoding="utf-8")
    return {"promoted": component, "backup": str(backup.relative_to(PROJECT_ROOT)), "next_step": "run python -m model.run_full_ml_pipeline and the test suites, then commit"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--promote", choices=["irt", "bkt"])
    args = parser.parse_args()
    if args.promote:
        print(json.dumps(promote(args.promote), indent=2))
        return 0
    if args.self_test:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            print(json.dumps(run_simulation_self_test(Path(tmp)), indent=2))
        return 0
    rep = run_student_response_training(csv_path=args.csv)
    print(json.dumps({k: rep[k] for k in ("status", "readiness")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
