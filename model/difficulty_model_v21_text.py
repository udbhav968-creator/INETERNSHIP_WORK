import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .difficulty_model_v2 import CATEGORICAL, NUMERIC, TIERS, _majority, _pipe, featurize
from .sample_paper_ingestion import load_accepted_sample_rows

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
PYQ_PATH = PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json"
V21_MODEL_PATH = MODEL_DIR / "question_difficulty_model_v21_text_candidate.joblib"
V21_REPORT_PATH = MODEL_DIR / "difficulty_model_v21_report.json"
PYQ_RELABEL_PROPOSAL_PATH = MODEL_DIR / "pyq_difficulty_relabel_proposal.json"

LR_C = 2.0
N_FOLDS = 5
NEAR_DUP_COSINE = 0.80
ECE_BINS = 10


_V21_MODEL_CACHE: dict = {}


def predict_tier_from_text(
    question_text: str,
    options: list[str] | None,
    subject: str,
    chapter: str,
    has_diagram: bool = False,
    question_type: str | None = None,
) -> dict | None:
    if "model" not in _V21_MODEL_CACHE:
        try:
            _V21_MODEL_CACHE["model"] = joblib.load(V21_MODEL_PATH) if V21_MODEL_PATH.exists() else None
        except Exception:
            _V21_MODEL_CACHE["model"] = None
    model = _V21_MODEL_CACHE["model"]
    if model is None:
        return None
    opts = list(options or []) + ["", "", "", ""]
    row = {
        "question_id": "cold_start",
        "question_text": question_text or "",
        "option_a": opts[0],
        "option_b": opts[1],
        "option_c": opts[2],
        "option_d": opts[3],
        "subject": subject,
        "chapter": chapter,
        "has_diagram": bool(has_diagram),
        "question_type": question_type or "",
    }
    proba = _ordered_proba(model, featurize_with_text([row]))[0]
    probs = {t: round(float(p), 4) for t, p in zip(TIERS, proba)}
    return {
        "tier": max(probs, key=probs.get),
        "probabilities": probs,
        "model": V21_MODEL_PATH.name,
        "trained_on": "human-verified sample-paper tiers",
    }


def question_text_blob(row: dict) -> str:
    opts = " | ".join(str(row.get(f"option_{k}") or "") for k in "abcd")
    return f"{row.get('question_text') or ''} || {opts}"


def featurize_with_text(rows: list[dict]) -> pd.DataFrame:
    df = featurize(rows)
    df["text"] = [question_text_blob(r) for r in rows]
    return df


def build_v21_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True), "text"),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=80000, sublinear_tf=True), "text"),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    )
    clf = LogisticRegression(C=LR_C, max_iter=3000, class_weight="balanced")
    return Pipeline([("pre", pre), ("clf", clf)])


def _ordered_proba(model, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    cls = list(model.classes_)
    return proba[:, [cls.index(t) for t in TIERS]]


def expected_calibration_error(P: np.ndarray, y_idx: np.ndarray, bins: int = ECE_BINS) -> float:
    conf = P.max(axis=1)
    acc = (P.argmax(axis=1) == y_idx).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - acc[m].mean())
    return float(ece)


def _metrics(P: np.ndarray, y_idx: np.ndarray) -> dict:
    pred = P.argmax(axis=1)
    return {
        "accuracy": round(float(accuracy_score(y_idx, pred)), 4),
        "macro_f1": round(float(f1_score(y_idx, pred, average="macro", labels=[0, 1, 2])), 4),
        "log_loss": round(float(log_loss(y_idx, np.clip(P, 1e-9, 1.0), labels=[0, 1, 2])), 4),
        "ece_top_label": round(expected_calibration_error(P, y_idx), 4),
        "n": int(len(y_idx)),
    }


def _nearest_train_similarity(df: pd.DataFrame, splits: list) -> np.ndarray:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit(df["text"])
    X = vec.transform(df["text"])
    sim = np.zeros(len(df))
    for tr, te in splits:
        nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(X[tr])
        dist, _ = nn.kneighbors(X[te])
        sim[te] = 1.0 - dist[:, 0]
    return sim


def train_and_evaluate_v21(save: bool = True) -> dict:
    rows, _ = load_accepted_sample_rows()
    if not rows:
        raise RuntimeError("No verified sample-paper rows available for v2.1 training")
    df = featurize_with_text(rows)
    labels = df["label"].to_numpy()
    y_idx = np.array([TIERS.index(v) for v in labels])
    splits = list(GroupKFold(n_splits=N_FOLDS).split(df, labels, groups=df["group"]))

    P_v2 = np.zeros((len(df), 3))
    P_v21 = np.zeros((len(df), 3))
    for tr, te in splits:
        P_v2[te] = _ordered_proba(_pipe(NUMERIC).fit(df.iloc[tr], labels[tr]), df.iloc[te])
        P_v21[te] = _ordered_proba(build_v21_pipeline().fit(df.iloc[tr], labels[tr]), df.iloc[te])

    sim = _nearest_train_similarity(df, splits)
    clean = sim < NEAR_DUP_COSINE

    report = {
        "model": "difficulty v2.1 text candidate",
        "labels": "verified sample-paper tiers (papers 001-101 accepted rows)",
        "evaluation": f"GroupKFold({N_FOLDS}) by paper_source; no paper appears in both train and test",
        "n_rows": int(len(df)),
        "n_papers": int(df["group"].nunique()),
        "class_balance": {k: round(float(v), 4) for k, v in pd.Series(labels).value_counts(normalize=True).items()},
        "majority_baseline": _majority(labels),
        "v2_structural_rf": _metrics(P_v2, y_idx),
        "v21_tfidf_logreg": _metrics(P_v21, y_idx),
        "near_duplicate_control": {
            "definition": f"test row whose nearest training row has word TF-IDF cosine >= {NEAR_DUP_COSINE}",
            "share_test_rows_with_near_duplicate": round(float((~clean).mean()), 4),
            "v2_structural_rf_on_clean_rows": _metrics(P_v2[clean], y_idx[clean]),
            "v21_tfidf_logreg_on_clean_rows": _metrics(P_v21[clean], y_idx[clean]),
        },
        "features": {
            "text": "question text + options: word 1-2 gram TF-IDF and char_wb 3-5 gram TF-IDF",
            "categorical": CATEGORICAL,
            "numeric": NUMERIC,
        },
        "excluded_features": ["ideal_time_sec", "marks", "negative_marks", "topic_tag"],
        "hyperparameters": {"logreg_C": LR_C, "class_weight": "balanced"},
    }

    final = build_v21_pipeline().fit(df, labels)
    pyq_rows = json.loads(PYQ_PATH.read_text(encoding="utf-8"))
    pyq_proba = _ordered_proba(final, featurize_with_text(pyq_rows))
    pyq_pred = np.array([TIERS[i] for i in pyq_proba.argmax(axis=1)])
    pyq_lab = np.array([str(r.get("difficulty_level") or "").lower() for r in pyq_rows])
    report["pyq_heuristic_tier_agreement_informational"] = round(float(np.mean(pyq_pred == pyq_lab)), 4)
    proposal = {
        "status": "PROPOSAL_NOT_APPLIED",
        "why": "stored PYQ tiers come from a text-complexity rule with kappa ~0.07 against human-verified tiers; v2.1 is trained on human-verified tiers",
        "model": V21_MODEL_PATH.name,
        "agreement_with_current_tiers": report["pyq_heuristic_tier_agreement_informational"],
        "current_distribution": {k: int(v) for k, v in pd.Series(pyq_lab).value_counts().items()},
        "proposed_distribution": {k: int(v) for k, v in pd.Series(pyq_pred).value_counts().items()},
        "domain_shift_warning": "v2.1 learned from sample papers; real PYQs may be harder, so review before adopting",
        "items": [
            {
                "question_id": r["question_id"],
                "current_tier": cur,
                "proposed_tier": str(pred),
                "proposed_probabilities": {t: round(float(p), 4) for t, p in zip(TIERS, pr)},
            }
            for r, cur, pred, pr in zip(pyq_rows, pyq_lab, pyq_pred, pyq_proba)
        ],
    }
    report["pyq_relabel_proposal"] = {k: v for k, v in proposal.items() if k != "items"}

    gain = report["near_duplicate_control"]["v21_tfidf_logreg_on_clean_rows"]["macro_f1"] - report["near_duplicate_control"]["v2_structural_rf_on_clean_rows"]["macro_f1"]
    report["decision"] = {
        "macro_f1_gain_on_clean_rows": round(float(gain), 4),
        "status": "candidate: beats v2 on duplicate-free held-out papers" if gain > 0 else "rejected: no gain over v2 on duplicate-free rows",
        "production_model_changed": False,
        "artifact": V21_MODEL_PATH.name,
    }
    if save:
        joblib.dump(final, V21_MODEL_PATH, compress=9)
        V21_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        PYQ_RELABEL_PROPOSAL_PATH.write_text(json.dumps(proposal, indent=1), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(train_and_evaluate_v21(), indent=2))
