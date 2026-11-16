import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .irt_3pl_model import extract_cold_start_nlp_features
from .sample_paper_ingestion import load_accepted_sample_rows

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
PYQ_PATH = PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json"
PYQ_FEATURES_CSV = PROJECT_ROOT / "data" / "training" / "question_features.csv"
V2_MODEL_PATH = MODEL_DIR / "question_difficulty_model_v2_candidate.joblib"
AUDIT_REPORT_PATH = MODEL_DIR / "difficulty_model_leakage_audit.json"

TIERS = ["basic", "medium", "advanced"]
CATEGORICAL = ["subject", "chapter", "question_type_norm"]
NUMERIC = [
    "has_diagram", "word_count", "char_length", "option_char_mean", "formula_density",
    "numerical_complexity", "negation_trap", "distractor_similarity", "statement_count",
]
SEED = 20261116

QTYPE_MAP = {
    "assertion reason": "assertion_reason",
    "match the column": "match", "match the following": "match", "match the list": "match",
    "multiple statement mcq": "statement", "statement based": "statement",
    "diagram based": "diagram",
    "numerical": "numerical", "direct formula": "numerical",
}


def _qtype(raw: str, text: str) -> str:
    key = re.sub(r"[^a-z]+", " ", str(raw or "").lower()).strip()
    if key in QTYPE_MAP:
        return QTYPE_MAP[key]
    low = str(text or "").lower()
    if "assertion" in low and "reason" in low:
        return "assertion_reason"
    if "list i" in low or "list-i" in low or "column" in low:
        return "match"
    if re.search(r"\b(?:statement|statements)\b", low):
        return "statement"
    return "single_choice"


def featurize(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        opts = [r.get(f"option_{k}") or "" for k in "abcd"]
        text = r.get("question_text") or ""
        f = extract_cold_start_nlp_features(text, opts)
        out.append(
            {
                "question_id": r["question_id"],
                "group": r.get("paper_source") or "unknown",
                "source_type": r.get("source_type", "pyq"),
                "subject": r["subject"],
                "chapter": r["chapter"],
                "question_type_norm": _qtype(r.get("question_type"), text),
                "has_diagram": 1.0 if r.get("has_diagram") else 0.0,
                "word_count": f["word_count"],
                "char_length": float(len(text)),
                "option_char_mean": float(np.mean([len(str(o)) for o in opts])),
                "formula_density": f["formula_density"],
                "numerical_complexity": f["numerical_complexity"],
                "negation_trap": f["negation_trap"],
                "distractor_similarity": f["distractor_similarity"],
                "statement_count": float(len(re.findall(r"(?:^|\s)(?:[A-E]|\([a-e]\)|[IVX]+)\.\s", text))),
                "ideal_time_sec": float(r.get("ideal_time_sec") or 0),
                "label": str(r.get("difficulty_level") or "").lower(),
            }
        )
    return pd.DataFrame(out)


def _pipe(numeric: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", "passthrough", numeric),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=4, class_weight="balanced", n_jobs=-1, random_state=SEED
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def _score(y_true, y_pred) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", labels=TIERS)), 4),
        "n": int(len(y_true)),
    }


def _majority(y) -> float:
    vals, counts = np.unique(np.asarray(y), return_counts=True)
    return round(float(counts.max() / counts.sum()), 4)


def _cv(df: pd.DataFrame, numeric: list[str], grouped: bool) -> dict:
    y = df["label"].to_numpy()
    if grouped:
        splitter = GroupKFold(n_splits=5)
        splits = splitter.split(df, y, groups=df["group"])
    else:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        splits = splitter.split(df, y)
    preds = np.empty(len(df), dtype=object)
    for tr, te in splits:
        m = _pipe(numeric).fit(df.iloc[tr], y[tr])
        preds[te] = m.predict(df.iloc[te])
    return {**_score(y, preds), "majority_baseline": _majority(y)}


def run_audit(save: bool = True) -> dict:
    pyq_rows = json.loads(PYQ_PATH.read_text(encoding="utf-8"))
    for r in pyq_rows:
        r["source_type"] = "pyq"
    sample_rows, _ = load_accepted_sample_rows()
    pyq = featurize(pyq_rows)
    sample = featurize(sample_rows) if sample_rows else pd.DataFrame(columns=pyq.columns)

    csv = pd.read_csv(PYQ_FEATURES_CSV)
    tier_time = csv.groupby("difficulty_level")["ideal_time_sec"].agg(["min", "max"]).round(1).to_dict(orient="index")
    pyq_sources = json.loads(PYQ_PATH.read_text(encoding="utf-8"))
    provenance = {
        "pyq_ideal_time_source": sorted({str(r.get("ideal_time_source")) for r in pyq_sources}),
        "pyq_difficulty_source": sorted({str(r.get("difficulty_source")) for r in pyq_sources}),
        "pyq_ideal_time_range_by_tier": tier_time,
    }

    report: dict = {
        "audit": "difficulty-classifier leakage and external validity",
        "provenance": provenance,
        "pyq_with_ideal_time_5fold": _cv(pyq, NUMERIC + ["ideal_time_sec"], grouped=False),
        "pyq_without_ideal_time_5fold": _cv(pyq, NUMERIC, grouped=False),
    }
    if len(sample):
        m = _pipe(NUMERIC).fit(pyq, pyq["label"])
        ext = m.predict(sample)
        report["pyq_trained_tested_on_verified_sample_tiers"] = {
            **_score(sample["label"], ext),
            "majority_baseline": _majority(sample["label"]),
        }
        report["sample_only_grouped_by_paper_5fold"] = _cv(sample, NUMERIC, grouped=True)
        both = pd.concat([pyq, sample], ignore_index=True)
        report["v2_pyq_plus_sample_grouped_by_paper_5fold"] = _cv(both, NUMERIC, grouped=True)
        v2 = _pipe(NUMERIC).fit(sample, sample["label"])
        if save:
            joblib.dump(v2, V2_MODEL_PATH, compress=9)
        pyq_relabel = v2.predict(pyq)
        report["pyq_heuristic_tier_vs_v2_prediction"] = {
            "agreement": round(float(np.mean(pyq_relabel == pyq["label"].to_numpy())), 4),
            "crosstab": pd.crosstab(pyq["label"], pyq_relabel).to_dict(),
        }
        report["v2_candidate"] = {
            "artifact": V2_MODEL_PATH.name,
            "trained_on": "verified sample-paper tiers only (PYQ tiers are a text-complexity heuristic and conflict with verified labels)",
            "expected_accuracy": report["sample_only_grouped_by_paper_5fold"],
            "train_rows": int(len(sample)),
            "features": CATEGORICAL + NUMERIC,
            "excluded_features": ["ideal_time_sec", "marks", "negative_marks", "topic_tag"],
            "status": "candidate only; production question_difficulty_model.joblib is unchanged",
        }
    report["conclusion"] = (
        "ideal_time_sec is generated from the tier, so accuracy with it measures the generator, not difficulty; "
        "accuracy without it is the honest cold-start estimate"
    )
    if save:
        AUDIT_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_audit(), indent=2))
