import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .question_similarity_audit import get_near_duplicate_siblings_map
from .sample_paper_ingestion import load_accepted_sample_rows

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
PYQ_PATH = PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json"
UNIFIED_SIBLINGS_PATH = MODEL_DIR / "unified_near_duplicate_map.json"

NEAR_DUP_THRESHOLD = 0.85
CHUNK = 1500

_CACHE: dict[str, list[str]] | None = None


def _text(r: dict) -> str:
    return " ".join(str(r.get(k) or "") for k in ("question_text", "option_a", "option_b", "option_c", "option_d"))


def build_unified_siblings(threshold: float = NEAR_DUP_THRESHOLD, save: bool = True) -> dict:
    pyq = json.loads(PYQ_PATH.read_text(encoding="utf-8"))
    for r in pyq:
        r["source_type"] = "pyq"
    sample, _ = load_accepted_sample_rows()
    rows = pyq + sample
    ids = [r["question_id"] for r in rows]
    subj = np.array([r["subject"] for r in rows])
    src = [r.get("source_type", "pyq") for r in rows]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X = vec.fit_transform([_text(r) for r in rows])
    siblings: dict[str, set[str]] = defaultdict(set)
    pair_kinds: Counter = Counter()
    for start in range(0, X.shape[0], CHUNK):
        block = (X[start:start + CHUNK] @ X.T).tocoo()
        for i, j, v in zip(block.row, block.col, block.data):
            gi = start + int(i)
            gj = int(j)
            if gj <= gi or v < threshold or subj[gi] != subj[gj]:
                continue
            siblings[ids[gi]].add(ids[gj])
            siblings[ids[gj]].add(ids[gi])
            pair_kinds["-".join(sorted((src[gi], src[gj])))] += 1
    for qid, sibs in get_near_duplicate_siblings_map().items():
        siblings[qid].update(sibs)
    out_map = {k: sorted(v) for k, v in sorted(siblings.items())}
    report = {
        "method": "word 1-2 gram sublinear TF-IDF cosine on question text + options, same subject only",
        "threshold": threshold,
        "items_scanned": len(rows),
        "items_with_near_duplicates": len(out_map),
        "near_duplicate_pairs_by_source": dict(pair_kinds),
        "siblings": out_map,
    }
    if save:
        UNIFIED_SIBLINGS_PATH.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def get_unified_siblings_map() -> dict[str, list[str]]:
    global _CACHE
    if _CACHE is None:
        if UNIFIED_SIBLINGS_PATH.exists():
            _CACHE = json.loads(UNIFIED_SIBLINGS_PATH.read_text(encoding="utf-8"))["siblings"]
        else:
            _CACHE = get_near_duplicate_siblings_map()
    return _CACHE


if __name__ == "__main__":
    rep = build_unified_siblings()
    print({k: v for k, v in rep.items() if k != "siblings"})
