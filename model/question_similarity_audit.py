import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
CLEAN_QUESTIONS_PATH = PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json"
SIMILARITY_REPORT_PATH = MODEL_DIR / "question_similarity_report.json"

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CACHED_SIBLINGS_MAP: dict[str, list[str]] | None = None


def _extract_hybrid_features(text: str) -> list[str]:
    norm = " ".join(_TOKEN_RE.findall(text.lower()))
    words = [w for w in norm.split() if len(w) >= 2]
    bigrams = [f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)]
    compact = norm.replace(" ", "")
    char4 = [compact[i : i + 4] for i in range(0, max(0, len(compact) - 3), 2)]
    return words + bigrams + char4


def run_question_similarity_audit(
    near_dup_threshold: float = 0.85,
    high_overlap_threshold: float = 0.78,
    save_report: bool = True,
) -> dict:
    with open(CLEAN_QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    by_subject: dict[str, list[dict]] = {}
    for q in questions:
        subj = str(q.get("subject") or "Biology").strip()
        by_subject.setdefault(subj, []).append(q)

    near_duplicate_pairs: list[dict] = []
    high_overlap_pairs_count = 0
    siblings_map: dict[str, set[str]] = {}

    for subj, subj_qs in sorted(by_subject.items()):
        n_items = len(subj_qs)
        if n_items < 2:
            continue

        doc_counters: list[Counter] = []
        df_counter: Counter = Counter()
        for q in subj_qs:
            opts = " ".join(
                str(q.get(k) or "") for k in ("option_a", "option_b", "option_c", "option_d")
            )
            full_text = f"{q.get('question_text') or q.get('text') or ''} {opts}"
            feats = _extract_hybrid_features(full_text)
            c = Counter(feats)
            doc_counters.append(c)
            for term in c:
                df_counter[term] += 1

        vocab = [term for term, df in df_counter.items() if 2 <= df <= int(0.65 * n_items)]
        if not vocab:
            continue
        term_to_idx = {t: i for i, t in enumerate(vocab)}
        idf = np.array(
            [math.log((1.0 + n_items) / (1.0 + df_counter[t])) + 1.0 for t in vocab],
            dtype=np.float32,
        )

        mat = np.zeros((n_items, len(vocab)), dtype=np.float32)
        for row_i, c in enumerate(doc_counters):
            for term, tf in c.items():
                col_j = term_to_idx.get(term)
                if col_j is not None:
                    mat[row_i, col_j] = (1.0 + math.log(tf)) * idf[col_j]

        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        mat /= norms

        sim = mat @ mat.T
        upper_i, upper_j = np.where(np.triu(sim, k=1) >= high_overlap_threshold)

        for i_idx, j_idx in zip(upper_i.tolist(), upper_j.tolist()):
            score = float(sim[i_idx, j_idx])
            qa = subj_qs[i_idx]
            qb = subj_qs[j_idx]
            qid_a = str(qa["question_id"])
            qid_b = str(qb["question_id"])
            if score >= near_dup_threshold:
                siblings_map.setdefault(qid_a, set()).add(qid_b)
                siblings_map.setdefault(qid_b, set()).add(qid_a)
                near_duplicate_pairs.append(
                    {
                        "question_id_a": qid_a,
                        "question_id_b": qid_b,
                        "subject": subj,
                        "chapter_a": str(qa.get("chapter") or qa.get("topic_id") or ""),
                        "chapter_b": str(qb.get("chapter") or qb.get("topic_id") or ""),
                        "cosine_similarity": round(score, 4),
                        "snippet_a": str(qa.get("question_text") or "")[:95],
                        "snippet_b": str(qb.get("question_text") or "")[:95],
                    }
                )
            else:
                high_overlap_pairs_count += 1

    near_duplicate_pairs.sort(key=lambda r: r["cosine_similarity"], reverse=True)
    serialized_siblings = {k: sorted(v) for k, v in sorted(siblings_map.items())}

    visited: set[str] = set()
    cluster_count = 0
    for qid in serialized_siblings:
        if qid in visited:
            continue
        cluster_count += 1
        stack = [qid]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            for nb in serialized_siblings.get(cur, []):
                if nb not in visited:
                    stack.append(nb)

    report = {
        "total_audited_questions": len(questions),
        "near_duplicate_threshold": near_dup_threshold,
        "high_overlap_threshold": high_overlap_threshold,
        "near_duplicate_pairs_count": len(near_duplicate_pairs),
        "near_duplicate_clusters_count": cluster_count,
        "questions_with_near_duplicates_count": len(serialized_siblings),
        "unique_effective_question_clusters": len(questions) - len(serialized_siblings) + cluster_count,
        "high_concept_overlap_pairs_count": high_overlap_pairs_count,
        "top_near_duplicate_pairs": near_duplicate_pairs[:25],
        "near_duplicate_siblings": serialized_siblings,
    }

    if save_report:
        with open(SIMILARITY_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    global _CACHED_SIBLINGS_MAP
    _CACHED_SIBLINGS_MAP = serialized_siblings
    return report


def get_near_duplicate_siblings_map() -> dict[str, list[str]]:
    global _CACHED_SIBLINGS_MAP
    if _CACHED_SIBLINGS_MAP is not None:
        return _CACHED_SIBLINGS_MAP
    if SIMILARITY_REPORT_PATH.exists():
        with open(SIMILARITY_REPORT_PATH, encoding="utf-8") as f:
            data = json.load(f)
        _CACHED_SIBLINGS_MAP = {
            str(k): list(v) for k, v in data.get("near_duplicate_siblings", {}).items()
        }
        return _CACHED_SIBLINGS_MAP
    rep = run_question_similarity_audit(save_report=False)
    return rep["near_duplicate_siblings"]


if __name__ == "__main__":
    out = run_question_similarity_audit(save_report=True)
    summary = {k: v for k, v in out.items() if k not in ("top_near_duplicate_pairs", "near_duplicate_siblings")}
    print(json.dumps(summary, indent=2))
