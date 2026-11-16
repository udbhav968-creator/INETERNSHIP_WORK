from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

from .train_psychometric_and_rag_models import NCERT_KB_PATH, RAG_SVD_META_JSON, _default_rag_heldout_rows

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
CLEAN_QS_PATH = PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json"
REPORT_PATH = MODEL_DIR / "retrieval_challenger_report.json"
GATE_MIN_TOP1_GAIN = 0.02
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

Encoder = Callable[[list[str]], np.ndarray]


def chapter_documents(clean_qs: list[dict]) -> tuple[list[str], list[str]]:
    kb_raw = json.loads(NCERT_KB_PATH.read_text(encoding="utf-8"))
    chunks = kb_raw.get("chunks", kb_raw) if isinstance(kb_raw, dict) else kb_raw
    by_ch: dict[str, list[str]] = defaultdict(list)
    for q in clean_qs:
        stem, tag, expl = str(q.get("question_text", "")).strip(), str(q.get("topic_tag", "")).strip(), str(q.get("explanation", "")).strip()
        if stem or expl:
            by_ch[q["chapter"]].append(f"{stem} {tag} {expl[:240]}")
    docs, chapters = [], []
    for c in chunks:
        ch = c["chapter"]
        docs.append(
            f"{c['subject']} {ch} {c['core_concept']} {' '.join(c.get('formulas', []))} "
            f"{' '.join(c.get('common_misconceptions', []))} {' '.join(by_ch.get(ch, []))}"
        )
        chapters.append(ch)
    return docs, chapters


def _normalise(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)


class EmbeddingRetriever:
    def __init__(self, encoder: Encoder, docs: list[str], chapters: list[str], name: str, doc_prefix: str = "", query_prefix: str = ""):
        self.encoder, self.chapters, self.name, self.query_prefix = encoder, chapters, name, query_prefix
        self.doc_vecs = _normalise(encoder([doc_prefix + d for d in docs]))

    def rank(self, texts: list[str]) -> np.ndarray:
        q = _normalise(self.encoder([self.query_prefix + t for t in texts]))
        return np.argsort(-(q @ self.doc_vecs.T), axis=1)


def char_ngram_encoder(docs: list[str]) -> Encoder:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2, max_features=200000)
    vec.fit(docs)
    return lambda texts: vec.transform(texts).toarray()


def sentence_transformer_encoder(model_name: str = DEFAULT_EMBEDDING_MODEL, batch_size: int = 64) -> Encoder:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("pip install sentence-transformers torch (a GPU is recommended)") from exc
    model = SentenceTransformer(model_name)
    return lambda texts: model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)


def evaluate_retriever(retriever: EmbeddingRetriever, rows: list[dict], batch: int = 512) -> dict:
    chapter_set = set(retriever.chapters)
    rows = [r for r in rows if r.get("chapter") in chapter_set]
    top1 = top3 = 0
    for i in range(0, len(rows), batch):
        part = rows[i : i + batch]
        order = retriever.rank([str(r.get("question_text", "")) for r in part])
        for r, o in zip(part, order):
            ranked = [retriever.chapters[j] for j in o[:3]]
            top1 += ranked[0] == r["chapter"]
            top3 += r["chapter"] in ranked
    n = len(rows)
    return {
        "retriever": retriever.name,
        "eval_questions": n,
        "top1_chapter_accuracy": round(top1 / n, 4) if n else None,
        "top3_chapter_accuracy": round(top3 / n, 4) if n else None,
        "random_baseline_top1": round(1 / len(chapter_set), 4) if chapter_set else None,
    }


def compare_with_production(encoder_kind: str = "char_ngram", model_name: str = DEFAULT_EMBEDDING_MODEL, limit: int | None = None, save: bool = True) -> dict:
    clean_qs = json.loads(CLEAN_QS_PATH.read_text(encoding="utf-8"))
    docs, chapters = chapter_documents(clean_qs)
    rows = _default_rag_heldout_rows(clean_qs)
    if limit:
        rows = rows[:limit]
    if encoder_kind == "sentence_transformers":
        enc = sentence_transformer_encoder(model_name)
        prefixes = ("passage: ", "query: ") if "e5" in model_name.lower() else ("", "")
        retriever = EmbeddingRetriever(enc, docs, chapters, f"sentence_transformers:{model_name}", *prefixes)
    else:
        retriever = EmbeddingRetriever(char_ngram_encoder(docs), docs, chapters, "char_ngram_tfidf (offline stand-in, lexical)")
    challenger = evaluate_retriever(retriever, rows)
    production = json.loads(RAG_SVD_META_JSON.read_text(encoding="utf-8")).get("heldout_chapter_retrieval", {})
    gain = None
    if challenger["top1_chapter_accuracy"] is not None and production.get("top1_chapter_accuracy") is not None:
        gain = round(challenger["top1_chapter_accuracy"] - production["top1_chapter_accuracy"], 4)
    report = {
        "protocol": "same held-out sample-paper questions as the production retriever (none of them in the index)",
        "production": {"retriever": "TF-IDF + 32-d LSA hybrid", **production},
        "challenger": challenger,
        "top1_gain": gain,
        "gate": {"min_top1_gain": GATE_MIN_TOP1_GAIN},
        "comparable": limit is None,
        "decision": "ELIGIBLE_FOR_REVIEW" if (gain is not None and gain >= GATE_MIN_TOP1_GAIN and limit is None) else "HOLD",
        "how_to_run_semantic": f"python -m model.retrieval_backends --encoder sentence_transformers --model {DEFAULT_EMBEDDING_MODEL} (needs torch, sentence-transformers and model download; a GPU is recommended)",
    }
    if save:
        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate a chapter retriever against production on held-out questions")
    ap.add_argument("--encoder", choices=("char_ngram", "sentence_transformers"), default="char_ngram")
    ap.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    rep = compare_with_production(args.encoder, args.model, args.limit)
    print(json.dumps({k: rep[k] for k in ("challenger", "production", "top1_gain", "decision")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
