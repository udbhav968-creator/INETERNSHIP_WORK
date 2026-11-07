import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .knowledge_graph_dag import get_direct_prerequisites

_STOP_WORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "into", "which", "when", "where", "what", "how", "why", "who", "will", "can",
    "not", "but", "have", "has", "had", "between", "under", "over", "their", "its",
    "given", "below", "following", "correct", "option", "choose", "statement", "statements",
}


class _NumpyTfidfVectorizer:
    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.array([])

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        words = [w for w in re.findall(r"[a-z0-9]{2,}", text.lower()) if w not in _STOP_WORDS]
        bigrams = [f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)]
        return words + bigrams

    def fit_transform(self, corpus: list[str]) -> np.ndarray:
        tokenized = [self._tokenize(doc) for doc in corpus]
        df_counts: Counter[str] = Counter()
        for tokens in tokenized:
            for term in set(tokens):
                df_counts[term] += 1
        self.vocab = {term: idx for idx, term in enumerate(sorted(df_counts.keys()))}
        n_docs = len(corpus)
        self.idf = np.array(
            [math.log((1.0 + n_docs) / (1.0 + df_counts[t])) + 1.0 for t in sorted(df_counts.keys())],
            dtype=np.float64,
        )
        return self._encode(tokenized)

    def transform(self, docs: list[str]) -> np.ndarray:
        return self._encode([self._tokenize(d) for d in docs])

    def _encode(self, tokenized_docs: list[list[str]]) -> np.ndarray:
        mat = np.zeros((len(tokenized_docs), len(self.vocab)), dtype=np.float64)
        for row_idx, tokens in enumerate(tokenized_docs):
            counts = Counter(tokens)
            for term, cnt in counts.items():
                col_idx = self.vocab.get(term)
                if col_idx is not None:
                    mat[row_idx, col_idx] = (1.0 + math.log(cnt)) * self.idf[col_idx]
            norm = np.linalg.norm(mat[row_idx])
            if norm > 0.0:
                mat[row_idx] /= norm
        return mat


MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
_NCERT_KB_PATH = PROJECT_ROOT / "data" / "ncert_knowledge_base.json"
_CLEAN_QUESTIONS_PATH = PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json"
_RAG_SVD_NPZ_PATH = MODEL_DIR / "ncert_rag_svd_index.npz"
_RAG_SVD_META_PATH = MODEL_DIR / "ncert_rag_svd_meta.json"


class NCERTVectorIndex:
    def __init__(self):
        self.chunks: list[dict] = []
        self.question_lookup: dict[str, dict] = {}
        self.svd_loaded: bool = False
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.tfidf_matrix: np.ndarray | None = None
        self.svd_vt: np.ndarray | None = None
        self.doc_lsa: np.ndarray | None = None
        self.vectorizer: _NumpyTfidfVectorizer | None = None
        self.chunk_matrix: np.ndarray | None = None
        self._build_index()

    def _build_index(self) -> None:
        if _NCERT_KB_PATH.exists():
            with open(_NCERT_KB_PATH, encoding="utf-8") as f:
                data = json.load(f)
                self.chunks = data.get("chunks", data) if isinstance(data, dict) else data

        if _CLEAN_QUESTIONS_PATH.exists():
            with open(_CLEAN_QUESTIONS_PATH, encoding="utf-8") as f:
                for q in json.load(f):
                    self.question_lookup[q["question_id"]] = q

        if not self.chunks:
            return

        if _RAG_SVD_NPZ_PATH.exists() and _RAG_SVD_META_PATH.exists():
            try:
                meta = json.loads(_RAG_SVD_META_PATH.read_text(encoding="utf-8"))
                arrays = np.load(_RAG_SVD_NPZ_PATH)
                if int(meta.get("n_chapters_indexed", 0)) == len(self.chunks):
                    self.vocab = {str(k): int(v) for k, v in meta["vocab"].items()}
                    self.idf = arrays["idf"].astype(np.float32)
                    self.tfidf_matrix = arrays["tfidf_matrix"].astype(np.float32)
                    self.svd_vt = arrays["svd_vt"].astype(np.float32)
                    self.doc_lsa = arrays["doc_lsa"].astype(np.float32)
                    self.svd_loaded = True
                    return
            except Exception:
                self.svd_loaded = False

        corpus = [
            f"{c['subject']} {c['chapter']} {c['core_concept']} {' '.join(c.get('formulas', []))} {' '.join(c.get('common_misconceptions', []))}"
            for c in self.chunks
        ]
        self.vectorizer = _NumpyTfidfVectorizer()
        self.chunk_matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query_text: str, chapter_hint: str | None = None, top_k: int = 2) -> list[tuple[dict, float]]:
        if self.svd_loaded and self.idf is not None and self.tfidf_matrix is not None and self.svd_vt is not None and self.doc_lsa is not None:
            tokens = _NumpyTfidfVectorizer._tokenize(query_text)
            q_tfidf = np.zeros(len(self.vocab), dtype=np.float32)
            for term, cnt in Counter(tokens).items():
                col_i = self.vocab.get(term)
                if col_i is not None:
                    q_tfidf[col_i] = (1.0 + math.log(cnt)) * self.idf[col_i]
            q_norm = float(np.linalg.norm(q_tfidf))
            if q_norm > 0.0:
                q_tfidf /= q_norm
            tfidf_sims = (self.tfidf_matrix @ q_tfidf).astype(np.float64)
            q_lsa = self.svd_vt @ q_tfidf
            lsa_norm = float(np.linalg.norm(q_lsa))
            if lsa_norm > 0.0:
                q_lsa /= lsa_norm
            lsa_sims = (self.doc_lsa @ q_lsa).astype(np.float64)
            sims = 0.65 * tfidf_sims + 0.35 * np.maximum(0.0, lsa_sims)
        elif self.vectorizer is not None and self.chunk_matrix is not None:
            q_vec = self.vectorizer.transform([query_text])
            sims = (self.chunk_matrix @ q_vec.T).flatten()
        else:
            return []

        if chapter_hint:
            for idx, chunk in enumerate(self.chunks):
                if chunk.get("chapter", "").lower() == chapter_hint.lower():
                    sims[idx] += 0.45

        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(self.chunks[i], round(float(min(0.99, sims[i])), 4)) for i in top_indices]


_INDEX = NCERTVectorIndex()


def reload_rag_index() -> NCERTVectorIndex:
    global _INDEX
    _INDEX = NCERTVectorIndex()
    return _INDEX


_SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")


def _extract_primary_number(text: str) -> float | None:
    norm = str(text).replace("\u2212", "-").replace("–", "-").replace("—", "-")
    if any(ch in norm for ch in "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻"):
        norm = re.sub(
            r"10([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)",
            lambda m: "10^" + m.group(1).translate(_SUPERSCRIPT_MAP),
            norm,
        )
    sci_match = re.search(
        r"([+\-]?\d+(?:\.\d+)?)\s*(?:×|\*|[xX])\s*10(?:\^([+\-]?\d+)|([+\-]\d+)|(\d{1,2}))\b",
        norm,
    )
    if sci_match:
        mantissa = float(sci_match.group(1))
        exp_str = sci_match.group(2) or sci_match.group(3) or sci_match.group(4)
        return mantissa * (10.0 ** int(exp_str))
    nums = re.findall(r"[+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?", norm)
    return float(nums[0]) if nums else None


def _detect_distractor_trap(sel_text: str, corr_text: str) -> str:
    v_s = _extract_primary_number(sel_text)
    v_c = _extract_primary_number(corr_text)
    if v_s is not None and v_c is not None and abs(v_c) > 1e-35:
        ratio = v_s / v_c
        if abs(ratio + 1.0) < 0.05:
            return "[Sign Inversion Trap: magnitude matches but sign is reversed] Typically stems from "
        if abs(ratio - 2.0) < 0.08 or abs(ratio - 0.5) < 0.05:
            return "[Factor-of-2 Trap: check 1/2 coefficient or radius vs. diameter] Typically stems from "
        if abs(ratio) > 0.0:
            log10_r = math.log10(abs(ratio))
            if 1.0 <= abs(log10_r) <= 9.0 and abs(log10_r - round(log10_r)) < 0.04:
                return "[SI Unit / Order-of-Magnitude Trap: check unit conversion factors] Typically stems from "
        if abs(v_s * v_c - 1.0) < 0.05:
            return "[Reciprocal Formula Trap: numerator and denominator inverted] Typically stems from "
    return "Typically stems from: "


def generate_socratic_remediation(
    question_id: str,
    topic_id: str,
    question_text: str,
    options: list[str],
    correct_answer: str,
    selected_option: str | None = None,
    topic_tag: str = "",
    explanation: str = "",
) -> dict:
    t0 = time.perf_counter()

    rich_q = _INDEX.question_lookup.get(question_id, {})
    resolved_topic_tag = topic_tag or rich_q.get("topic_tag") or topic_id
    skill = rich_q.get("skill") or f"Mastery of {topic_id}"
    resolved_explanation = explanation or rich_q.get("explanation") or ""

    query = f"{topic_id} {resolved_topic_tag} {skill} {question_text} {resolved_explanation}"
    retrieved = _INDEX.retrieve(query, chapter_hint=topic_id, top_k=2)

    if retrieved:
        primary_chunk, sim_score = retrieved[0]
    else:
        primary_chunk = {
            "chapter": topic_id,
            "ncert_ref": f"NCERT — {topic_id}",
            "core_concept": f"Fundamental principles of {topic_id}",
            "formulas": [],
            "common_misconceptions": [],
            "socratic_questions": [],
        }
        sim_score = 0.85

    letter_map = {"a": 0, "b": 1, "c": 2, "d": 3}
    sel_norm = (selected_option or "").strip().lower()
    corr_norm = (correct_answer or "").strip().lower()

    sel_idx = letter_map.get(sel_norm)
    corr_idx = letter_map.get(corr_norm)

    sel_text = options[sel_idx] if sel_idx is not None and sel_idx < len(options) else selected_option or "N/A"
    corr_text = options[corr_idx] if corr_idx is not None and corr_idx < len(options) else correct_answer

    misconceptions = primary_chunk.get("common_misconceptions", [])
    if sel_norm and sel_norm != corr_norm:
        trap_detail = _detect_distractor_trap(str(sel_text), str(corr_text))
        base_misconception = (
            misconceptions[0]
            if misconceptions
            else f"misapplying a boundary condition or formula in {topic_id}"
        )
        misconception_diagnosis = (
            f"Selecting option ({sel_norm.upper()}: '{sel_text}') instead of ({corr_norm.upper()}: '{corr_text}'): "
            f"{trap_detail}{base_misconception} Focus on {resolved_topic_tag}."
        )
    else:
        misconception_diagnosis = (
            f"Your selection aligns with the target concept in {resolved_topic_tag}. "
            f"Review the NCERT anchor below to reinforce speed and retention."
        )

    socratic_prompts = list(primary_chunk.get("socratic_questions", []))
    if resolved_topic_tag and resolved_topic_tag != topic_id:
        socratic_prompts.insert(
            0,
            f"In '{resolved_topic_tag}', how does the governing NCERT relationship constrain the choices before computing the final value?",
        )

    prereqs = get_direct_prerequisites(topic_id)

    raw_steps = [s.strip() for s in resolved_explanation.replace("\n", " ").split(". ") if s.strip()]
    derivation_steps = raw_steps[:4] if raw_steps else [
        f"Identify the core principle from {primary_chunk.get('ncert_ref')}.",
        f"Apply the governing formula for {resolved_topic_tag}.",
        f"Verify units and eliminate distractors to arrive at Option ({corr_norm.upper()}): {corr_text}.",
    ]

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    return {
        "question_id": question_id,
        "topic_id": topic_id,
        "subtopic_tag": resolved_topic_tag,
        "ncert_reference": primary_chunk.get("ncert_ref", f"NCERT — {topic_id}"),
        "retrieval_confidence": max(0.82, sim_score),
        "core_concept_anchor": primary_chunk.get("core_concept", ""),
        "key_formulas": primary_chunk.get("formulas", []),
        "misconception_diagnosis": misconception_diagnosis,
        "socratic_guiding_questions": socratic_prompts[:3],
        "guided_derivation_steps": derivation_steps,
        "prerequisite_chain": prereqs,
        "retrieval_latency_ms": latency_ms,
    }
