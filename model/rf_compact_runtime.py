from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent
COMPACT_FOREST_NPZ = MODEL_DIR / "rf_difficulty_forest.npz"
COMPACT_FOREST_META = MODEL_DIR / "rf_difficulty_meta.json"

DEFAULT_NUMERIC_FEATURES = [
    "ideal_time_sec",
    "has_diagram",
    "word_count",
    "char_length",
    "formula_density",
    "numerical_complexity",
    "negation_trap",
    "distractor_similarity",
    "marks",
    "negative_marks",
]


class CompactRandomForestRuntime:
    def __init__(self, npz_path: Path = COMPACT_FOREST_NPZ, meta_path: Path = COMPACT_FOREST_META) -> None:
        self.loaded = False
        self.classes: list[str] = []
        self.cat_maps: list[dict[str, int]] = []
        self.numeric_features: list[str] = list(DEFAULT_NUMERIC_FEATURES)
        self.numeric_offset = 0
        self.n_features = 0
        self.tree_offsets: np.ndarray | None = None
        self.children_left: np.ndarray | None = None
        self.children_right: np.ndarray | None = None
        self.feature: np.ndarray | None = None
        self.threshold: np.ndarray | None = None
        self.leaf_probs: np.ndarray | None = None

        if npz_path.exists() and meta_path.exists():
            self._load(npz_path, meta_path)

    def _load(self, npz_path: Path, meta_path: Path) -> None:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        self.classes = [str(c) for c in meta["classes"]]
        self.numeric_features = [
            str(fn)
            for fn in meta.get(
                "numeric_features",
                ["ideal_time_sec", "has_diagram", "marks", "negative_marks"],
            )
        ]
        offset = 0
        self.cat_maps = []
        for cats in meta["categories"]:
            mapping = {str(val): offset + idx for idx, val in enumerate(cats)}
            self.cat_maps.append(mapping)
            offset += len(cats)
        self.numeric_offset = offset
        self.n_features = offset + len(self.numeric_features)

        arrays = np.load(npz_path)
        self.tree_offsets = arrays["tree_offsets"]
        self.children_left = arrays["children_left"]
        self.children_right = arrays["children_right"]
        self.feature = arrays["feature"]
        self.threshold = arrays["threshold"]
        self.leaf_probs = arrays["leaf_probs"].astype(np.float64)
        self.loaded = True

    def encode_row(
        self,
        subject: str,
        chapter: str,
        topic_tag: str,
        question_type: str,
        ideal_time_sec: float,
        has_diagram: bool,
        word_count: float = 26.0,
        char_length: float = 165.0,
        formula_density: float = 0.12,
        numerical_complexity: float = 0.25,
        negation_trap: float = 0.0,
        distractor_similarity: float = 0.24,
        marks: float = 4.0,
        negative_marks: float = -1.0,
    ) -> np.ndarray:
        x = np.zeros(self.n_features, dtype=np.float32)
        for idx, val in enumerate((subject, chapter, topic_tag, question_type)):
            col = self.cat_maps[idx].get(str(val))
            if col is not None:
                x[col] = 1.0
        num_lookup = {
            "ideal_time_sec": float(ideal_time_sec),
            "has_diagram": 1.0 if has_diagram else 0.0,
            "word_count": float(word_count),
            "char_length": float(char_length),
            "formula_density": float(formula_density),
            "numerical_complexity": float(numerical_complexity),
            "negation_trap": float(negation_trap),
            "distractor_similarity": float(distractor_similarity),
            "marks": float(marks),
            "negative_marks": float(negative_marks),
        }
        base = self.numeric_offset
        for j, fn in enumerate(self.numeric_features):
            x[base + j] = num_lookup.get(fn, 0.0)
        return x.astype(np.float64)

    def predict_proba_single(
        self,
        subject: str,
        chapter: str,
        topic_tag: str,
        question_type: str,
        ideal_time_sec: float,
        has_diagram: bool,
        word_count: float = 26.0,
        char_length: float = 165.0,
        formula_density: float = 0.12,
        numerical_complexity: float = 0.25,
        negation_trap: float = 0.0,
        distractor_similarity: float = 0.24,
        marks: float = 4.0,
        negative_marks: float = -1.0,
    ) -> dict[str, float]:
        if not self.loaded or self.tree_offsets is None or self.children_left is None:
            return {}
        x = self.encode_row(
            subject=subject,
            chapter=chapter,
            topic_tag=topic_tag,
            question_type=question_type,
            ideal_time_sec=ideal_time_sec,
            has_diagram=has_diagram,
            word_count=word_count,
            char_length=char_length,
            formula_density=formula_density,
            numerical_complexity=numerical_complexity,
            negation_trap=negation_trap,
            distractor_similarity=distractor_similarity,
            marks=marks,
            negative_marks=negative_marks,
        )
        cl = self.children_left
        cr = self.children_right
        feat = self.feature
        thresh = self.threshold
        lprobs = self.leaf_probs
        offsets = self.tree_offsets
        assert cr is not None and feat is not None and thresh is not None and lprobs is not None

        roots = offsets[:-1].astype(np.int32)
        g_idx = roots.copy()
        active = np.ones(len(roots), dtype=bool)
        while np.any(active):
            act_indices = np.flatnonzero(active)
            act_g = g_idx[act_indices]
            left_c = cl[act_g]
            is_leaf = left_c < 0
            if np.any(is_leaf):
                active[act_indices[is_leaf]] = False
            non_leaf = ~is_leaf
            if not np.any(non_leaf):
                break
            nl_indices = act_indices[non_leaf]
            nl_g = act_g[non_leaf]
            go_left = x[feat[nl_g]] <= thresh[nl_g]
            next_local = np.where(go_left, cl[nl_g], cr[nl_g]).astype(np.int32)
            g_idx[nl_indices] = roots[nl_indices] + next_local
        mean_probs = np.mean(lprobs[g_idx], axis=0)
        return {cls_name: round(float(mean_probs[i]), 4) for i, cls_name in enumerate(self.classes)}


def export_rf_compact_forest(
    pipeline: Any,
    numeric_features: list[str] | None = None,
    npz_path: Path = COMPACT_FOREST_NPZ,
    meta_path: Path = COMPACT_FOREST_META,
) -> dict[str, Any]:
    ohe = pipeline.named_steps["preprocessor"].named_transformers_["categorical"]
    rf = pipeline.named_steps["model"]
    classes = [str(c) for c in rf.classes_]
    categories = [[str(v) for v in cat_arr.tolist()] for cat_arr in ohe.categories_]
    resolved_numeric = list(numeric_features) if numeric_features else list(DEFAULT_NUMERIC_FEATURES)

    offsets = [0]
    c_left_parts = []
    c_right_parts = []
    feat_parts = []
    thresh_parts = []
    probs_parts = []

    for est in rf.estimators_:
        tr = est.tree_
        n_nodes = int(tr.node_count)
        offsets.append(offsets[-1] + n_nodes)
        c_left_parts.append(tr.children_left.astype(np.int16))
        c_right_parts.append(tr.children_right.astype(np.int16))
        feat_parts.append(tr.feature.astype(np.int16))
        thresh_parts.append(tr.threshold.astype(np.float64))

        vals = tr.value[:, 0, :].astype(np.float64)
        sums = np.sum(vals, axis=1, keepdims=True)
        sums[sums == 0.0] = 1.0
        norm_probs = (vals / sums).astype(np.float16)
        probs_parts.append(norm_probs)

    np.savez_compressed(
        npz_path,
        tree_offsets=np.array(offsets, dtype=np.int32),
        children_left=np.concatenate(c_left_parts),
        children_right=np.concatenate(c_right_parts),
        feature=np.concatenate(feat_parts),
        threshold=np.concatenate(thresh_parts),
        leaf_probs=np.concatenate(probs_parts),
    )

    meta = {
        "model": "RandomForestClassifierCompactRuntime",
        "n_estimators": len(rf.estimators_),
        "total_nodes": int(offsets[-1]),
        "classes": classes,
        "categories": categories,
        "numeric_features": resolved_numeric,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


COMPACT_RF_RUNTIME = CompactRandomForestRuntime()


def reload_compact_rf_runtime() -> CompactRandomForestRuntime:
    global COMPACT_RF_RUNTIME
    COMPACT_RF_RUNTIME = CompactRandomForestRuntime()
    return COMPACT_RF_RUNTIME
