from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dina_cognitive_diagnosis import COGNITIVE_SKILLS, SKILL_IDS, extract_item_q_vector
from .irt_3pl_model import D_SCALING, IRTItemParams, calibrate_question_bank, irt_3pl_probability, fisher_information
from .knowledge_graph_dag import DAG_NODES, get_all_ancestors

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
CLEAN_QS_PATH = PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json"
FEATURES_CSV_PATH = PROJECT_ROOT / "data" / "training" / "question_features.csv"
NCERT_KB_PATH = PROJECT_ROOT / "data" / "ncert_knowledge_base.json"
IMPORTANCE_PATH = MODEL_DIR / "topic_importance.json"

IRT_RIDGE_NPZ = MODEL_DIR / "irt_cold_start_ridge.npz"
BKT_CHAPTER_JSON = MODEL_DIR / "bkt_chapter_parameters.json"
DINA_CALIBRATED_JSON = MODEL_DIR / "dina_calibrated_parameters.json"
RAG_SVD_NPZ = MODEL_DIR / "ncert_rag_svd_index.npz"
RAG_SVD_META_JSON = MODEL_DIR / "ncert_rag_svd_meta.json"
LINUCB_PRIOR_JSON = MODEL_DIR / "linucb_prior_init.json"
CONFORMAL_LAPLACIAN_JSON = MODEL_DIR / "conformal_laplacian_model.json"

_STOP_WORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "into", "which", "when", "where", "what", "how", "why", "who", "will", "can",
    "not", "but", "have", "has", "had", "between", "under", "over", "their", "its",
    "given", "below", "following", "correct", "option", "choose", "statement", "statements",
}


def _gauss_hermite_nodes_and_weights(n_pts: int = 31, sigma: float = 0.85) -> tuple[np.ndarray, np.ndarray]:
    raw_x, raw_w = np.polynomial.hermite.hermgauss(n_pts)
    thetas = math.sqrt(2.0) * sigma * raw_x
    weights = raw_w / math.sqrt(math.pi)
    return thetas, weights


def train_irt_cold_start_ridge_regressor(
    df_features: pd.DataFrame,
    calibrated_items: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    feature_cols = [
        "word_count",
        "char_length",
        "formula_density",
        "numerical_complexity",
        "negation_trap",
        "distractor_similarity",
        "explanation_steps",
        "has_diagram",
        "ideal_time_sec",
    ]
    subj_dummies = pd.get_dummies(df_features["subject"], prefix="subj").reindex(
        columns=["subj_Biology", "subj_Chemistry", "subj_Physics"], fill_value=0
    )
    tier_map = {"basic": 0.0, "medium": 0.5, "advanced": 1.0}
    tier_num = df_features["difficulty_level"].map(tier_map).fillna(0.5).to_numpy(dtype=np.float64).reshape(-1, 1)

    X_raw = np.hstack(
        [
            df_features[feature_cols].to_numpy(dtype=np.float64),
            subj_dummies.to_numpy(dtype=np.float64),
            tier_num,
        ]
    )
    Y = np.array(
        [
            [
                float(calibrated_items[qid]["a"]),
                float(calibrated_items[qid]["b"]),
                float(calibrated_items[qid]["c"]),
            ]
            for qid in df_features["question_id"]
        ],
        dtype=np.float64,
    )

    n_rows, n_feats = X_raw.shape
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_rows)
    folds = np.array_split(perm, 5)

    candidate_alphas = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0]
    best_alpha = 1.0
    best_loss = float("inf")
    best_oof = np.zeros_like(Y)

    for cand_alpha in candidate_alphas:
        cand_oof = np.zeros_like(Y)
        for k in range(5):
            val_idx = folds[k]
            tr_idx = np.concatenate([folds[j] for j in range(5) if j != k])
            mu = X_raw[tr_idx].mean(axis=0)
            sd = np.where(X_raw[tr_idx].std(axis=0) < 1e-8, 1.0, X_raw[tr_idx].std(axis=0))
            X_tr = np.hstack([np.ones((len(tr_idx), 1)), (X_raw[tr_idx] - mu) / sd])
            X_va = np.hstack([np.ones((len(val_idx), 1)), (X_raw[val_idx] - mu) / sd])
            reg = cand_alpha * np.eye(n_feats + 1)
            reg[0, 0] = 0.0
            W = np.linalg.solve(X_tr.T @ X_tr + reg, X_tr.T @ Y[tr_idx])
            cand_oof[val_idx] = X_va @ W
        cand_loss = float(sum(np.sqrt(np.mean((Y[:, idx] - cand_oof[:, idx]) ** 2)) for idx in range(3)))
        if cand_loss < best_loss:
            best_loss = cand_loss
            best_alpha = cand_alpha
            best_oof = cand_oof

    oof_pred = best_oof
    ridge_alpha = float(best_alpha)

    mu_full = X_raw.mean(axis=0)
    sd_full = np.where(X_raw.std(axis=0) < 1e-8, 1.0, X_raw.std(axis=0))
    X_full = np.hstack([np.ones((n_rows, 1)), (X_raw - mu_full) / sd_full])
    reg_full = ridge_alpha * np.eye(n_feats + 1)
    reg_full[0, 0] = 0.0
    W_full = np.linalg.solve(X_full.T @ X_full + reg_full, X_full.T @ Y)

    np.savez_compressed(
        IRT_RIDGE_NPZ,
        weights=W_full.astype(np.float64),
        feature_mean=mu_full.astype(np.float64),
        feature_std=sd_full.astype(np.float64),
    )

    param_names = ("a", "b", "c")
    metrics: dict[str, Any] = {
        "n_samples": int(n_rows),
        "ridge_alpha": ridge_alpha,
        "optimal_alpha": ridge_alpha,
        "tuning_alphas_searched": candidate_alphas,
    }
    for idx, p_name in enumerate(param_names):
        rmse = float(np.sqrt(np.mean((Y[:, idx] - oof_pred[:, idx]) ** 2)))
        ss_tot = float(np.sum((Y[:, idx] - np.mean(Y[:, idx])) ** 2))
        ss_res = float(np.sum((Y[:, idx] - oof_pred[:, idx]) ** 2))
        r2 = 1.0 - (ss_res / max(1e-12, ss_tot))
        metrics[f"{p_name}_5fold_rmse"] = round(rmse, 5)
        metrics[f"{p_name}_5fold_r2"] = round(r2, 5)
    return metrics


def train_chapter_wise_bkt_parameters(
    clean_qs: list[dict[str, Any]],
    calibrated_items: dict[str, dict[str, Any]],
    topic_imp: dict[str, float],
) -> dict[str, Any]:
    thetas, gh_weights = _gauss_hermite_nodes_and_weights(n_pts=21, sigma=0.85)
    by_ch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in clean_qs:
        by_ch[q["chapter"]].append(q)

    chapters_out: dict[str, dict[str, Any]] = {}
    all_priors: list[float] = []
    weighted_sum = 0.0
    weight_total = 0.0

    for ch in sorted(DAG_NODES.keys()):
        meta = DAG_NODES[ch]
        ancestors = get_all_ancestors(ch)
        depth = len(ancestors)
        ch_items = by_ch[ch]
        imp = float(topic_imp.get(ch, 1.0))

        marginal_probs: list[float] = []
        mean_c_vals: list[float] = []
        mean_a_vals: list[float] = []
        mean_b_vals: list[float] = []
        mean_t_vals: list[float] = []

        for q in ch_items:
            ip = calibrated_items[q["question_id"]]
            a_i, b_i, c_i = float(ip["a"]), float(ip["b"]), float(ip["c"])
            p_nodes = c_i + (1.0 - c_i) / (1.0 + np.exp(-D_SCALING * a_i * (thetas - b_i)))
            marginal_p = float(np.sum(gh_weights * p_nodes))
            marginal_probs.append(marginal_p)
            mean_c_vals.append(c_i)
            mean_a_vals.append(a_i)
            mean_b_vals.append(b_i)
            mean_t_vals.append(float(q.get("ideal_time_sec", 85)))

        emp_marginal_acc = float(np.mean(marginal_probs)) if marginal_probs else 0.52
        emp_c = float(np.mean(mean_c_vals)) if mean_c_vals else 0.22
        emp_a = float(np.mean(mean_a_vals)) if mean_a_vals else 1.30
        emp_b = float(np.mean(mean_b_vals)) if mean_b_vals else 0.05
        emp_time = float(np.mean(mean_t_vals)) if mean_t_vals else 85.0

        p_slip = float(np.clip(0.145 - 0.028 * (emp_a - 1.15) + 0.008 * min(depth, 4), 0.065, 0.165))
        p_guess = float(np.clip(emp_c * 0.96, 0.16, 0.26))
        denom = max(0.25, 1.0 - p_slip - p_guess)
        raw_l0 = (emp_marginal_acc - p_guess) / denom
        depth_adj = -0.022 * min(depth, 5)
        subj_adj = 0.035 if meta.get("subject") == "Biology" else (-0.025 if meta.get("subject") == "Physics" else 0.0)
        p_init = round(float(np.clip(0.55 * raw_l0 + 0.45 * (0.38 - 0.12 * emp_b) + depth_adj + subj_adj, 0.22, 0.54)), 4)
        p_transit = round(float(np.clip(0.175 - 0.009 * (emp_time - 70.0) / 15.0 - 0.008 * min(depth, 4), 0.105, 0.215)), 4)
        half_life_h = round(float(np.clip(28.0 - 1.8 * min(depth, 5) - 2.5 * max(0.0, emp_b), 16.0, 34.0)), 2)

        chapters_out[ch] = {
            "subject": meta.get("subject", "Biology"),
            "cluster": meta.get("cluster", "General"),
            "p_init": p_init,
            "p_transit": p_transit,
            "p_guess": round(p_guess, 4),
            "p_slip": round(p_slip, 4),
            "half_life_base_hours": half_life_h,
            "mean_irt_b": round(emp_b, 4),
            "mean_irt_a": round(emp_a, 4),
            "marginal_pass_rate": round(emp_marginal_acc, 4),
            "prerequisite_depth": depth,
            "chapter_pyq_count": len(ch_items),
        }
        all_priors.append(p_init)
        weighted_sum += p_init * imp
        weight_total += imp

    weighted_mean_prior = round(weighted_sum / max(1e-9, weight_total), 4)
    payload = {
        "model_type": "EmpiricalBayes_65Chapter_GaussHermite_BKT",
        "total_chapters": len(chapters_out),
        "weighted_global_prior": weighted_mean_prior,
        "min_chapter_prior": round(min(all_priors), 4),
        "max_chapter_prior": round(max(all_priors), 4),
        "chapters": chapters_out,
    }
    BKT_CHAPTER_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def train_dina_8skill_q_matrix_parameters(
    clean_qs: list[dict[str, Any]],
    calibrated_items: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    thetas, gh_weights = _gauss_hermite_nodes_and_weights(n_pts=21, sigma=0.85)
    skill_items: dict[str, list[dict[str, Any]]] = {k: [] for k in SKILL_IDS}

    for q in clean_qs:
        q_vec = extract_item_q_vector(
            question_text=q["question_text"],
            topic_id=q["chapter"],
            difficulty=q["difficulty_level"],
        )
        for sk_id, active in q_vec.items():
            if active == 1:
                skill_items[sk_id].append(q)

    skills_calibrated: dict[str, dict[str, Any]] = {}
    for meta in COGNITIVE_SKILLS:
        sk_id = meta["skill_id"]
        items = skill_items[sk_id]
        b_vals = [float(calibrated_items[q["question_id"]]["b"]) for q in items]
        a_vals = [float(calibrated_items[q["question_id"]]["a"]) for q in items]
        c_vals = [float(calibrated_items[q["question_id"]]["c"]) for q in items]

        marg_rates = []
        for a_i, b_i, c_i in zip(a_vals, b_vals, c_vals):
            p_nodes = c_i + (1.0 - c_i) / (1.0 + np.exp(-D_SCALING * a_i * (thetas - b_i)))
            marg_rates.append(float(np.sum(gh_weights * p_nodes)))

        mean_b = float(np.mean(b_vals)) if b_vals else 0.0
        mean_a = float(np.mean(a_vals)) if a_vals else 1.3
        mean_c = float(np.mean(c_vals)) if c_vals else 0.22
        emp_pass = float(np.mean(marg_rates)) if marg_rates else 0.50

        slip_k = round(float(np.clip(0.14 - 0.025 * (mean_a - 1.2) + 0.015 * max(0.0, mean_b), 0.075, 0.175)), 4)
        guess_k = round(float(np.clip(mean_c * 0.98, 0.17, 0.26)), 4)
        prior_mastery = round(
            float(np.clip((emp_pass - guess_k) / max(0.25, 1.0 - slip_k - guess_k) * 0.65 + (0.44 - 0.10 * mean_b) * 0.35, 0.28, 0.56)),
            4,
        )
        effective_strength = 6.0
        alpha_0 = round(prior_mastery * effective_strength, 4)
        beta_0 = round((1.0 - prior_mastery) * effective_strength, 4)
        prior_sd = round(math.sqrt(prior_mastery * (1.0 - prior_mastery) / (effective_strength + 1.0)), 4)

        skills_calibrated[sk_id] = {
            "name": meta["name"],
            "corpus_pyq_count": len(items),
            "prior_mastery": prior_mastery,
            "prior_alpha": alpha_0,
            "prior_beta": beta_0,
            "prior_sd": prior_sd,
            "slip": slip_k,
            "guess": guess_k,
            "mean_irt_b": round(mean_b, 4),
        }

    q_mat = np.zeros((len(clean_qs), len(SKILL_IDS)), dtype=np.float64)
    for idx, q in enumerate(clean_qs):
        vec = extract_item_q_vector(q["question_text"], q["chapter"], q["difficulty_level"])
        for s_idx, s_id in enumerate(SKILL_IDS):
            q_mat[idx, s_idx] = vec.get(s_id, 0)
    cov_mat = np.corrcoef(q_mat, rowvar=False)
    corr_dict = {
        s_a: {s_b: round(float(cov_mat[i, j]), 4) for j, s_b in enumerate(SKILL_IDS)}
        for i, s_a in enumerate(SKILL_IDS)
    }

    payload = {
        "model_type": "DINA_8Skill_MarginalIRT_Calibrated",
        "total_corpus_items": len(clean_qs),
        "skills": skills_calibrated,
        "skill_correlation_matrix": corr_dict,
    }
    DINA_CALIBRATED_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _tokenize_rag(text: str) -> list[str]:
    words = [w for w in re.findall(r"[a-z0-9]{2,}", text.lower()) if w not in _STOP_WORDS]
    bigrams = [f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)]
    return words + bigrams


def train_ncert_and_pyq_svd_rag_index(clean_qs: list[dict[str, Any]]) -> dict[str, Any]:
    kb_raw = json.loads(NCERT_KB_PATH.read_text(encoding="utf-8"))
    chunks = kb_raw.get("chunks", kb_raw) if isinstance(kb_raw, dict) else kb_raw

    pyq_expl_by_ch: dict[str, list[str]] = defaultdict(list)
    for q in clean_qs:
        expl = str(q.get("explanation", "")).strip()
        tag = str(q.get("topic_tag", "")).strip()
        if expl:
            pyq_expl_by_ch[q["chapter"]].append(f"{tag} {expl[:240]}")

    docs: list[str] = []
    for c in chunks:
        ch_name = c["chapter"]
        sampled_expls = " ".join(pyq_expl_by_ch.get(ch_name, [])[:18])
        doc_txt = (
            f"{c['subject']} {ch_name} {c['core_concept']} "
            f"{' '.join(c.get('formulas', []))} {' '.join(c.get('common_misconceptions', []))} "
            f"{sampled_expls}"
        )
        docs.append(doc_txt)

    tokenized = [_tokenize_rag(d) for d in docs]
    df_counts: Counter[str] = Counter()
    for toks in tokenized:
        for term in set(toks):
            df_counts[term] += 1

    kept_terms = sorted(t for t, dfv in df_counts.items() if dfv >= 1)[:6000]
    vocab = {t: i for i, t in enumerate(kept_terms)}
    n_docs = len(docs)
    idf = np.array([math.log((1.0 + n_docs) / (1.0 + df_counts[t])) + 1.0 for t in kept_terms], dtype=np.float32)

    tfidf_mat = np.zeros((n_docs, len(vocab)), dtype=np.float32)
    for r_i, toks in enumerate(tokenized):
        cnts = Counter(toks)
        for term, cnt in cnts.items():
            c_i = vocab.get(term)
            if c_i is not None:
                tfidf_mat[r_i, c_i] = (1.0 + math.log(cnt)) * idf[c_i]
        norm = float(np.linalg.norm(tfidf_mat[r_i]))
        if norm > 0.0:
            tfidf_mat[r_i] /= norm

    u_mat, s_vals, vt_mat = np.linalg.svd(tfidf_mat.astype(np.float64), full_matrices=False)
    k_dim = min(32, len(s_vals))
    vt_k = vt_mat[:k_dim, :].astype(np.float32)
    doc_lsa = (u_mat[:, :k_dim] * s_vals[:k_dim]).astype(np.float32)
    lsa_norms = np.linalg.norm(doc_lsa, axis=1, keepdims=True)
    lsa_norms[lsa_norms == 0.0] = 1.0
    doc_lsa_normed = (doc_lsa / lsa_norms).astype(np.float32)

    total_var = float(np.sum(s_vals ** 2))
    explained_ratio = round(float(np.sum(s_vals[:k_dim] ** 2) / max(1e-12, total_var)), 4)
    variance_spectrum = {
        f"rank_{r}": round(float(np.sum(s_vals[:r] ** 2) / max(1e-12, total_var)), 4)
        for r in (16, 24, 32, 48)
        if r <= len(s_vals)
    }

    np.savez_compressed(
        RAG_SVD_NPZ,
        idf=idf,
        tfidf_matrix=tfidf_mat.astype(np.float16),
        svd_vt=vt_k.astype(np.float16),
        doc_lsa=doc_lsa_normed.astype(np.float16),
    )
    meta_payload = {
        "model_type": "Hybrid_SublinearTFIDF_TruncatedSVD_LSA",
        "n_chapters_indexed": n_docs,
        "vocab_size": len(vocab),
        "lsa_dimensions": k_dim,
        "lsa_explained_variance_ratio": explained_ratio,
        "svd_variance_spectrum": variance_spectrum,
        "vocab": vocab,
    }
    RAG_SVD_META_JSON.write_text(json.dumps(meta_payload, separators=(",", ":")) + "\n", encoding="utf-8")
    return {k: v for k, v in meta_payload.items() if k != "vocab"}


def train_linucb_warm_start_and_conformal_laplacian(
    clean_qs: list[dict[str, Any]],
    calibrated_items: dict[str, dict[str, Any]],
    bkt_payload: dict[str, Any],
    topic_imp: dict[str, float],
) -> dict[str, Any]:
    bkt_ch = bkt_payload["chapters"]
    X_ctx: list[list[float]] = []
    y_util: list[float] = []

    for theta_probe in (-0.8, 0.0, 0.8):
        for q in clean_qs:
            ch = q["chapter"]
            ip = calibrated_items[q["question_id"]]
            params = IRTItemParams(a=float(ip["a"]), b=float(ip["b"]), c=float(ip["c"]))
            p_succ = irt_3pl_probability(theta_probe, params)
            f_info = fisher_information(theta_probe, params)
            m_prior = float(bkt_ch[ch]["p_init"])
            imp = float(topic_imp.get(ch, 1.0))
            prereqs = DAG_NODES[ch].get("prerequisites", [])
            readiness = (
                sum(float(bkt_ch[p]["p_init"]) for p in prereqs) / len(prereqs) if prereqs else 1.0
            )

            zpd_align = max(0.0, 1.0 - abs(p_succ - 0.70) * 1.5)
            norm_info = min(1.0, f_info / 1.5)
            m_def = max(0.0, 1.0 - m_prior)
            norm_imp = min(1.0, imp / 5.5)
            r_clamped = max(0.1, min(1.0, readiness))
            pace_sig = 0.25

            ctx = [zpd_align, norm_info, m_def, norm_imp, r_clamped, pace_sig]
            pedagogical_reward = (
                0.42 * zpd_align
                + 0.26 * norm_info
                + 0.14 * m_def * r_clamped
                + 0.12 * norm_imp
                + 0.06 * r_clamped
            )
            X_ctx.append(ctx)
            y_util.append(pedagogical_reward)

    X_arr = np.array(X_ctx, dtype=np.float64)
    y_arr = np.array(y_util, dtype=np.float64)
    gram = (X_arr.T @ X_arr) / max(1.0, len(X_arr)) * 4.0 + np.eye(6, dtype=np.float64)
    theta_hat = np.linalg.solve(X_arr.T @ X_arr + 5.0 * np.eye(6), X_arr.T @ y_arr)
    b_init = gram @ theta_hat

    pred_y = X_arr @ theta_hat
    ss_res = float(np.sum((y_arr - pred_y) ** 2))
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    bandit_r2 = round(1.0 - ss_res / max(1e-12, ss_tot), 5)

    linucb_payload = {
        "model_type": "DisjointLinUCB_RidgeWarmStart",
        "context_dim": 6,
        "default_alpha": 0.35,
        "training_pairs": len(X_arr),
        "prior_fit_r2": bandit_r2,
        "A_init": np.round(gram, 6).tolist(),
        "b_init": np.round(b_init, 6).tolist(),
        "theta_prior": np.round(theta_hat, 6).tolist(),
    }
    LINUCB_PRIOR_JSON.write_text(json.dumps(linucb_payload, indent=2) + "\n", encoding="utf-8")

    chapters_sorted = sorted(DAG_NODES.keys())
    idx_map = {c: i for i, c in enumerate(chapters_sorted)}
    n_c = len(chapters_sorted)
    adj = np.zeros((n_c, n_c), dtype=np.float64)
    for c_name, meta in DAG_NODES.items():
        u = idx_map[c_name]
        for p in meta.get("prerequisites", []):
            if p in idx_map:
                v = idx_map[p]
                adj[u, v] = 1.0
                adj[v, u] = 1.0
    deg = np.sum(adj, axis=1)
    inv_sqrt_deg = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    d_inv_sqrt = np.diag(inv_sqrt_deg)
    lap_norm = np.eye(n_c) - d_inv_sqrt @ adj @ d_inv_sqrt
    eigvals, eigvecs = np.linalg.eigh(lap_norm)
    beta_diff = 0.55
    heat_kernel = eigvecs @ np.diag(np.exp(-beta_diff * np.maximum(0.0, eigvals))) @ eigvecs.T

    top_diffusion: dict[str, dict[str, float]] = {}
    for i, c_name in enumerate(chapters_sorted):
        off_diag = float(np.sum(heat_kernel[i]) - heat_kernel[i, i])
        top_diffusion[c_name] = {
            "self_kernel_weight": round(float(heat_kernel[i, i]), 4),
            "downstream_diffusion_mass": round(max(0.0, off_diag), 4),
        }

    pos_eigs = [float(ev) for ev in eigvals if float(ev) > 1e-6]
    conf_payload = {
        "model_type": "NormalizedGraphLaplacian_HeatKernel_ConformalTable",
        "n_chapters": n_c,
        "heat_beta": beta_diff,
        "connected_subject_components": int(sum(1 for ev in eigvals if float(ev) <= 1e-6)),
        "fiedler_positive_spectral_gap": round(pos_eigs[0] if pos_eigs else 0.0, 5),
        "max_eigenvalue": round(float(eigvals[-1]), 5),
        "chapter_diffusion_summary": top_diffusion,
    }
    CONFORMAL_LAPLACIAN_JSON.write_text(json.dumps(conf_payload, indent=2) + "\n", encoding="utf-8")
    return {
        "linucb_prior_r2": bandit_r2,
        "laplacian_spectral_gap": conf_payload["fiedler_positive_spectral_gap"],
    }


def train_all_auxiliary_models() -> dict[str, Any]:
    clean_qs = json.loads(CLEAN_QS_PATH.read_text(encoding="utf-8"))
    df_features = pd.read_csv(FEATURES_CSV_PATH)
    topic_imp = json.loads(IMPORTANCE_PATH.read_text(encoding="utf-8")) if IMPORTANCE_PATH.exists() else {}
    calibrated_items = calibrate_question_bank()

    irt_ridge_metrics = train_irt_cold_start_ridge_regressor(df_features, calibrated_items)
    bkt_payload = train_chapter_wise_bkt_parameters(clean_qs, calibrated_items, topic_imp)
    dina_payload = train_dina_8skill_q_matrix_parameters(clean_qs, calibrated_items)
    rag_svd_metrics = train_ncert_and_pyq_svd_rag_index(clean_qs)
    bandit_lap_metrics = train_linucb_warm_start_and_conformal_laplacian(
        clean_qs, calibrated_items, bkt_payload, topic_imp
    )

    return {
        "irt_cold_start_ridge_regressor": irt_ridge_metrics,
        "bkt_65_chapter_empirical_bayes": {
            "total_chapters": bkt_payload["total_chapters"],
            "weighted_global_prior": bkt_payload["weighted_global_prior"],
            "min_chapter_prior": bkt_payload["min_chapter_prior"],
            "max_chapter_prior": bkt_payload["max_chapter_prior"],
        },
        "dina_8_skill_calibrated": {
            "total_corpus_items": dina_payload["total_corpus_items"],
            "skill_priors": {k: v["prior_mastery"] for k, v in dina_payload["skills"].items()},
        },
        "ncert_pyq_svd_rag": rag_svd_metrics,
        "linucb_and_laplacian": bandit_lap_metrics,
    }


if __name__ == "__main__":
    summary = train_all_auxiliary_models()
    print(json.dumps(summary, indent=2))
