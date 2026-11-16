import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .chapter_resolver import ChapterResolver
from .irt_3pl_model import CALIBRATED_ITEMS, predict_cold_start_difficulty_and_irt

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
TAGGED_DIR = PROJECT_ROOT / "data" / "tagged_questions"
PYQ_BANK_PATH = TAGGED_DIR / "neet_questions_clean.json"
SAMPLE_BANK_PATH = TAGGED_DIR / "neet_samples_001_to_101_verified.json"
SAMPLE_PARAMS_PATH = MODEL_DIR / "sample_paper_irt_parameters.json"
LINKING_PATH = MODEL_DIR / "cold_start_irt_linking.json"
INGESTION_REPORT_PATH = MODEL_DIR / "sample_paper_ingestion_report.json"
REJECTED_PATH = PROJECT_ROOT / "data" / "quarantine" / "sample_papers_rejected.json"

SUBJECTS = ("Physics", "Chemistry", "Biology")
TIERS = ("basic", "medium", "advanced")
DIFFICULTY_MAP = {
    "basic": "basic", "base": "basic", "easy": "basic",
    "medium": "medium", "intermediate": "medium", "moderate": "medium",
    "advanced": "advanced", "hard": "advanced", "difficult": "advanced",
}
REQUIRED_FIELDS = (
    "question_id", "subject", "chapter", "difficulty_level", "question_text",
    "option_a", "option_b", "option_c", "option_d", "correct_option",
)


def _norm_key(text: str) -> str:
    s = str(text or "").lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _content_hash(row: dict) -> str:
    parts = [_norm_key(row.get(k, "")) for k in ("question_text", "option_a", "option_b", "option_c", "option_d")]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:20]


def _row_hash(row: dict) -> str:
    keys = ("question_text", "option_a", "option_b", "option_c", "option_d", "subject", "chapter", "difficulty_level", "has_diagram", "ideal_time_sec", "question_type")
    return hashlib.sha256(json.dumps([row.get(k) for k in keys], sort_keys=True, default=str).encode("utf-8")).hexdigest()[:20]


def _load_json(path: Path, default):
    if not Path(path).exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def canonical_chapters(pyq_rows: list[dict]) -> dict[str, dict[str, str]]:
    table: dict[str, dict[str, str]] = defaultdict(dict)
    for r in pyq_rows:
        table[r["subject"]][_norm_key(r["chapter"])] = r["chapter"]
    return table


def _cold_start(row: dict) -> dict:
    opts = [row.get("option_a"), row.get("option_b"), row.get("option_c"), row.get("option_d")]
    res = predict_cold_start_difficulty_and_irt(
        question_text=row.get("question_text", ""),
        options=opts,
        has_diagram=bool(row.get("has_diagram")),
        subject=row.get("subject", "Physics"),
        chapter=row.get("chapter", ""),
        topic_tag=row.get("topic_tag") or row.get("chapter", ""),
        question_type=row.get("question_type"),
        ideal_time_sec=int(row.get("ideal_time_sec") or 90),
    )
    return res["irt_params"]


def fit_linking_constants(pyq_rows: list[dict] | None = None, save: bool = True) -> dict:
    pyq_rows = pyq_rows if pyq_rows is not None else _load_json(PYQ_BANK_PATH, [])
    cells: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"pred_b": [], "cal_b": [], "pred_a": [], "cal_a": []})
    for r in pyq_rows:
        cal = CALIBRATED_ITEMS.get(r["question_id"])
        tier = DIFFICULTY_MAP.get(str(r.get("difficulty_level", "")).strip().lower())
        if cal is None or tier is None:
            continue
        pred = _cold_start(r)
        for key in (f"{r['subject']}::{tier}", f"*::{tier}"):
            cells[key]["pred_b"].append(pred["b"])
            cells[key]["cal_b"].append(float(cal["b"]))
            cells[key]["pred_a"].append(pred["a"])
            cells[key]["cal_a"].append(float(cal["a"]))
    constants = {}
    for key, v in sorted(cells.items()):
        pb, cb = np.array(v["pred_b"]), np.array(v["cal_b"])
        constants[key] = {
            "n": int(len(pb)),
            "mu_pred_b": round(float(pb.mean()), 4),
            "sd_pred_b": round(float(pb.std()) or 1.0, 4),
            "mu_cal_b": round(float(cb.mean()), 4),
            "sd_cal_b": round(float(cb.std()), 4),
            "a_ratio": round(float(np.mean(v["cal_a"]) / max(1e-6, np.mean(v["pred_a"]))), 4),
        }
    all_pred, all_cal, linked = [], [], []
    for key, v in sorted(cells.items()):
        if key.startswith("*"):
            continue
        c = constants[key]
        pb = np.array(v["pred_b"])
        all_pred.extend(pb.tolist())
        all_cal.extend(v["cal_b"])
        linked.extend((c["mu_cal_b"] + (pb - c["mu_pred_b"]) * c["sd_cal_b"] / c["sd_pred_b"]).tolist())
    linked_arr, cal_arr = np.array(linked), np.array(all_cal)
    out = {
        "method": "mean-sigma linking of cold-start ridge b onto the calibrated PYQ anchor scale, per subject x verified difficulty tier",
        "anchor_items": len(all_cal),
        "raw_b_correlation": round(float(np.corrcoef(all_pred, all_cal)[0, 1]), 4),
        "linked_b_rmse": round(float(np.sqrt(np.mean((linked_arr - cal_arr) ** 2))), 4),
        "linked_b_correlation": round(float(np.corrcoef(linked_arr, cal_arr)[0, 1]), 4),
        "raw_b_rmse": round(float(np.sqrt(np.mean((np.array(all_pred) - np.array(all_cal)) ** 2))), 4),
        "cells": constants,
    }
    if save:
        _write_json(LINKING_PATH, out)
    return out


def link_params(raw: dict, subject: str, tier: str, linking: dict) -> dict:
    cells = linking["cells"]
    c = cells.get(f"{subject}::{tier}") or cells.get(f"*::{tier}")
    if not c or c["n"] < 5:
        c = cells.get(f"*::{tier}")
    b = c["mu_cal_b"] + (raw["b"] - c["mu_pred_b"]) * c["sd_cal_b"] / max(1e-6, c["sd_pred_b"])
    a = raw["a"] * c["a_ratio"]
    return {
        "a": round(float(np.clip(a, 0.5, 2.5)), 4),
        "b": round(float(np.clip(b, -3.0, 3.0)), 4),
        "c": round(float(np.clip(raw["c"], 0.10, 0.30)), 4),
    }


def pyq_cell_moments(pyq_rows: list[dict]) -> dict[str, dict[str, float]]:
    cells: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"a": [], "b": [], "c": []})
    for r in pyq_rows:
        cal = CALIBRATED_ITEMS.get(r["question_id"])
        tier = DIFFICULTY_MAP.get(str(r.get("difficulty_level", "")).strip().lower())
        if cal is None or tier is None:
            continue
        for key in (f"{r['subject']}::{tier}", f"*::{tier}"):
            for p in ("a", "b", "c"):
                cells[key][p].append(float(cal[p]))
    return {
        k: {
            "n": len(v["b"]),
            "mu_a": float(np.mean(v["a"])), "mu_b": float(np.mean(v["b"])), "sd_b": float(np.std(v["b"])),
            "mu_c": float(np.mean(v["c"])),
        }
        for k, v in cells.items()
    }


def link_cellwise(raw_items: dict[str, dict], pyq_moments: dict[str, dict]) -> tuple[dict[str, dict], dict]:
    cells: dict[str, list[str]] = defaultdict(list)
    for qid, it in raw_items.items():
        cells[f"{it['subject']}::{it['tier']}"].append(qid)
    linked: dict[str, dict] = {}
    constants = {}
    for key, qids in sorted(cells.items()):
        target = pyq_moments.get(key) or pyq_moments.get("*::" + key.split("::", 1)[1])
        ra = np.array([raw_items[q]["raw"]["a"] for q in qids])
        rb = np.array([raw_items[q]["raw"]["b"] for q in qids])
        rc = np.array([raw_items[q]["raw"]["c"] for q in qids])
        sd_rb = float(rb.std()) or 1.0
        a = np.clip(ra * target["mu_a"] / max(1e-6, float(ra.mean())), 0.5, 2.5)
        b = np.clip(target["mu_b"] + (rb - rb.mean()) * target["sd_b"] / sd_rb, -3.0, 3.0)
        c = np.clip(rc - rc.mean() + target["mu_c"], 0.10, 0.30)
        for q, av, bv, cv in zip(qids, a, b, c):
            linked[q] = {"a": round(float(av), 4), "b": round(float(bv), 4), "c": round(float(cv), 4)}
        constants[key] = {
            "n_sample": len(qids),
            "raw_mean_b": round(float(rb.mean()), 4), "raw_sd_b": round(sd_rb, 4),
            "target_mean_b": round(target["mu_b"], 4), "target_sd_b": round(target["sd_b"], 4),
            "target_mean_a": round(target["mu_a"], 4), "target_mean_c": round(target["mu_c"], 4),
        }
    return linked, constants


def validate_rows(
    rows: list[dict],
    pyq_rows: list[dict],
    assets_root: Path,
    resolver: ChapterResolver,
) -> tuple[list[dict], list[dict], dict]:
    resolutions, resolver_audit = resolver.resolve(rows)
    pyq_ids = {r["question_id"] for r in pyq_rows}
    pyq_hashes = {_content_hash(r) for r in pyq_rows}
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    accepted, rejected = [], []
    for row, res in zip(rows, resolutions):
        reasons = []
        qid = str(row.get("question_id") or "").strip()
        for k in REQUIRED_FIELDS:
            if str(row.get(k) or "").strip() == "":
                reasons.append(f"missing_{k}")
        subject = str(row.get("subject") or "").strip()
        if subject and subject not in SUBJECTS:
            reasons.append("unknown_subject")
        tier = DIFFICULTY_MAP.get(str(row.get("difficulty_level") or "").strip().lower())
        if row.get("difficulty_level") and tier is None:
            reasons.append("unknown_difficulty")
        if str(row.get("verification_status") or "").strip().lower() != "verified":
            reasons.append("not_verified")
        if str(row.get("correct_option") or "").strip().lower() not in ("a", "b", "c", "d"):
            reasons.append("bad_correct_option")
        opts = [_norm_key(row.get(f"option_{x}", "")) for x in "abcd"]
        if all(opts) and len(set(opts)) < 4:
            reasons.append("duplicate_options")
        if subject in SUBJECTS and res["chapter"] is None:
            reasons.append(res["method"])
        if row.get("has_diagram"):
            asset = str(row.get("image_asset") or "").strip()
            if not asset:
                reasons.append("diagram_without_asset")
            elif not any((base / asset).exists() for base in (assets_root, PROJECT_ROOT, TAGGED_DIR)):
                reasons.append("diagram_asset_missing")
        if qid in pyq_ids:
            reasons.append("id_collides_with_pyq")
        if qid in seen_ids:
            reasons.append("duplicate_id")
        h = _content_hash(row)
        if h in pyq_hashes:
            reasons.append("duplicate_of_pyq")
        elif h in seen_hashes:
            reasons.append("duplicate_content")
        seen_ids.add(qid)
        seen_hashes.add(h)
        if reasons:
            rejected.append(
                {
                    "question_id": qid,
                    "paper_source": row.get("paper_source"),
                    "subject": subject,
                    "source_chapter": row.get("chapter"),
                    "reasons": reasons,
                    "chapter_suggestions": res["suggestions"],
                }
            )
        else:
            clean = dict(row)
            clean["source_chapter"] = row.get("chapter")
            clean["chapter"] = res["chapter"]
            clean["chapter_method"] = res["method"]
            clean["chapter_confidence"] = res["confidence"]
            clean["difficulty_level"] = tier
            clean["source_type"] = "sample_paper"
            accepted.append(clean)
    return accepted, rejected, resolver_audit


def _cold_start_batch(rows: list[dict]) -> list[dict]:
    return [_cold_start(r) for r in rows]


def _parallel_cold_start(rows: list[dict]) -> list[dict]:
    workers = max(1, min(8, (os.cpu_count() or 1)))
    if workers == 1 or len(rows) < 200:
        return _cold_start_batch(rows)
    size = (len(rows) + workers - 1) // workers
    chunks = [rows[i:i + size] for i in range(0, len(rows), size)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_cold_start_batch, chunks))
    return [p for part in parts for p in part]


def ingest(
    sample_path: Path = SAMPLE_BANK_PATH,
    params_path: Path = SAMPLE_PARAMS_PATH,
    report_path: Path = INGESTION_REPORT_PATH,
    rejected_path: Path = REJECTED_PATH,
    assets_root: Path | None = None,
    linking: dict | None = None,
    progress: bool = False,
    resolver: ChapterResolver | None = None,
) -> dict:
    sample_path = Path(sample_path)
    if not sample_path.exists():
        report = {"status": "dataset_not_present", "expected_path": str(sample_path.relative_to(PROJECT_ROOT)) if sample_path.is_relative_to(PROJECT_ROOT) else str(sample_path)}
        _write_json(report_path, report)
        return report
    rows = _load_json(sample_path, [])
    pyq_rows = _load_json(PYQ_BANK_PATH, [])
    linking = linking or _load_json(LINKING_PATH, None) or fit_linking_constants(pyq_rows)
    assets_root = Path(assets_root) if assets_root else sample_path.parent
    if resolver is None:
        resolver = ChapterResolver(pyq_rows)
        resolver.fit(rows)
        if progress:
            print("  chapter resolver fitted", flush=True)
    accepted, rejected, resolver_audit = validate_rows(rows, pyq_rows, assets_root, resolver)

    cache = _load_json(params_path, {}).get("items", {})
    items = {}
    todo = []
    for row in accepted:
        rh = _row_hash(row) + "|" + str(row["chapter"])
        prev = cache.get(row["question_id"])
        if prev and prev.get("row_hash") == rh:
            items[row["question_id"]] = prev
        else:
            todo.append((row, rh))
    raws = _parallel_cold_start([r for r, _ in todo])
    for (row, rh), raw in zip(todo, raws):
        items[row["question_id"]] = {
            "raw": {"a": raw["a"], "b": raw["b"], "c": raw["c"]},
            "subject": row["subject"],
            "chapter": row["chapter"],
            "source_chapter": row["source_chapter"],
            "chapter_method": row["chapter_method"],
            "chapter_confidence": row["chapter_confidence"],
            "tier": row["difficulty_level"],
            "paper_source": row.get("paper_source"),
            "has_diagram": bool(row.get("has_diagram")),
            "row_hash": rh,
            "irt_source": "cold_start_ridge_tier_anchored_linking",
        }
    linked, cell_constants = link_cellwise(items, pyq_cell_moments(pyq_rows))
    for qid, params in linked.items():
        items[qid].update(params)
    recomputed = len(todo)
    items = {k: items[k] for k in sorted(items)}

    _write_json(params_path, {"source_file": sample_path.name, "item_count": len(items), "items": items})
    _write_json(rejected_path, {"source_file": sample_path.name, "rejected_count": len(rejected), "items": rejected})

    reason_counts = Counter(r for rej in rejected for r in rej["reasons"])
    b_by_tier = defaultdict(list)
    for v in items.values():
        b_by_tier[v["tier"]].append(v["b"])
    cell_counts = Counter((r["subject"], r["chapter"], r["difficulty_level"]) for r in accepted)
    report = {
        "status": "ingested",
        "source_file": sample_path.name,
        "rows_in_file": len(rows),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "rejection_reasons": dict(reason_counts),
        "cold_start_recomputed": recomputed,
        "cold_start_cached": len(items) - recomputed,
        "paper_sources": len({r.get("paper_source") for r in accepted}),
        "by_subject": dict(Counter(r["subject"] for r in accepted)),
        "by_difficulty": dict(Counter(r["difficulty_level"] for r in accepted)),
        "by_subject_difficulty": {s: dict(Counter(r["difficulty_level"] for r in accepted if r["subject"] == s)) for s in SUBJECTS},
        "with_diagrams": sum(1 for r in accepted if r.get("has_diagram")),
        "chapters_covered": len({(r["subject"], r["chapter"]) for r in accepted}),
        "chapters_missing_a_tier": sorted(
            f"{s}::{ch}::{t}"
            for s, chs in canonical_chapters(pyq_rows).items()
            for ch in chs.values()
            for t in TIERS
            if cell_counts.get((s, ch, t), 0) == 0
        ),
        "linked_b_mean_by_tier": {t: round(float(np.mean(v)), 4) for t, v in sorted(b_by_tier.items())},
        "chapter_resolution": resolver_audit,
        "accepted_by_chapter_method": dict(Counter(r["chapter_method"] for r in accepted)),
        "linking": {
            "method": "tier-anchored linking: within each subject x verified-tier cell, cold-start (a, b, c) are rescaled so the cell's mean/spread match the calibrated PYQ cell; cold-start ordering within the cell is kept",
            "assumption": "a verified difficulty tier means the same in sample papers as in PYQs; replace with empirical calibration once student responses exist",
            "cells": cell_constants,
            "cold_start_validity_on_pyq_anchors": {k: linking[k] for k in ("anchor_items", "raw_b_correlation", "raw_b_rmse", "linked_b_rmse", "linked_b_correlation")},
        },
    }
    _write_json(report_path, report)
    return report


def load_accepted_sample_rows(sample_path: Path = SAMPLE_BANK_PATH, params_path: Path = SAMPLE_PARAMS_PATH) -> tuple[list[dict], dict]:
    params = _load_json(params_path, {}).get("items", {})
    if not params or not Path(sample_path).exists():
        return [], {}
    rows = []
    taken: set[str] = set()
    for r in _load_json(sample_path, []):
        qid = str(r.get("question_id") or "").strip()
        p = params.get(qid)
        if p is None or qid in taken:
            continue
        taken.add(qid)
        clean = dict(r)
        clean["source_chapter"] = r.get("chapter")
        clean["chapter"] = p["chapter"]
        clean["chapter_method"] = p.get("chapter_method")
        clean["difficulty_level"] = p["tier"]
        clean["source_type"] = "sample_paper"
        rows.append(clean)
    return rows, params


def main(argv: list[str]) -> dict:
    if "--refit-linking" in argv or not LINKING_PATH.exists():
        lk = fit_linking_constants()
        print(f"linking: anchors={lk['anchor_items']} raw_corr={lk['raw_b_correlation']} raw_rmse={lk['raw_b_rmse']} linked_rmse={lk['linked_b_rmse']} linked_corr={lk['linked_b_correlation']}")
    path = Path(argv[argv.index("--file") + 1]) if "--file" in argv else SAMPLE_BANK_PATH
    rep = ingest(sample_path=path, progress=True)
    if rep.get("status") == "ingested" and "--no-registry" not in argv:
        from .ml_extension_registry import build_extension_registry
        from .unified_similarity_guard import build_unified_siblings

        build_unified_siblings()
        build_extension_registry()
        print("  near-duplicate map and ML extension SHA-256 registry refreshed", flush=True)
    print(json.dumps({k: rep[k] for k in rep if k not in ("chapters_missing_a_tier",)}, indent=2))
    return rep


if __name__ == "__main__":
    main(sys.argv[1:])
