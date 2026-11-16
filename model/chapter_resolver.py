import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
PYQ_BANK_PATH = PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json"
RESOLVER_REPORT_PATH = MODEL_DIR / "chapter_resolver_report.json"

RESOLVER_VERSION = "chapter-resolver-v1"
TARGET_PRECISION = 0.90
MIN_THRESHOLD = 0.50
HOLDOUT_FRACTION = 0.2
OVERREPRESENTATION_RATIO = 2.5
OVERREPRESENTATION_MIN_ITEMS = 100

NCERT_ALIASES = {
    "Biology": {
        "Biodiversity & Conservation": "Ecology & Environment",
        "Ecology": "Ecology & Environment",
        "Ecosystem": "Ecology & Environment",
        "Organisms & Populations": "Ecology & Environment",
        "Biotechnology & its Applications": "Biotechnology",
        "Biotechnology: Principles & Processes": "Biotechnology",
        "Body Fluids & Circulation": "Circulation & Excretion",
        "Excretory Products & their Elimination": "Circulation & Excretion",
        "Breathing & Exchange of Gases": "Human Digestion & Respiration",
        "Digestion & Absorption": "Human Digestion & Respiration",
        "Chemical Coordination & Integration": "Locomotion & Neuro-Endocrine Control",
        "Neural Control & Coordination": "Locomotion & Neuro-Endocrine Control",
        "Locomotion & Movement": "Locomotion & Neuro-Endocrine Control",
        "Human Health & Disease": "Human Health, Disease & Microbes",
        "Microbes in Human Welfare": "Human Health, Disease & Microbes",
        "Human Reproduction": "Reproduction & Human Development",
        "Reproduction in Organisms": "Reproduction & Human Development",
        "Reproductive Health": "Reproduction & Human Development",
        "Sexual Reproduction in Flowering Plants": "Reproduction & Human Development",
        "Molecular Basis of Inheritance": "Genetics & Molecular Inheritance",
        "Principles of Inheritance & Variation": "Genetics & Molecular Inheritance",
        "Photosynthesis in Higher Plants": "Plant Physiology",
        "Respiration in Plants": "Plant Physiology",
        "Plant Growth & Development": "Plant Physiology",
        "Transport in Plants": "Plant Physiology",
        "Mineral Nutrition": "Plant Physiology",
    },
    "Chemistry": {
        "Amines": "Amines & Nitrogen Compounds",
        "Biomolecules": "Biomolecules & Polymers",
        "Polymers": "Biomolecules & Polymers",
        "Chemical Bonding & Molecular Structure": "Chemical Bonding",
        "Chemical Thermodynamics": "Thermodynamics",
        "Classification of Elements & Periodicity": "Periodic Table",
        "Classification of Elements & Periodicity in Properties": "Periodic Table",
        "Coordination Compounds": "Coordination Chemistry",
        "Electrochemistry": "Redox Reactions & Electrochemistry",
        "Redox Reactions": "Redox Reactions & Electrochemistry",
        "General Principles & Processes of Isolation of Elements": "General Principles & Metallurgy",
        "Organic Chemistry - Some Basic Principles & Techniques": "General Organic Chemistry",
        "Solid State": "States of Matter",
        "Some Basic Concepts of Chemistry": "Mole Concept / Stoichiometry",
        "The d- & f-Block Elements": "d- and f-Block Elements",
        "The p-Block Elements": "p-Block Elements",
        "p-Block Elements (Group 13 & 14)": "p-Block Elements",
        "p-Block Elements (Group 15 to 18)": "p-Block Elements",
    },
    "Physics": {
        "Atoms": "Atoms & Nuclei",
        "Nuclei": "Atoms & Nuclei",
        "Dual Nature of Radiation & Matter": "Dual Nature of Matter",
        "Electric Charges & Fields": "Electrostatics",
        "Electrostatic Potential & Capacitance": "Electrostatics",
        "Kinetic Theory of Gases": "Thermal Physics",
        "Thermal Properties of Matter": "Thermal Physics",
        "Mechanical Properties of Fluids": "Properties of Matter",
        "Mechanical Properties of Solids": "Properties of Matter",
        "Motion in a Plane": "Kinematics",
        "Motion in a Straight Line": "Kinematics",
        "Moving Charges & Magnetism": "Magnetic Effects of Current",
        "Oscillations": "Oscillations & Waves",
        "Waves": "Oscillations & Waves",
        "Ray Optics & Optical Instruments": "Ray Optics",
        "Semiconductor Electronics: Materials, Devices & Simple Circuits": "Semiconductor Electronics",
        "System of Particles & Rotational Motion": "Rotational Motion",
        "Units & Measurements": "Units & Dimensions",
    },
}


def norm_key(text: str) -> str:
    s = str(text or "").lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def content_hash(row: dict) -> str:
    parts = [norm_key(row.get(k, "")) for k in ("question_text", "option_a", "option_b", "option_c", "option_d")]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:20]


def item_text(row: dict) -> str:
    return " ".join(str(row.get(k) or "") for k in ("question_text", "option_a", "option_b", "option_c", "option_d"))


def _pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("clf", LogisticRegression(C=10.0, max_iter=3000)),
        ]
    )


class ChapterResolver:
    def __init__(self, pyq_rows: list[dict] | None = None):
        pyq_rows = pyq_rows if pyq_rows is not None else json.loads(PYQ_BANK_PATH.read_text(encoding="utf-8"))
        self.canonical: dict[str, dict[str, str]] = defaultdict(dict)
        for r in pyq_rows:
            self.canonical[r["subject"]][norm_key(r["chapter"])] = r["chapter"]
        self.aliases = {
            s: {norm_key(k): v for k, v in table.items()} for s, table in NCERT_ALIASES.items()
        }
        self.all_subject_keys = {
            norm_key(ch): s for s, table in self.canonical.items() for ch in table.values()
        }
        for s, table in NCERT_ALIASES.items():
            for k in table:
                self.all_subject_keys.setdefault(norm_key(k), s)
        self.pyq_rows = pyq_rows
        self.models: dict[str, Pipeline] = {}
        self.thresholds: dict[str, float] = {}
        self.metrics: dict[str, dict] = {}

    def label_route(self, subject: str, chapter: str) -> tuple[str, str | None]:
        key = norm_key(chapter)
        if not key or key.startswith("general "):
            return "generic", None
        if key in self.canonical.get(subject, {}):
            return "exact", self.canonical[subject][key]
        if key in self.aliases.get(subject, {}):
            return "alias", self.aliases[subject][key]
        owner = self.all_subject_keys.get(key)
        if owner and owner != subject:
            return "subject_conflict", None
        return "unknown", None

    def fit(self, sample_rows: list[dict], seed: int = 20261116) -> dict:
        rng = np.random.default_rng(seed)
        by_subject: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
        for r in self.pyq_rows:
            by_subject[r["subject"]].setdefault(content_hash(r), (item_text(r), r["chapter"]))
        for r in sample_rows:
            route, label = self.label_route(r.get("subject", ""), r.get("chapter", ""))
            if route in ("exact", "alias"):
                by_subject[r["subject"]].setdefault(content_hash(r), (item_text(r), label))
        for subject, items in sorted(by_subject.items()):
            keys = sorted(items)
            texts = [items[k][0] for k in keys]
            labels = [items[k][1] for k in keys]
            idx = rng.permutation(len(keys))
            n_hold = int(len(keys) * HOLDOUT_FRACTION)
            hold, train = idx[:n_hold], idx[n_hold:]
            m = _pipeline().fit([texts[i] for i in train], [labels[i] for i in train])
            proba = m.predict_proba([texts[i] for i in hold])
            pred = m.classes_[proba.argmax(axis=1)]
            conf = proba.max(axis=1)
            truth = np.array([labels[i] for i in hold])
            correct = pred == truth
            threshold = 1.0
            for t in np.round(np.arange(MIN_THRESHOLD, 0.99, 0.01), 2):
                sel = conf >= t
                if sel.sum() >= 20 and correct[sel].mean() >= TARGET_PRECISION:
                    threshold = float(t)
                    break
            sel = conf >= threshold
            self.thresholds[subject] = threshold
            self.metrics[subject] = {
                "train_items": int(len(train)),
                "holdout_items": int(n_hold),
                "classes": int(len(m.classes_)),
                "holdout_top1_accuracy": round(float(correct.mean()), 4),
                "confidence_threshold": threshold,
                "holdout_coverage_at_threshold": round(float(sel.mean()), 4),
                "holdout_precision_at_threshold": round(float(correct[sel].mean()) if sel.any() else 0.0, 4),
            }
            self.models[subject] = _pipeline().fit(texts, labels)
        return self.metrics

    def predict(self, rows: list[dict], top_k: int = 3) -> list[list[tuple[str, float]]]:
        out: list[list[tuple[str, float]]] = [[] for _ in rows]
        grouped: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(rows):
            if r.get("subject") in self.models:
                grouped[r["subject"]].append(i)
        for subject, idxs in grouped.items():
            m = self.models[subject]
            proba = m.predict_proba([item_text(rows[i]) for i in idxs])
            order = np.argsort(-proba, axis=1)[:, :top_k]
            for j, i in enumerate(idxs):
                out[i] = [(str(m.classes_[k]), round(float(proba[j][k]), 4)) for k in order[j]]
        return out

    def resolve(self, rows: list[dict]) -> tuple[list[dict], dict]:
        routes = [self.label_route(r.get("subject", ""), r.get("chapter", "")) for r in rows]
        need = [i for i, (route, _) in enumerate(routes) if route not in ("exact", "alias")]
        preds = self.predict([rows[i] for i in need])
        pred_of = dict(zip(need, preds))
        results = []
        for i, (r, (route, label)) in enumerate(zip(rows, routes)):
            if route in ("exact", "alias"):
                results.append({"chapter": label, "method": route, "confidence": None, "suggestions": []})
                continue
            top = pred_of.get(i, [])
            thr = self.thresholds.get(r.get("subject", ""), 1.0)
            if route != "subject_conflict" and top and top[0][1] >= thr:
                results.append({"chapter": top[0][0], "method": f"classifier_{route}", "confidence": top[0][1], "suggestions": top})
            else:
                reason = "subject_chapter_conflict" if route == "subject_conflict" else "chapter_unresolved_low_confidence"
                results.append({"chapter": None, "method": reason, "confidence": top[0][1] if top else None, "suggestions": top})
        audit = {
            "resolver_version": RESOLVER_VERSION,
            "policy": "exact or NCERT-alias labels are kept as verified by the content team; generic, unknown and conflicting labels are classified only at >= 90% held-out precision, otherwise quarantined with top-3 suggestions",
            "classifier_metrics_by_subject": self.metrics,
            "resolution_counts": dict(Counter(x["method"] for x in results)),
            "label_distribution_red_flags": self.distribution_red_flags(rows, results),
        }
        return results, audit

    def distribution_red_flags(self, rows: list[dict], results: list[dict]) -> list[dict]:
        pyq_share: dict[tuple[str, str], float] = {}
        pyq_subj = Counter(r["subject"] for r in self.pyq_rows)
        for (s, ch), n in Counter((r["subject"], r["chapter"]) for r in self.pyq_rows).items():
            pyq_share[(s, ch)] = n / pyq_subj[s]
        uniq: dict[tuple[str, str], set[str]] = defaultdict(set)
        for r, res in zip(rows, results):
            if res["chapter"]:
                uniq[(r["subject"], res["chapter"])].add(content_hash(r))
        subj_tot = Counter()
        for (s, _), hs in uniq.items():
            subj_tot[s] += len(hs)
        flags = []
        for (s, ch), hs in sorted(uniq.items()):
            share = len(hs) / max(1, subj_tot[s])
            base = pyq_share.get((s, ch), 0.0)
            ratio = share / base if base else float("inf")
            if len(hs) >= OVERREPRESENTATION_MIN_ITEMS and ratio >= OVERREPRESENTATION_RATIO:
                flags.append(
                    {
                        "subject": s,
                        "chapter": ch,
                        "unique_items": len(hs),
                        "share_in_samples": round(share, 4),
                        "share_in_pyq": round(base, 4),
                        "overrepresentation": round(ratio, 2),
                    }
                )
        return sorted(flags, key=lambda f: -f["overrepresentation"])
