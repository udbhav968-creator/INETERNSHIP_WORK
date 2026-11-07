from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
RAW_FLAWLESS_PATH = PROJECT_ROOT / "data" / "tagged_questions" / "neet_all_papers_2013_2026_flawless.json"
CLEAN_OUT_PATH = PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json"
QUARANTINE_OUT_PATH = PROJECT_ROOT / "data" / "tagged_questions" / "neet_all_papers_2013_2026_flawless.quarantine.json"
DAG_PATH = PROJECT_ROOT / "data" / "taxonomy" / "prerequisite_dag.json"
WEIGHTAGE_PATH = PROJECT_ROOT / "data" / "taxonomy" / "NEET_Unified_Weightage_And_Marks_2013_2026.json"
CHAPTER_SUMMARY_OUT_PATH = PROJECT_ROOT / "data" / "taxonomy" / "chapter_wise_neet_pyq_summary.json"
QUESTION_FEATURES_CSV_PATH = PROJECT_ROOT / "data" / "training" / "question_features.csv"
CHAPTER_FEATURES_CSV_PATH = PROJECT_ROOT / "data" / "training" / "chapter_wise_features.csv"

LEAKED_OPTS_RE = re.compile(
    r"^(.*?)\s*(?:Select the correct option[^\n(]*|Choose the correct answer[^\n(]*|Which of the following is the correct option\??)?\s*"
    r"(?:\(a\)\s*\(b\)\s*\(c\)\s*\(d\)\s*)?"
    r"\(1\)\s*(.+?)\s*\(2\)\s*(.+?)\s*\(3\)\s*(.+?)\s*\(4\)\s*(.+)$",
    re.S | re.I,
)

EXPLICIT_OCR_FIXES: dict[str, dict[str, Any]] = {
    "NEET2013_PHY_001": {
        "question_text": (
            "In an experiment four quantities a, b, c and d are measured with percentage error 1%, 2%, 3% and 4% "
            "respectively. Quantity P is calculated as follows: P = (a^3 * b^2) / (c * d). Percentage error in P is:"
        ),
    },
    "NEET2013_PHY_002": {
        "question_text": (
            "The velocity of a projectile at the initial point A is (2i + 3j) m/s. "
            "Its velocity (in m/s) at the return point B on the same horizontal level is:"
        ),
        "option_a": "-2i - 3j",
        "option_b": "-2i + 3j",
        "option_c": "2i - 3j",
        "option_d": "2i + 3j",
    },
    "NEET2013_PHY_006": {
        "question_text": (
            "A uniform force of (3i + j) newton acts on a particle of mass 2 kg. "
            "Hence the particle is displaced from position (2i + k) meter to position (4i + 3j - k) meter. "
            "The work done by the force on the particle is:"
        ),
    },
    "NEET2018_PHY_012": {
        "question_text": (
            "An electron of mass m with an initial velocity v = v0 i (v0 > 0) enters an electric field "
            "E = -E0 i (E0 = constant > 0) at t = 0. If lambda_0 is its de-Broglie wavelength initially, "
            "then its de-Broglie wavelength at time t is:"
        ),
    },
    "NEET2018_PHY_018": {
        "subject": "Physics",
        "chapter": "Electromagnetic Waves",
        "topic_tag": "Electromagnetic Waves & Properties",
        "question_text": (
            "An electromagnetic wave is propagating in a medium with a velocity v = v i. "
            "The instantaneous oscillating electric field of this em wave is along +y axis. "
            "Then the direction of oscillating magnetic field of the em wave will be along:"
        ),
    },
    "NEET2018_PHY_036": {
        "question_text": (
            "The moment of the force F = 4i + 5j - 6k at (2, 0, -3), about the point (2, -2, -2), is given by:"
        ),
    },
    "NEET2021_PHY_044": {
        "subject": "Physics",
        "chapter": "Magnetic Effects of Current",
        "topic_tag": "Lorentz Force & Motion in Magnetic Field",
        "question_text": (
            "In the product F = q(v x B) = q v x (B i + B j + B0 k), for q = 1 and v = 2i + 4j + 6k "
            "and F = 4i - 20j + 12k, what will be the complete expression for B?"
        ),
    },
    "NEET2023_PHY_008": {
        "question_text": (
            "A 12 V, 60 W lamp is connected to the secondary of a step-down transformer, whose primary is "
            "connected to AC mains of 220 V. Assuming the transformer to be ideal, what is the current in the primary winding?"
        ),
        "option_a": "0.27 A",
        "option_b": "2.7 A",
        "option_c": "3.7 A",
        "option_d": "0.37 A",
    },
    "NEET2023_CHEM_060": {
        "chapter": "Solutions",
        "topic_tag": "Solubility & Henry's Law",
        "question_text": (
            "Given below are two statements: one is labelled as Assertion A and the other is labelled as Reason R.\n"
            "Assertion A: Helium is used to dilute oxygen in diving apparatus.\n"
            "Reason R: Helium has high solubility in O2.\n"
            "In the light of the above statements, choose the correct answer from the options given below:"
        ),
        "option_a": "Both A and R are true and R is the correct explanation of A",
        "option_b": "Both A and R are true but R is NOT the correct explanation of A",
        "option_c": "A is true but R is false",
        "option_d": "A is false but R is true",
        "correct_option": "c",
    },
    "NEET2024_CHEM_093": {
        "question_text": (
            "On complete combustion, 0.30 g of an organic compound gave 0.20 g of CO2 and 0.10 g of H2O. "
            "The percentage composition of carbon and hydrogen in the compound, respectively is:"
        ),
        "option_a": "4.07% and 15.02%",
        "option_b": "18.18% and 3.70%",
        "option_c": "15.02% and 4.07%",
        "option_d": "3.70% and 18.18%",
    },
}

EXPLICIT_CHAPTER_FIXES: dict[str, tuple[str, str, str]] = {
    "NEET2014_CHEM_082": ("Chemistry", "Environmental Chemistry", "Atmospheric Pollution & Smog"),
    "NEET2017_CHEM_071": ("Chemistry", "Environmental Chemistry", "Tropospheric Pollution & Carbon Monoxide"),
    "NEET2019_CHEM_052": ("Chemistry", "Environmental Chemistry", "Greenhouse Effect & Global Warming"),
    "NEET2023_CHEM_098": ("Chemistry", "Environmental Chemistry", "Water Pollution, BOD & Eutrophication"),
    "NEET2016_CHEM_052": ("Chemistry", "s-Block Elements", "Biological Importance of Mg and Ca"),
    "NEET2016_CHEM_071": ("Chemistry", "s-Block Elements", "Compounds of Calcium & Alkaline Earth Metals"),
    "NEET2019_CHEM_054": ("Chemistry", "s-Block Elements", "Biological Importance of Magnesium"),
    "NEET2019_CHEM_058": ("Chemistry", "p-Block Elements", "Industrial Preparation of Inorganic Compounds"),
    "NEET2019_CHEM_088": ("Chemistry", "s-Block Elements", "Hardness of Water & Clark's Method"),
    "NEET2020_CHEM_106": ("Chemistry", "Redox Reactions & Electrochemistry", "Faraday's Laws of Electrolysis"),
    "NEET2020_CHEM_130": ("Chemistry", "s-Block Elements", "Hydrides & Hardness of Water"),
    "NEET2021_CHEM_079": ("Chemistry", "s-Block Elements", "Anomalous Behaviour of Beryllium"),
    "NEET2014_PHY_029": ("Physics", "Magnetism & Matter", "Magnetic Dipole & Bar Magnet Torque"),
    "NEET2018_PHY_027": ("Physics", "Magnetism & Matter", "Diamagnetic, Paramagnetic & Ferromagnetic Substances"),
    "NEET2019_PHY_026": ("Physics", "Magnetism & Matter", "Earth's Magnetism & Angle of Dip"),
    "NEET2015_CHEM_034": ("Chemistry", "Haloalkanes & Haloarenes", "SN1 & SN2 Substitution Stereochemistry"),
    "NEET2015_CHEM_043": ("Chemistry", "Haloalkanes & Haloarenes", "Preparation of Alkyl Halides"),
    "NEET2018_CHEM_062": ("Chemistry", "Haloalkanes & Haloarenes", "Halogenation & Wurtz Synthesis"),
    "NEET2020_CHEM_114": ("Chemistry", "Haloalkanes & Haloarenes", "Beta-Elimination & Saytzeff Rule"),
    "NEET2021_CHEM_068": ("Chemistry", "Haloalkanes & Haloarenes", "Dehydrohalogenation & Saytzeff's Rule"),
    "NEET2022_CHEM_067": ("Chemistry", "Haloalkanes & Haloarenes", "Sandmeyer Synthesis of Haloarenes"),
    "NEET2024_CHEM_086": ("Chemistry", "Haloalkanes & Haloarenes", "Grignard Reagent & Organohalides"),
    "NEET2015_CHEM_041": ("Chemistry", "Alcohols, Phenols & Ethers", "Dehydration of Alcohols & Carbocation Rearrangement"),
    "NEET2025_CHEM_094": ("Biology", "Circulation & Excretion", "Human Circulatory System & Blood Vessels"),
}

VALID_SCIENTIFIC_LONG_WORDS = {
    "dichlorotetraaquachrominum",
    "dichlorotetraaquachromium",
    "dimethylcyclohexanol",
    "disproportionation",
    "chlorofluorocarbons",
    "dehydrohalogenation",
    "diaminetetraacetate",
    "ethylenediaminetetraacetate",
    "phosphoenolpyruvate",
    "aucgaucgaucgaucgaucgaucgaucg",
    "atcgatcgatcgatcgatcgatcgatcg",
    "uagcuagcuagcuagcuagcuagcuagc",
    "deoxyribonucleotides",
    "acetylcholinesterase",
    "spectrophotometer",
    "phenylketonuria",
    "hypercholesterolemia",
    "magnetohydrodynamic",
    "phosphofructokinase",
    "ribulosebisphosphate",
    "photophosphorylation",
    "chemiosmotic",
    "electroencephalogram",
    "immunoelectrophoresis",
}

GLUED_LONG_ALPHA_RE = re.compile(r"[a-zA-Z]{20,}")
FORMULA_TOKEN_RE = re.compile(
    r"(\^|_[0-9a-zA-Z]|=|\\frac|≥|≤|±|×|÷|√|π|θ|λ|μ|ω|Δ|∫|∑|\b(?:sin|cos|tan|log|ln)\b|\b10\s*[-^+]?\s*\d+)",
    re.I,
)
NUMBER_TOKEN_RE = re.compile(r"(?<![a-zA-Z])\d+(?:\.\d+)?(?:\s*[×x*]\s*10\s*[-^−+]?\s*\d+)?")
NEGATION_RE = re.compile(r"\b(not|incorrect|false|wrong|except|never|neither|cannot|absent)\b", re.I)


def _normalize_whitespace_and_typography(text: str) -> str:
    if not text:
        return ""
    out = text.replace("\x02", "").replace("\ufffd", "'").replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in out.split("\n") if ln.strip()]
    merged: list[str] = []
    for ln in lines:
        if not merged:
            merged.append(ln)
            continue
        is_structural = bool(
            re.match(
                r"^(?:Statement\s*(?:I|II|1|2)|Assertion|Reason|\([a-e1-4i]+\)|[A-E1-4][\.\)]\s|List|Column)",
                ln,
                re.I,
            )
        )
        if is_structural:
            merged.append("\n" + ln)
        else:
            merged.append(" " + ln)
    cleaned = "".join(merged)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def _repair_leaked_options_if_present(q: dict[str, Any]) -> bool:
    opt_d = str(q.get("option_d", "")).strip()
    m = LEAKED_OPTS_RE.match(opt_d)
    if not m:
        return False
    stmt_d_tail, real_1, real_2, real_3, real_4 = (g.strip() for g in m.groups())
    new_opts = [real_1, real_2, real_3, real_4]
    if len(set(new_opts)) < 4 or any(not x for x in new_opts):
        return False

    stmt_a = str(q.get("option_a", "")).strip()
    stmt_b = str(q.get("option_b", "")).strip()
    stmt_c = str(q.get("option_c", "")).strip()
    stem = str(q.get("question_text", "")).strip()
    appended_block = f"\n(a) {stmt_a}\n(b) {stmt_b}\n(c) {stmt_c}\n(d) {stmt_d_tail}"
    q["question_text"] = stem + appended_block
    q["option_a"] = real_1
    q["option_b"] = real_2
    q["option_c"] = real_3
    q["option_d"] = real_4
    return True


def _diagnose_structural_defect(q: dict[str, Any], valid_chapters: set[str]) -> str | None:
    opts = [str(q.get(k, "")).strip() for k in ("option_a", "option_b", "option_c", "option_d")]
    if any(not o for o in opts):
        return "empty_option"
    if len(set(o.lower() for o in opts)) < 4:
        return "duplicate_options"

    ans = str(q.get("correct_option", "")).strip().lower()
    if ans not in {"a", "b", "c", "d"}:
        return "invalid_correct_option"

    ch = str(q.get("chapter", "")).strip()
    if ch not in valid_chapters:
        return "unmapped_chapter"

    txt = str(q.get("question_text", "")).strip()
    if len(txt) < 12:
        return "truncated_short_stem"

    if txt[0].islower() and not txt.startswith(
        ("pH", "pOH", "pKa", "d-", "f-", "s-", "p-", "mRNA", "tRNA", "rRNA", "i-", "n-", "o-", "α", "β")
    ):
        return "truncated_lowercase_start"

    if re.search(r"Statement\s*(?:I|1)\b", txt, re.I) and not re.search(r"Statement\s*(?:II|2|Il|ll)\b", txt, re.I):
        return "missing_statement_2"

    if re.search(r"Assertion\s*\(?A\)?", txt, re.I) and not re.search(r"Reason\s*\(?R\)?", txt, re.I):
        return "missing_reason_R"

    is_ar = bool(re.search(r"Assertion", txt, re.I) and re.search(r"Reason", txt, re.I))
    if not is_ar and all(re.search(r"\([a-e]\)\s*(?:,|\band\b|only)", o, re.I) for o in opts[:2]):
        if not re.search(r"(?:^|\n)\s*\(?[abAB1][\.\)]", txt) and not re.search(r"\([ab]\)", txt):
            return "missing_lettered_statements_in_stem"

    if re.search(r"^\s*Column\s*[-–]?\s*I\s+Column\s*[-–]?\s*II\s*$", txt, re.I):
        return "empty_match_columns"

    if re.search(r"\bas shown in (?:the )?(?:figure|diagram|circuit|graph)\b", txt, re.I) and not q.get("image_asset"):
        return "missing_required_figure"

    for long_tok in GLUED_LONG_ALPHA_RE.findall(txt):
        if long_tok.lower() not in VALID_SCIENTIFIC_LONG_WORDS:
            return f"glued_ocr_token:{long_tok[:24]}"

    return None


def compute_item_feature_dict(q: dict[str, Any]) -> dict[str, Any]:
    txt = str(q.get("question_text", ""))
    expl = str(q.get("explanation", ""))
    opts = [str(q.get(k, "")) for k in ("option_a", "option_b", "option_c", "option_d")]
    words = txt.split()
    word_count = max(1, len(words))
    char_length = len(txt)

    f_hits = len(FORMULA_TOKEN_RE.findall(txt)) + sum(len(FORMULA_TOKEN_RE.findall(o)) for o in opts) // 2
    formula_density = round(min(1.0, f_hits / max(4.0, word_count * 0.30)), 4)

    n_hits = len(NUMBER_TOKEN_RE.findall(txt)) + sum(1 for o in opts if NUMBER_TOKEN_RE.search(o)) // 2
    numerical_complexity = round(min(1.0, n_hits / 6.0), 4)

    negation_trap = 1 if NEGATION_RE.search(txt) else 0

    token_sets = [set(re.findall(r"[a-zA-Z0-9]+", o.lower())) for o in opts]
    pair_sims: list[float] = []
    for i in range(4):
        for j in range(i + 1, 4):
            u = len(token_sets[i] | token_sets[j])
            pair_sims.append(len(token_sets[i] & token_sets[j]) / u if u > 0 else 0.5)
    distractor_similarity = round(sum(pair_sims) / len(pair_sims), 4)

    expl_steps = max(1, expl.count(".") + expl.count("=") + expl.count("->") + expl.count("→"))
    has_diagram = 1 if bool(q.get("has_diagram", False)) else 0
    ideal_time = int(q.get("ideal_time_sec") or 85)
    q_type = str(q.get("question_type") or "Single_Choice_MCQ")
    subj = str(q.get("subject") or "Biology")

    subj_offset = 0.14 if subj == "Physics" else (0.07 if subj == "Chemistry" else 0.0)
    type_offset = (
        0.14
        if any(k in q_type.lower() for k in ("assertion", "statement", "match", "numerical"))
        or re.search(r"Assertion|Statement\s*I|Match List|Column", txt, re.I)
        else 0.0
    )

    latent_complexity = (
        0.26 * min(1.0, word_count / 52.0)
        + 0.22 * formula_density
        + 0.20 * numerical_complexity
        + 0.10 * negation_trap
        + 0.08 * distractor_similarity
        + 0.10 * min(1.0, expl_steps / 8.0)
        + 0.08 * has_diagram
        + subj_offset
        + type_offset
    )

    return {
        "question_id": q["question_id"],
        "subject": subj,
        "chapter": q["chapter"],
        "topic_tag": q.get("topic_tag") or q["chapter"],
        "question_type": q_type,
        "word_count": word_count,
        "char_length": char_length,
        "formula_density": formula_density,
        "numerical_complexity": numerical_complexity,
        "negation_trap": negation_trap,
        "distractor_similarity": distractor_similarity,
        "explanation_steps": expl_steps,
        "has_diagram": has_diagram,
        "ideal_time_sec": ideal_time,
        "marks": int(q.get("marks", 4)),
        "negative_marks": int(q.get("negative_marks", -1)),
        "latent_complexity": round(latent_complexity, 5),
    }


def _deduplicate_exact_stems(questions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen_fingerprints: set[str] = set()
    kept: list[dict[str, Any]] = []
    removed = 0
    for q in questions:
        norm_stem = re.sub(r"[^a-z0-9]+", "", q["question_text"].lower())
        norm_opts = "|".join(sorted(re.sub(r"[^a-z0-9]+", "", q[k].lower()) for k in ("option_a", "option_b", "option_c", "option_d")))
        fp = f"{q['subject']}::{norm_stem[:140]}::{norm_opts}"
        if fp in seen_fingerprints:
            removed += 1
            continue
        seen_fingerprints.add(fp)
        kept.append(q)
    return kept, removed


def remediate_and_build_all_datasets() -> dict[str, Any]:
    raw_items = json.loads(RAW_FLAWLESS_PATH.read_text(encoding="utf-8"))
    dag_nodes = json.loads(DAG_PATH.read_text(encoding="utf-8"))["nodes"]
    valid_chapters = set(dag_nodes.keys())

    repaired_leaked = 0
    explicit_fixed = 0
    reclassified_chapters = 0
    clean_candidates: list[dict[str, Any]] = []
    quarantined_items: list[dict[str, Any]] = []

    for orig in raw_items:
        q = dict(orig)
        qid = q["question_id"]

        if qid in EXPLICIT_OCR_FIXES:
            q.update(EXPLICIT_OCR_FIXES[qid])
            explicit_fixed += 1

        if qid in EXPLICIT_CHAPTER_FIXES:
            subj_new, ch_new, tag_new = EXPLICIT_CHAPTER_FIXES[qid]
            q["subject"] = subj_new
            q["chapter"] = ch_new
            q["topic_tag"] = tag_new
            reclassified_chapters += 1

        if _repair_leaked_options_if_present(q):
            repaired_leaked += 1

        q["question_text"] = _normalize_whitespace_and_typography(str(q.get("question_text", "")))
        for opt_k in ("option_a", "option_b", "option_c", "option_d"):
            q[opt_k] = _normalize_whitespace_and_typography(str(q.get(opt_k, "")))
        q["explanation"] = _normalize_whitespace_and_typography(str(q.get("explanation", "")))

        if str(q.get("topic_tag", "")).strip() in ("", "General Core Concepts"):
            q["topic_tag"] = f"{q['chapter']} Core Concepts"

        defect = _diagnose_structural_defect(q, valid_chapters)
        if defect is not None:
            q["quarantine_reason"] = defect
            quarantined_items.append(q)
        else:
            clean_candidates.append(q)

    clean_items, exact_dups_removed = _deduplicate_exact_stems(clean_candidates)

    feats_by_id = {q["question_id"]: compute_item_feature_dict(q) for q in clean_items}
    items_by_chapter: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in clean_items:
        items_by_chapter[q["chapter"]].append(q)

    tier_recalibrated_count = 0
    for ch, ch_items in items_by_chapter.items():
        ch_items.sort(key=lambda item: feats_by_id[item["question_id"]]["latent_complexity"])
        n = len(ch_items)
        if n >= 3:
            n_basic = max(1, int(round(n * 0.26)))
            n_adv = max(1, int(round(n * 0.26)))
            if n_basic + n_adv >= n:
                n_basic = 1
                n_adv = 1
            for idx, item in enumerate(ch_items):
                fc = feats_by_id[item["question_id"]]
                if idx < n_basic:
                    target_tier = "basic"
                    base_time = 55 if item["subject"] == "Biology" else (70 if item["subject"] == "Chemistry" else 80)
                    ideal_t = int(round(base_time + 18.0 * fc["latent_complexity"]))
                elif idx >= n - n_adv:
                    target_tier = "advanced"
                    base_time = 85 if item["subject"] == "Biology" else (105 if item["subject"] == "Chemistry" else 125)
                    ideal_t = int(round(base_time + 28.0 * fc["latent_complexity"]))
                else:
                    target_tier = "medium"
                    base_time = 68 if item["subject"] == "Biology" else (85 if item["subject"] == "Chemistry" else 100)
                    ideal_t = int(round(base_time + 22.0 * fc["latent_complexity"]))

                if item.get("difficulty_level") != target_tier:
                    tier_recalibrated_count += 1
                item["difficulty_level"] = target_tier
                item["ideal_time_sec"] = ideal_t
                fc["ideal_time_sec"] = ideal_t
                fc["difficulty_level"] = target_tier

                txt_l = item["question_text"].lower()
                if re.search(r"assertion.*reason", txt_l, re.S):
                    item["question_type"] = "Assertion-Reason"
                elif re.search(r"statement\s*(?:i|1)\b.*statement\s*(?:ii|2)\b", txt_l, re.S):
                    item["question_type"] = "Statement-Based"
                elif re.search(r"match\s+(?:the\s+following|list|column)", txt_l):
                    item["question_type"] = "Match-the-Column"
                elif fc["numerical_complexity"] >= 0.34 or (fc["formula_density"] >= 0.35 and fc["numerical_complexity"] >= 0.17):
                    item["question_type"] = "Numerical"
                elif fc["has_diagram"] == 1:
                    item["question_type"] = "Diagram-Based"
                elif fc["formula_density"] >= 0.25:
                    item["question_type"] = "Direct Formula"
                else:
                    item["question_type"] = "Conceptual"
                fc["question_type"] = item["question_type"]

    clean_items.sort(key=lambda x: x["question_id"])
    CLEAN_OUT_PATH.write_text(json.dumps(clean_items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    QUARANTINE_OUT_PATH.write_text(json.dumps(quarantined_items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    q_csv_cols = [
        "question_id",
        "subject",
        "chapter",
        "topic_tag",
        "question_type",
        "word_count",
        "char_length",
        "formula_density",
        "numerical_complexity",
        "negation_trap",
        "distractor_similarity",
        "explanation_steps",
        "has_diagram",
        "ideal_time_sec",
        "marks",
        "negative_marks",
        "difficulty_level",
    ]
    with open(QUESTION_FEATURES_CSV_PATH, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=q_csv_cols, lineterminator="\n")
        writer.writeheader()
        for q in clean_items:
            fc = feats_by_id[q["question_id"]]
            writer.writerow({k: fc[k] for k in q_csv_cols})

    importance_path = MODEL_DIR / "topic_importance.json"
    topic_imp = json.loads(importance_path.read_text(encoding="utf-8")) if importance_path.exists() else {}

    chapter_rows: list[dict[str, Any]] = []
    chapter_summary_map: dict[str, dict[str, Any]] = {}
    for ch in sorted(valid_chapters):
        meta = dag_nodes[ch]
        ch_qs = items_by_chapter[ch]
        tier_cnt = Counter(q["difficulty_level"] for q in ch_qs)
        years = sorted(
            {
                int(m.group(1))
                for q in ch_qs
                for m in [re.search(r"(20\d{2})", str(q.get("paper_source", "")) + q["question_id"])]
                if m
            }
        )
        f_list = [feats_by_id[q["question_id"]] for q in ch_qs]
        n_q = len(ch_qs)
        mean_comp = sum(x["latent_complexity"] for x in f_list) / max(1, n_q)
        mean_time = sum(x["ideal_time_sec"] for x in f_list) / max(1, n_q)
        mean_form = sum(x["formula_density"] for x in f_list) / max(1, n_q)
        mean_num = sum(x["numerical_complexity"] for x in f_list) / max(1, n_q)
        mean_words = sum(x["word_count"] for x in f_list) / max(1, n_q)
        prereqs = list(meta.get("prerequisites", []))
        dependents_cnt = sum(1 for _, m2 in dag_nodes.items() if ch in m2.get("prerequisites", []))
        imp_val = float(topic_imp.get(ch, 1.0))

        row_obj = {
            "chapter": ch,
            "subject": meta.get("subject", "Biology"),
            "cluster": meta.get("cluster", "General"),
            "total_pyqs": n_q,
            "basic_pyqs": int(tier_cnt.get("basic", 0)),
            "medium_pyqs": int(tier_cnt.get("medium", 0)),
            "advanced_pyqs": int(tier_cnt.get("advanced", 0)),
            "distinct_exam_years": len(years),
            "earliest_year": years[0] if years else 2013,
            "latest_year": years[-1] if years else 2026,
            "mean_word_count": round(mean_words, 2),
            "mean_formula_density": round(mean_form, 4),
            "mean_numerical_complexity": round(mean_num, 4),
            "mean_latent_complexity": round(mean_comp, 4),
            "mean_ideal_time_sec": round(mean_time, 2),
            "prerequisite_count": len(prereqs),
            "dependent_chapters_count": dependents_cnt,
            "neet_importance_weight": round(imp_val, 4),
        }
        chapter_rows.append(row_obj)
        chapter_summary_map[ch] = {
            **row_obj,
            "prerequisites": prereqs,
            "years_present": years,
            "sample_question_ids": [q["question_id"] for q in ch_qs[:5]],
        }

    with open(CHAPTER_FEATURES_CSV_PATH, "w", newline="", encoding="utf-8") as f_ch:
        writer = csv.DictWriter(f_ch, fieldnames=list(chapter_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(chapter_rows)

    summary_payload = {
        "total_canonical_chapters": len(valid_chapters),
        "total_verified_pyqs": len(clean_items),
        "total_quarantined_items": len(quarantined_items),
        "repaired_leaked_statement_items": repaired_leaked,
        "explicit_ocr_repaired_items": explicit_fixed,
        "reclassified_chapter_items": reclassified_chapters,
        "exact_duplicates_removed": exact_dups_removed,
        "tier_recalibrated_items": tier_recalibrated_count,
        "chapters_missing_basic": sum(1 for r in chapter_rows if r["basic_pyqs"] == 0),
        "chapters_missing_medium": sum(1 for r in chapter_rows if r["medium_pyqs"] == 0),
        "chapters_missing_advanced": sum(1 for r in chapter_rows if r["advanced_pyqs"] == 0),
        "subject_totals": {
            subj: {
                "chapters": sum(1 for r in chapter_rows if r["subject"] == subj),
                "total_pyqs": sum(r["total_pyqs"] for r in chapter_rows if r["subject"] == subj),
                "basic": sum(r["basic_pyqs"] for r in chapter_rows if r["subject"] == subj),
                "medium": sum(r["medium_pyqs"] for r in chapter_rows if r["subject"] == subj),
                "advanced": sum(r["advanced_pyqs"] for r in chapter_rows if r["subject"] == subj),
            }
            for subj in ("Physics", "Chemistry", "Biology")
        },
        "chapters": chapter_summary_map,
    }
    CHAPTER_SUMMARY_OUT_PATH.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary_payload


if __name__ == "__main__":
    res = remediate_and_build_all_datasets()
    print(json.dumps({k: v for k, v in res.items() if k != "chapters"}, indent=2))
