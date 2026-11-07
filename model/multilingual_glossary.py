from __future__ import annotations

from typing import Any

CHAPTER_HINDI_DICTIONARY: dict[str, dict[str, Any]] = {
    "Mole Concept / Stoichiometry": {
        "hindi_name": "मोल संकल्पना एवं रससमीकरणमिति",
        "subject_hi": "रसायन विज्ञान",
        "core_terms": [
            {"en": "Molar Mass", "hi": "मोलर द्रव्यमान"},
            {"en": "Limiting Reagent", "hi": "सीमांत अभिकर्मक"},
            {"en": "Avogadro Number", "hi": "आवोगाद्रो संख्या"},
            {"en": "Empirical Formula", "hi": "मूलानुपाती सूत्र"},
        ],
        "socratic_tip_hi": "पहले समीकरण को संतुलित करें और सीमांत अभिकर्मक के आधार पर उत्पाद के मोलों की गणना करें।",
    },
    "Cell: The Unit of Life": {
        "hindi_name": "कोशिका: जीवन की इकाई",
        "subject_hi": "जीव विज्ञान",
        "core_terms": [
            {"en": "Prokaryote", "hi": "प्रोकैरियोट"},
            {"en": "Mitochondria", "hi": "माइटोकॉन्ड्रिया"},
            {"en": "Endoplasmic Reticulum", "hi": "अंतर्द्रव्यी जालिका"},
            {"en": "Cell Membrane", "hi": "कोशिका झिल्ली"},
        ],
        "socratic_tip_hi": "प्रोकैरियोटिक और यूकैरियोटिक राइबोसोम (70S बनाम 80S) के अंतर पर ध्यान दें।",
    },
    "The Living World & Taxonomy": {
        "hindi_name": "जीव जगत एवं वर्गिकी",
        "subject_hi": "जीव विज्ञान",
        "core_terms": [
            {"en": "Binomial Nomenclature", "hi": "द्विपद नामकरण"},
            {"en": "Taxonomic Hierarchy", "hi": "वर्गिकीय पदानुक्रम"},
            {"en": "Herbarium", "hi": "हर्बेरियम"},
        ],
        "socratic_tip_hi": "वर्गिकी पदानुक्रम: जगत -> संघ -> वर्ग -> गण -> कुल -> वंश -> जाति।",
    },
    "Units & Dimensions": {
        "hindi_name": "मात्रक एवं विमाएं",
        "subject_hi": "भौतिकी",
        "core_terms": [
            {"en": "Dimensional Formula", "hi": "विमीय सूत्र"},
            {"en": "Significant Figures", "hi": "सार्थक अंक"},
            {"en": "Percentage Error", "hi": "प्रतिशत त्रुटि"},
        ],
        "socratic_tip_hi": "विमीय समांगता के सिद्धांत का उपयोग करें: केवल समान विमा वाली राशियों को जोड़ा या घटाया जा सकता है।",
    },
    "Biomolecules": {
        "hindi_name": "जैव अणु",
        "subject_hi": "जीव विज्ञान",
        "core_terms": [
            {"en": "Peptide Bond", "hi": "पेप्टाइड बंध"},
            {"en": "Enzyme Kinetics", "hi": "एंजाइम बलगतिकी"},
            {"en": "Nucleotide", "hi": "न्यूक्लियोटाइड"},
            {"en": "Competitive Inhibition", "hi": "प्रतिस्पर्धी संदमन"},
        ],
        "socratic_tip_hi": "प्रतिस्पर्धी संदमन में Km बढ़ता है जबकि Vmax अपरिवर्तित रहता है।",
    },
    "Kinematics": {
        "hindi_name": "शुद्ध गतिकी",
        "subject_hi": "भौतिकी",
        "core_terms": [
            {"en": "Instantaneous Velocity", "hi": "तात्कालिक वेग"},
            {"en": "Projectile Motion", "hi": "प्रक्षेप्य गति"},
            {"en": "Uniform Acceleration", "hi": "एकसमान त्वरण"},
        ],
        "socratic_tip_hi": "प्रक्षेप्य गति में क्षैतिज वेग घटक हमेशा नियत रहता है यदि वायु प्रतिरोध नगण्य हो।",
    },
    "Genetics & Molecular Inheritance": {
        "hindi_name": "आनुवंशिकी एवं आण्विक वंशागति",
        "subject_hi": "जीव विज्ञान",
        "core_terms": [
            {"en": "DNA Replication", "hi": "डीएनए प्रतिकृतियन"},
            {"en": "Transcription", "hi": "अनुलेखन"},
            {"en": "Translation", "hi": "अनुवादन"},
            {"en": "Codon", "hi": "प्रकूट"},
        ],
        "socratic_tip_hi": "अनुलेखन सदैव 5' से 3' दिशा में होता है और टेम्पलेट रज्जुक 3' से 5' होता है।",
    },
}

TRAP_HINDI_MAP: dict[str, dict[str, str]] = {
    "SIGN_INVERSION": {
        "label_hi": "चिह्न उत्क्रमण जाल (Sign Error)",
        "explanation_hi": "आपने मान तो सही निकाला, लेकिन धनात्मक या ऋणात्मक चिह्न में गलती की (जैसे कार्य = -PΔV या दर्पण सूत्र चिह्न परिपाटी)।",
    },
    "FACTOR_OF_TWO": {
        "label_hi": "दोगुना/आधा गुणांक त्रुटि (Factor of 2 Trap)",
        "explanation_hi": "आप सूत्र में 1/2 या 2 का गुणांक भूल गए (जैसे गतिज ऊर्जा में 1/2mv² या त्रिज्या के स्थान पर व्यास का उपयोग)।",
    },
    "ORDER_OF_MAGNITUDE_UNIT": {
        "label_hi": "इकाई एवं कोटि मान त्रुटि (Unit & Order Trap)",
        "explanation_hi": "SI और CGS मात्रक के रूपांतरण में गलती हुई (जैसे cm को m में बदलना या J को kJ में बदलना)।",
    },
    "RECIPROCAL_INVERSION": {
        "label_hi": "व्युत्क्रम उत्क्रमण त्रुटि (Reciprocal Inversion)",
        "explanation_hi": "आपने अनुपात को उलट दिया (जैसे 1/R_eq के स्थान पर सीधे R_eq मान लेना)।",
    },
    "NEGATION_LOGIC": {
        "label_hi": "नकारात्मक कथन भ्रांति (Not Correct Trap)",
        "explanation_hi": "प्रश्न में 'सही नहीं है' (NOT correct) पूछा गया था, परंतु आपने सही कथन चुन लिया।",
    },
    "CONCEPTUAL_SUBSTITUTION": {
        "label_hi": "अवधारणात्मक प्रतिस्थापन त्रुटि (Concept Trap)",
        "explanation_hi": "दो मिलती-जुलती वैज्ञानिक अवधारणाओं या अपवादों में संशय हुआ।",
    },
}


def get_chapter_multilingual_metadata(chapter_name: str) -> dict[str, Any]:
    meta = CHAPTER_HINDI_DICTIONARY.get(chapter_name)
    if meta:
        return {
            "chapter_en": chapter_name,
            "chapter_hi": meta["hindi_name"],
            "subject_hi": meta["subject_hi"],
            "core_terms": meta["core_terms"],
            "socratic_tip_hi": meta["socratic_tip_hi"],
        }
    return {
        "chapter_en": chapter_name,
        "chapter_hi": chapter_name,
        "subject_hi": "एनसीईआरटी",
        "core_terms": [],
        "socratic_tip_hi": "एनसीईआरटी के मूल सिद्धांतों और सूत्रों का पुनरावलोकन करें।",
    }


def translate_distractor_trap_to_hindi(trap_category: str) -> dict[str, str]:
    return TRAP_HINDI_MAP.get(trap_category, {
        "label_hi": "वैज्ञानिक अवधारणात्मक संशय",
        "explanation_hi": "विकल्प का विश्लेषण करें और संबंधित एनसीईआरटी नियम की जांच करें।",
    })
