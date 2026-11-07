from __future__ import annotations

from typing import Any
from model.multilingual_glossary import translate_distractor_trap_to_hindi

TRAP_VOICE_TEMPLATES = {
    "SIGN_INVERSION": (
        "Hold on for a second. Look closely at the sign in your answer. "
        "In this question from {chapter}, you computed the right magnitude, but the convention tripped you up. "
        "Ask yourself: is work done on the system or by the system? That negative sign changes everything."
    ),
    "FACTOR_OF_TWO": (
        "Let us pause right there. Your calculation is very close, but notice how option {selected_option} "
        "is exactly double or half of option {correct_option}. "
        "Did you forget the one-half in the formula, or use diameter instead of radius?"
    ),
    "ORDER_OF_MAGNITUDE_UNIT": (
        "Watch out for the units here. You had the formula down, but check your powers of ten. "
        "Converting centimeters to meters or milliliters to liters is the classic trap NTA sets every single year."
    ),
    "RECIPROCAL_INVERSION": (
        "Take a step back. You solved for the reciprocal, but did you flip it back at the end? "
        "Whenever you calculate parallel resistors or focal lengths, remember one over R is not the final resistance."
    ),
    "NEGATION_LOGIC": (
        "Careful! Re-read the stem of the question. Notice the key word: which of the following is NOT correct. "
        "You picked a statement that is scientifically true, but the question asked for the incorrect statement."
    ),
    "CONCEPTUAL_SUBSTITUTION": (
        "Let us dissect this together. In {chapter}, this particular exception is highlighted right in NCERT. "
        "Take a breath and think about the foundational principle behind this reaction mechanism."
    ),
}


def generate_socratic_audio_script(
    question_stem: str,
    selected_option: str,
    correct_option: str,
    chapter_name: str,
    distractor_trap: str,
    language: str = "en",
) -> dict[str, Any]:
    trap_key = distractor_trap if distractor_trap in TRAP_VOICE_TEMPLATES else "CONCEPTUAL_SUBSTITUTION"
    template = TRAP_VOICE_TEMPLATES[trap_key]

    spoken_en = template.format(
        chapter=chapter_name,
        selected_option=selected_option,
        correct_option=correct_option,
    )

    hindi_meta = translate_distractor_trap_to_hindi(trap_key)
    spoken_hi = f"ध्यान से देखें: {hindi_meta['explanation_hi']} {chapter_name} के इस प्रश्न में अपने उत्तर का पुनः विश्लेषण करें।"

    ssml = (
        f"<speak>"
        f"<prosody rate='95%' pitch='+0%'>"
        f"<p>{spoken_en if language == 'en' else spoken_hi}</p>"
        f"<break time='600ms'/>"
        f"<emphasis level='moderate'>Try solving it again using this hint.</emphasis>"
        f"</prosody>"
        f"</speak>"
    )

    return {
        "trap_category": trap_key,
        "audio_language": language,
        "spoken_text": spoken_en if language == "en" else spoken_hi,
        "ssml_payload": ssml,
        "duration_seconds_est": max(4.0, round(len(spoken_en.split()) * 0.42, 1)),
        "hindi_audio_summary": spoken_hi,
    }
