"""Prompt format, language names, and supported trainer versions."""
import re

OPEN = "<english_translation>"


CLOSE = "</english_translation>"


SUPPORTED_TRL_VERSIONS = {"0.24.0", "1.9.2"}


TRANSLATION_RE = re.compile(re.escape(OPEN) + r"(.*?)" + re.escape(CLOSE), re.S)


LANGUAGE_NAMES = {
    "en": "English", "bn": "Bengali", "de": "German", "es": "Spanish",
    "fr": "French", "ja": "Japanese", "ru": "Russian", "sw": "Swahili",
    "th": "Thai", "zh": "Chinese",
}


SYSTEM = (
    "Solve the mathematical word problem. First translate the complete question "
    "faithfully into English inside <english_translation>...</english_translation>. "
    "Preserve all quantities, units, and relationships. Then reason in English "
    "and give the final answer in \\boxed{...}. Do not put reasoning inside the "
    "translation tags."
)
