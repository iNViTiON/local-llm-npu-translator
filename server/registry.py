"""Model registry and language tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Backend = Literal["openvino", "openvino_seq2seq", "openvino_vlm", "stub"]

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ModelEntry:
    short_name: str
    backend: Backend
    path: Path
    owned_by: str
    note: str
    chat: bool
    languages_supported: int
    max_prompt_len: int = 4096
    npu_piece_tokens: int | None = None  # NPU: translate longer text in sentence groups of at most this many tokens
    aliases: tuple[str, ...] = field(default_factory=tuple)


REGISTRY: dict[str, ModelEntry] = {
    # Named like the extension's default model pref, so /api/tags satisfies its setup check with no configuration.
    "translategemma:4b": ModelEntry(
        short_name="translategemma:4b",
        backend="openvino_vlm",
        path=ROOT / "models/translategemma-4b-it-int4-sym-ov",
        owned_by="google",
        note="Gemma-3-4B fine-tuned for translation (55 languages), run text-only through VLMPipeline.",
        chat=True,
        languages_supported=55,
        max_prompt_len=768,  # NPU: correct at <= 896, garbage at >= 1024 (Gemma-3 sliding window is 1024)
        npu_piece_tokens=128,  # prompt + output must fit 896 tokens, and output degrades on long prompts
        aliases=("translategemma-4b-it",),
    ),
}

# FLORES-200 codes (NLLB). Keys are the ISO-639-1 codes the extension uses.
NLLB_LANG_CODES = {
    "en": "eng_Latn", "et": "est_Latn", "fi": "fin_Latn", "de": "deu_Latn",
    "fr": "fra_Latn", "es": "spa_Latn", "sv": "swe_Latn", "no": "nob_Latn",
    "da": "dan_Latn", "ru": "rus_Cyrl", "pl": "pol_Latn", "it": "ita_Latn",
    "pt": "por_Latn", "nl": "nld_Latn", "cs": "ces_Latn", "hu": "hun_Latn",
    "el": "ell_Grek", "ro": "ron_Latn", "lv": "lvs_Latn", "lt": "lit_Latn",
    "uk": "ukr_Cyrl", "tr": "tur_Latn", "ar": "arb_Arab", "he": "heb_Hebr",
    "hi": "hin_Deva", "ja": "jpn_Jpan", "ko": "kor_Hang", "zh-Hans": "zho_Hans",
    "zh-Hant": "zho_Hant", "zh": "zho_Hans", "id": "ind_Latn", "th": "tha_Thai",
    "vi": "vie_Latn", "bg": "bul_Cyrl", "hr": "hrv_Latn", "sk": "slk_Latn",
    "sl": "slv_Latn", "ca": "cat_Latn", "ga": "gle_Latn", "mt": "mlt_Latn",
}

LANG_NAMES = {
    "en": "English", "et": "Estonian", "fi": "Finnish", "de": "German",
    "fr": "French", "es": "Spanish", "sv": "Swedish", "no": "Norwegian",
    "da": "Danish", "ru": "Russian", "pl": "Polish", "it": "Italian",
    "pt": "Portuguese", "nl": "Dutch", "cs": "Czech", "hu": "Hungarian",
    "el": "Greek", "ro": "Romanian", "lv": "Latvian", "lt": "Lithuanian",
    "uk": "Ukrainian", "tr": "Turkish", "ar": "Arabic", "hi": "Hindi",
    "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "zh-Hans": "Chinese",
    "zh-Hant": "Chinese", "bg": "Bulgarian", "hr": "Croatian",
    "sk": "Slovak", "sl": "Slovenian", "ca": "Catalan", "ga": "Irish",
    "mt": "Maltese",
}


def find_model(name: str) -> ModelEntry | None:
    """Resolve a client-supplied model name. Tolerates Ollama-style `:tag` suffixes."""
    name = name.strip()
    if name in REGISTRY:
        return REGISTRY[name]
    base = name.split(":", 1)[0]
    return REGISTRY.get(base) or next((e for e in REGISTRY.values() if name in e.aliases or base in e.aliases), None)
