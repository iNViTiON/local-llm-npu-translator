"""Prompt building and parsing of the extension's default (translategemma) template."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .registry import LANG_NAMES

SYSTEM_PROMPT = (
    "You are a translation assistant. Translate the user's text{src} to {tgt}. "
    "Preserve HTML tags, markdown formatting, code blocks, URLs, and proper nouns "
    "exactly as they appear. Do not add commentary, explanation, or quotation marks "
    "around the translation. Output only the translated text."
)


def lang_name(code: str | None) -> str | None:
    if not code:
        return None
    return LANG_NAMES.get(code) or LANG_NAMES.get(code.split("-")[0])


def system_prompt(src: str | None, tgt: str | None) -> str:
    src_name, tgt_name = lang_name(src), lang_name(tgt)
    return SYSTEM_PROMPT.format(
        src=f" from {src_name}" if src_name else "",
        tgt=tgt_name or "English",
    )


def translategemma_prompt(src: str | None, tgt: str, text: str) -> str:
    """TranslateGemma's own template (identical to the one the extension ships), for the OpenAI src/tgt route."""
    s_name, t_name = lang_name(src), lang_name(tgt) or tgt
    s_label = f"{s_name} ({src})" if s_name else "source-language"
    s_text = s_name or "source-language"
    return (
        f"You are a professional {s_label} to {t_name} ({tgt}) translator. Your goal is to accurately convey the meaning "
        f"and nuances of the original {s_text} text while adhering to {t_name} grammar, vocabulary, and cultural "
        f"sensitivities.\nProduce only the {t_name} translation, without any additional explanations or commentary. "
        f"Please translate the following {s_text} text into {t_name}:\n\n\n{text}"
    )


@dataclass
class ParsedPrompt:
    src: str | None
    tgt: str | None
    text: str


# The extension's default template:
#   "You are a professional {source} ({source_code}) to {target} ({target_code}) translator. ...
#    Please translate the following {source} text into {target}:\n\n\n{text}"
# {source} is rendered empty/"auto" when the user picks auto-detect.
# The codes must sit in the template's opening sentence (anchored at the start, confined to line 1) and the text
# starts after the template's own first `:` + blank lines, so page text that merely imitates the header (inside a
# customised prompt) can never choose the language codes or the payload.
_CODE = r"[A-Za-z]{2,3}(?:-[A-Za-z]+)?"
_HEADER = re.compile(
    rf"\A\s*You are a professional [^\n]*?\(({_CODE}|auto)\) to [^\n]*?\(({_CODE})\) translator\b"
)
_SPLIT = ":\n\n\n"


def parse_extension_prompt(prompt: str) -> ParsedPrompt | None:
    """Recover (src, tgt, text) from the extension's default prompt, or None if it was customised."""
    m = _HEADER.match(prompt)
    if not m:
        return None
    _, sep, text = prompt[m.end():].partition(_SPLIT)
    if not sep:
        return None
    src = m.group(1)
    return ParsedPrompt(src=None if src == "auto" else src, tgt=m.group(2), text=text)
