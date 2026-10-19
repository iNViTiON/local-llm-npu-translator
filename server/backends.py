"""Inference backends. Each exposes `generate_chat` / `translate`; loading is lazy and lock-serialised by the app."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Protocol

from .registry import NLLB_LANG_CODES, ModelEntry

log = logging.getLogger("npu-translator")

# NPU LLMs prefill in 128-token chunks instead of one pass padded to MAX_PROMPT_LEN, so a typical 90-150 token prompt
# costs one small chunk: time to first token 0.82 -> 0.37 s (TranslateGemma) and 0.46 -> 0.17 s (EuroLLM), same decode
# speed. Output is identical for one-chunk prompts; longer ones can differ slightly in wording (numerics). A chunk of 64
# does not compile; NPUW_LLM_ENABLE_PREFIX_CACHING nets out (-0.26 s first token on two chunks, ~3% slower decode).
NPU_LLM_CONFIG = {"NPUW_LLM_PREFILL_HINT": "DYNAMIC", "NPUW_LLM_PREFILL_CHUNK_SIZE": 128}


class BackendError(RuntimeError):
    pass


FALLBACKS = {"none": [], "gpu": ["GPU"], "cpu": ["CPU"], "all": ["GPU", "CPU"]}


def device_chain(device: str, fallback: str) -> list[str]:
    return [device] + [d for d in FALLBACKS[fallback] if d != device]


@dataclass
class Stats:
    """Timing of the last request. Decode speed is measured after the first token (LLM: it comes out of prefill)."""

    prompt_tokens: int = 0
    new_tokens: int = 0
    prompt_seconds: float = 0.0  # prefill (LLM) or encoder (NLLB)
    decode_seconds: float = 0.0
    decode_tokens: int = 0  # tokens produced inside decode_seconds

    @property
    def tokens_per_second(self) -> float:
        return self.decode_tokens / self.decode_seconds if self.decode_seconds > 0 else 0.0

    @property
    def eval_seconds(self) -> float:
        """decode_seconds scaled to all new_tokens, so eval_count / eval_duration (Ollama's fields) is the decode rate."""
        return self.decode_seconds * self.new_tokens / self.decode_tokens if self.decode_tokens else 0.0

    def add(self, other: Stats | None) -> None:
        if other is not None:
            self.prompt_tokens += other.prompt_tokens
            self.new_tokens += other.new_tokens
            self.prompt_seconds += other.prompt_seconds
            self.decode_seconds += other.decode_seconds
            self.decode_tokens += other.decode_tokens


# The extension sends inline elements (links, bold) as `[N:text]`; a split inside one would tear the element apart.
_MARKER = re.compile(r"([\[［]\s*\d+\s*[:：])|[\]］]")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])\s*|\n+")


def split_sentences(text: str) -> list[tuple[str, str]]:
    """Sentences and lines of `text`, each with the separator to put after its translation: the original line breaks
    (so a batch of lines comes back with as many lines), else one space. Never splits inside a `[N:...]` marker."""
    events = [(m.start(), 1 if m.group(1) else -1) for m in _MARKER.finditer(text)]
    out, start, depth, e = [], 0, 0, 0
    for m in _SENTENCE_END.finditer(text):
        while e < len(events) and events[e][0] < m.start():
            depth, e = max(depth + events[e][1], 0), e + 1
        sentence = text[start:m.start()].strip()
        if depth or not sentence:
            continue
        out.append((sentence, "\n" * m.group().count("\n") or " "))
        start = m.end()
    if rest := text[start:].strip():
        out.append((rest, ""))
    return out


def fit(sentences: list[tuple[str, str]], budget: int, size: Callable[[str], int], group: bool) -> list[tuple[str, str]]:
    """Pieces of at most `budget` tokens (by `size`), with their separators. With `group`, consecutive sentences share a
    piece while they fit. A sentence over the budget is cut at the longest prefix that fits (binary search on token
    count, so O(log n) tokenizations per cut), preferably at a space; nothing is dropped."""
    out: list[tuple[str, str]] = []
    for sent, sep in sentences:
        while size(sent) > budget:
            lo, hi = 1, len(sent)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                lo, hi = (mid, hi) if size(sent[:mid]) <= budget else (lo, mid - 1)
            cut = sent.rfind(" ", 0, lo + 1)
            cut = cut if cut > lo // 2 else lo
            out.append((sent[:cut].strip(), " "))
            sent = sent[cut:].strip()
        if not sent:
            if out:
                out[-1] = (out[-1][0], sep)
        elif group and out and size(joined := out[-1][0] + out[-1][1] + sent) <= budget:
            out[-1] = (joined, sep)
        else:
            out.append((sent, sep))
    return out


class ModelBackend(Protocol):
    entry: ModelEntry
    last_stats: Stats | None

    def load(self) -> None: ...
    @property
    def loaded(self) -> bool: ...
    def chat(self, messages: list[dict], max_new_tokens: int, temperature: float) -> str: ...
    def chat_stream(self, messages: list[dict], max_new_tokens: int, temperature: float) -> Iterator[str]: ...
    def translate(self, text: str, src: str | None, tgt: str, max_new_tokens: int) -> str: ...


class LLMBackend:
    """Decoder-only model through openvino_genai.LLMPipeline (EuroLLM)."""

    def __init__(self, entry: ModelEntry, device: str, fallback: str = "none", cache_dir: Path | None = None):
        self.entry, self.device, self.fallback, self._pipe = entry, device, fallback, None
        self.cache_dir, self.active_device, self.last_stats = cache_dir, None, None

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    @staticmethod
    def _stats(result) -> Stats | None:
        pm = getattr(result, "perf_metrics", None)  # openvino_genai DecodedResults
        if pm is None:
            return None
        n, ttft = pm.get_num_generated_tokens(), pm.get_ttft().mean / 1000
        total = pm.get_generate_duration().mean / 1000
        return Stats(pm.get_num_input_tokens(), n, ttft, max(total - ttft, 0.0), max(n - 1, 0))

    def load(self) -> None:
        if self._pipe is not None:
            return
        import openvino_genai as ovg

        if not self.entry.path.exists():
            raise BackendError(f"model not converted: {self.entry.path} (run scripts.convert_translation_models)")
        chain = device_chain(self.device, self.fallback)
        last = ""
        for dev in chain:
            kwargs = {"MAX_PROMPT_LEN": self.entry.max_prompt_len, **NPU_LLM_CONFIG} if dev == "NPU" else {}
            if self.cache_dir:  # compiled-blob cache: restarts skip the ~40 s NPU compile
                kwargs["CACHE_DIR"] = str(self.cache_dir)
            log.info("compiling %s on %s (first run can take minutes)", self.entry.short_name, dev)
            try:
                self._pipe = self._open(ovg, dev, kwargs)
                self.active_device = dev
                return
            except Exception as e:
                last = str(e).strip().splitlines()[-1]
                log.warning("%s compile failed on %s: %s", self.entry.short_name, dev, last)
        raise BackendError(f"could not compile {self.entry.short_name} on {chain}: {last} ( --fallback gpu|cpu|all allows other devices)")

    def _open(self, ovg, device: str, kwargs: dict):
        return ovg.LLMPipeline(str(self.entry.path), device, **kwargs)

    def _limit_error(self, e: Exception) -> Exception:
        """The NPU pipeline has a fixed prompt window; say so instead of surfacing a bare RuntimeError (a 500)."""
        if "MAX_PROMPT_LEN" in str(e):
            return BackendError(f"input too long: {self.entry.short_name} accepts at most {self.entry.max_prompt_len} prompt tokens on the NPU")
        return e

    def _generate(self, prompt: str, cfg, streamer=None):
        # A list input returns DecodedResults (text + perf_metrics); a plain str returns only the text.
        try:
            return self._pipe.generate([prompt], cfg, streamer) if streamer else self._pipe.generate([prompt], cfg)
        except RuntimeError as e:
            raise self._limit_error(e) from None

    def _config(self, max_new_tokens: int, temperature: float):
        import openvino_genai as ovg

        cfg = ovg.GenerationConfig()
        cfg.max_new_tokens = max_new_tokens
        if temperature > 0:
            cfg.do_sample, cfg.temperature = True, temperature
        else:
            cfg.do_sample = False
        return cfg

    def _prompt(self, messages: list[dict]) -> str:
        history = self._pipe.get_tokenizer()
        return history.apply_chat_template(messages, add_generation_prompt=True)

    def chat(self, messages, max_new_tokens, temperature) -> str:
        self.load()
        out = self._generate(self._prompt(messages), self._config(max_new_tokens, temperature))
        self.last_stats = self._stats(out)
        return out.texts[0].strip()

    def chat_stream(self, messages, max_new_tokens, temperature) -> Iterator[str]:
        # openvino_genai streams through a callback; bridge it to a generator with a queue.
        import queue
        import threading

        self.load()
        q: queue.Queue = queue.Queue()
        done, stop, results = object(), threading.Event(), []

        def cb(piece: str) -> bool:
            q.put(piece)
            return stop.is_set()  # True aborts generation

        def run():
            try:
                results.append(self._generate(self._prompt(messages), self._config(max_new_tokens, temperature), cb))
            except Exception as e:  # surface in consumer (_generate already turned a prompt-limit error into BackendError)
                q.put(e)
            finally:
                q.put(done)

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        try:
            while (item := q.get()) is not done:
                if isinstance(item, Exception):
                    raise item if isinstance(item, BackendError) else BackendError(str(item))
                yield item
        finally:
            # Consumer gone (client disconnect) or finished: never leave generate() running on a pipeline the
            # next request is about to use.
            stop.set()
            worker.join()
            self.last_stats = self._stats(results[0]) if results else None

    def translate(self, text, src, tgt, max_new_tokens) -> str:
        self.load()
        pieces = self._pieces(text)
        if len(pieces) < 2:
            return self._translate_piece(text, src, tgt, max_new_tokens)
        out, stats = [], Stats()
        for piece, sep in pieces:
            out.append(self._translate_piece(piece, src, tgt, min(max_new_tokens, len(piece) * 2 + 64)) + sep)
            stats.add(self.last_stats)
        self.last_stats = stats
        return "".join(out).strip()

    def _pieces(self, text: str) -> list[tuple[str, str]]:
        """On the NPU, text over `entry.npu_piece_tokens` goes in sentence groups. Its static KV cache caps prompt plus
        output (a 370-word paragraph came back cut off mid-sentence), and TranslateGemma's NPU output degrades on long
        prompts (garbled Estonian at ~480 prompt tokens, clean in ~180-token pieces)."""
        budget = self.entry.npu_piece_tokens
        if self.active_device != "NPU" or not budget:
            return [(text, "")]
        return fit(split_sentences(text), budget, self._size, group=True)

    def _size(self, text: str) -> int:
        return self._pipe.get_tokenizer().encode([text], add_special_tokens=False).input_ids.shape[1]

    def _piece_messages(self, text, src, tgt) -> list[dict]:
        from .prompts import system_prompt

        return [{"role": "system", "content": system_prompt(src, tgt)}, {"role": "user", "content": text}]

    def _translate_piece(self, text, src, tgt, max_new_tokens) -> str:
        return self.chat(self._piece_messages(text, src, tgt), max_new_tokens, 0.0)

    def translate_stream(self, text, src, tgt, max_new_tokens) -> Iterator[str]:
        """`translate`, as it is produced: the same pieces and prompts, token by token. A piece's translation is
        trimmed like `chat` trims it; the separators between pieces are emitted between them."""
        self.load()
        pieces = self._pieces(text)
        stats = Stats()
        try:
            for i, (piece, sep) in enumerate(pieces):
                budget = max_new_tokens if len(pieces) < 2 else min(max_new_tokens, len(piece) * 2 + 64)
                yield from _trimmed(self.chat_stream(self._piece_messages(piece, src, tgt), budget, 0.0))
                stats.add(self.last_stats)
                if sep and i < len(pieces) - 1:
                    yield sep
        finally:
            self.last_stats = stats


def _trimmed(tokens: Iterator[str]) -> Iterator[str]:
    """Drop the whitespace around a streamed text, the way `str.strip` does for the whole: leading whitespace is
    skipped and trailing whitespace is held back until more text follows (and dropped at the end)."""
    started, pending = False, ""
    for token in tokens:
        if not started:
            token = token.lstrip()
            if not token:
                continue
            started = True
        text = pending + token
        core = text.rstrip()
        pending = text[len(core):]
        if core:
            yield core


class VLMBackend(LLMBackend):
    """TranslateGemma: a Gemma3ForConditionalGeneration export, used text-only through VLMPipeline.

    The model's shipped chat template (a ~580-line language table) does not parse in GenAI's Jinja engine, so the
    template is skipped and Gemma's turn markup is written here; the extension's prompt already is TranslateGemma's
    instruction text. Stop ids come from the export's generation_config.json (Gemma has two: <eos> and
    <end_of_turn>), which the pipeline does not always pick up on its own.
    """

    _stop_ids: set[int] = set()

    def _open(self, ovg, device: str, kwargs: dict):
        import json

        cfg_file = self.entry.path / "generation_config.json"
        eos = json.loads(cfg_file.read_text()).get("eos_token_id", []) if cfg_file.exists() else []
        self._stop_ids = set(eos if isinstance(eos, list) else [eos])
        return ovg.VLMPipeline(str(self.entry.path), device, **kwargs)

    def _config(self, max_new_tokens: int, temperature: float):
        cfg = super()._config(max_new_tokens, temperature)
        cfg.apply_chat_template = False
        if self._stop_ids:
            cfg.stop_token_ids = self._stop_ids
        return cfg

    def _prompt(self, messages: list[dict]) -> str:
        # Single-turn: system text (if any) and the user text become one user turn. The tokenizer adds <bos>.
        text = "\n\n".join(m["content"] for m in messages if m["role"] in ("system", "user") and m["content"])
        return f"<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n"

    def _generate(self, prompt: str, cfg, streamer=None):
        kwargs = {"streamer": streamer} if streamer else {}  # the binding rejects streamer=None
        try:
            return self._pipe.generate(prompt, generation_config=cfg, **kwargs)
        except RuntimeError as e:
            raise self._limit_error(e) from None

    def _piece_messages(self, text, src, tgt) -> list[dict]:
        from .prompts import translategemma_prompt

        return [{"role": "user", "content": translategemma_prompt(src, tgt, text)}]


class Seq2SeqBackend:
    """NLLB through raw OpenVINO with static shapes, so it can run on NPU.

    optimum-intel's seq2seq path keeps a dynamic KV cache, which the NPU rejects. Instead the model is exported
    without cache; encoder and decoder are reshaped to fixed [1, SEQ] and the greedy loop re-feeds the (padded)
    decoder prefix each step. Causal masking makes the padding harmless. Fine for sentence-sized chunks.
    """

    SEQ = 128

    def __init__(self, entry: ModelEntry, device: str, fallback: str = "none", cache_dir: Path | None = None):
        self.entry, self.device, self.fallback, self.cache_dir = entry, device, fallback, cache_dir
        self._enc = self._dec = self._tok = None
        self.active_device, self.last_stats = None, None

    @property
    def loaded(self) -> bool:
        return self._enc is not None

    def load(self) -> None:
        if self._enc is not None:
            return
        import openvino as ov
        from transformers import AutoTokenizer

        path = self.entry.path
        if not (path / "openvino_encoder_model.xml").exists():
            raise BackendError(f"model not converted: {path} (run scripts.convert_translation_models)")
        self._tok = AutoTokenizer.from_pretrained(path)
        core, n = ov.Core(), self.SEQ
        enc = core.read_model(path / "openvino_encoder_model.xml")
        dec = core.read_model(path / "openvino_decoder_model.xml")
        if any(i.get_any_name() == "past_key_values.0.key" for i in dec.inputs):
            raise BackendError("NLLB export has a KV cache; re-export with scripts.convert_translation_models")
        enc.reshape({i.get_any_name(): [1, n] for i in enc.inputs})
        dec.reshape({"encoder_attention_mask": [1, n], "input_ids": [1, n], "encoder_hidden_states": [1, n, 1024]})
        chain, last = device_chain(self.device, self.fallback), ""
        props = {"CACHE_DIR": str(self.cache_dir)} if self.cache_dir else {}
        for dev in chain:
            log.info("compiling %s on %s", self.entry.short_name, dev)
            try:
                self._enc, self._dec = core.compile_model(enc, dev, props), core.compile_model(dec, dev, props)
                self.active_device = dev
                return
            except Exception as e:
                self._enc = self._dec = None
                last = str(e).strip().splitlines()[-1]
                log.warning("%s compile failed on %s: %s", self.entry.short_name, dev, last)
        raise BackendError(f"could not compile {self.entry.short_name} on {chain}: {last} (--fallback gpu|cpu|all allows other devices)")

    def chat(self, messages, max_new_tokens, temperature) -> str:
        raise BackendError(f"{self.entry.short_name} is translation-only; use src_lang/tgt_lang")

    def chat_stream(self, messages, max_new_tokens, temperature):
        raise BackendError(f"{self.entry.short_name} is translation-only; use src_lang/tgt_lang")

    def _pieces(self, text: str) -> list[tuple[str, str]]:
        """One sentence per piece (NLLB is sentence-level and drops/loops on multi-sentence input); one over the window
        is cut (see `fit`). Line breaks are kept, so a batch of lines comes back with as many lines."""
        size = lambda s: len(self._tok(s, add_special_tokens=False)["input_ids"])
        return fit(split_sentences(text), self.SEQ - 4, size, group=False)  # room for the language token and eos

    def _translate_chunk(self, text: str, src_code: str, tgt_code: str, max_new_tokens: int, stats: Stats) -> str:
        import numpy as np

        n, tok = self.SEQ, self._tok
        tok.src_lang = src_code
        x = tok(text, return_tensors="np", truncation=True, max_length=n, padding="max_length")
        mask = x["attention_mask"].astype(np.int64)
        t0 = time.perf_counter()
        hidden = self._enc({"input_ids": x["input_ids"].astype(np.int64), "attention_mask": mask})[0]
        t1 = time.perf_counter()
        eos = tok.eos_token_id
        ids = [eos, tok.convert_tokens_to_ids(tgt_code)]  # decoder start, forced target-language token
        for _ in range(min(max_new_tokens, n - len(ids))):
            inp = np.full((1, n), tok.pad_token_id, dtype=np.int64)
            inp[0, : len(ids)] = ids
            logits = self._dec({"encoder_attention_mask": mask, "input_ids": inp, "encoder_hidden_states": hidden})[0]
            nxt = int(logits[0, len(ids) - 1].argmax())
            if nxt == eos:
                break
            ids.append(nxt)
        stats.prompt_tokens += int(mask.sum())
        stats.prompt_seconds += t1 - t0
        stats.decode_seconds += time.perf_counter() - t1
        stats.new_tokens += len(ids) - 2
        stats.decode_tokens += len(ids) - 2
        return tok.decode(ids[2:], skip_special_tokens=True).strip()

    def translate(self, text, src, tgt, max_new_tokens) -> str:
        self.load()
        if not src:
            raise BackendError("NLLB needs src_lang")
        try:
            src_code, tgt_code = NLLB_LANG_CODES[src], NLLB_LANG_CODES[tgt]
        except KeyError as e:
            raise BackendError(f"unsupported language code {e}") from None
        stats = Stats()
        out = "".join(
            self._translate_chunk(c, src_code, tgt_code, max_new_tokens, stats) + sep for c, sep in self._pieces(text)
        )
        self.last_stats = stats
        return out.strip()


class StubBackend:
    """Echo backend for tests / wiring checks without a model."""

    def __init__(self, entry: ModelEntry, device: str = "CPU"):
        self.entry, self._loaded = entry, False
        self.last_stats = Stats(prompt_tokens=3, new_tokens=4, prompt_seconds=0.01, decode_seconds=0.2, decode_tokens=3)

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def chat(self, messages, max_new_tokens, temperature) -> str:
        self.load()
        return f"[stub] {messages[-1]['content']}"

    def chat_stream(self, messages, max_new_tokens, temperature):
        yield self.chat(messages, max_new_tokens, temperature)

    def translate(self, text, src, tgt, max_new_tokens) -> str:
        self.load()
        return f"[stub {src}->{tgt}] {text}"

    def translate_stream(self, text, src, tgt, max_new_tokens):
        words = self.translate(text, src, tgt, max_new_tokens).split(" ")
        for i, word in enumerate(words):
            yield word if i == 0 else " " + word


def make_backend(
    entry: ModelEntry, device: str, stub: bool = False, fallback: str = "none", cache_dir: Path | None = None
) -> ModelBackend:
    if stub:
        return StubBackend(entry)
    backends = {"openvino": LLMBackend, "openvino_vlm": VLMBackend, "openvino_seq2seq": Seq2SeqBackend}
    return backends[entry.backend](entry, device, fallback, cache_dir)
