"""The server serves only TranslateGemma now. The backend code still supports a decoder-only LLM and an NLLB-style
seq2seq model, and the tests exercise it through these two entries, which are registered for the test run only."""

from server.registry import REGISTRY, ROOT, ModelEntry

REGISTRY["nllb-200-600m"] = ModelEntry(
    short_name="nllb-200-600m", backend="openvino_seq2seq", path=ROOT / "models/nllb-200-distilled-600M-int4-sym-ov",
    owned_by="facebook", note="Pure MT seq2seq, 200 langs. Needs src/tgt lang.", chat=False,
    languages_supported=200, max_prompt_len=512,
)
REGISTRY["eurollm-1.7b-it"] = ModelEntry(
    short_name="eurollm-1.7b-it", backend="openvino", path=ROOT / "models/eurollm-1.7b-it-int4-sym-ov",
    owned_by="utter-project", note="EU-trained translation LLM. ET/FI/all EU langs <-> EN.", chat=True,
    languages_supported=35, npu_piece_tokens=1024,
)
