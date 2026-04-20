"""Speech token generation from text using TTS backends."""

from .normalizer import TextNormalizer, PolishTextNormalizer

# Defer torch-dependent imports to avoid ImportError when torch is unavailable
# (e.g. local testing of the normalizer only).


def __getattr__(name):
    if name in ("TTSBackend", "TTSOutput"):
        from .base import TTSBackend, TTSOutput

        globals()["TTSBackend"] = TTSBackend
        globals()["TTSOutput"] = TTSOutput
        return globals()[name]
    if name == "save_sample":
        from .saver import save_sample

        globals()["save_sample"] = save_sample
        return save_sample
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TTSBackend",
    "TTSOutput",
    "TextNormalizer",
    "PolishTextNormalizer",
    "save_sample",
]
