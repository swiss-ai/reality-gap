"""Utility functions for audio tokenization."""


def __getattr__(name):
    if name in ("add_audio_tokens", "get_audio_vocab_size_from_tokenizer"):
        from .omni_tokenizer import add_audio_tokens, get_audio_vocab_size_from_tokenizer
        return {"add_audio_tokens": add_audio_tokens, "get_audio_vocab_size_from_tokenizer": get_audio_vocab_size_from_tokenizer}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "add_audio_tokens",
    "get_audio_vocab_size_from_tokenizer",
]
