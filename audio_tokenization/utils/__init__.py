"""Utility functions for audio tokenization."""

from .omni_tokenizer import (
    add_audio_tokens,
    get_audio_vocab_size_from_tokenizer,
)

__all__ = [
    "add_audio_tokens",
    "get_audio_vocab_size_from_tokenizer",
]
