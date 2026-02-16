"""
Omni-Tokenizer Audio Extension Module

This module provides utilities for adding audio tokens to omni-tokenizers.
Supports WavTokenizer with configurable codebook sizes.

Main functions:
- add_audio_tokens: Add audio tokens to an existing omni-tokenizer
- get_audio_vocab_size_from_tokenizer: Auto-detect audio vocab size from tokenizer class

Usage:
    from audio_tokenization.utils.omni_tokenizer import add_audio_tokens

    # Add audio tokens to existing omni-tokenizer (vocab size auto-detected)
    tokenizer, stats = add_audio_tokens(
        input_tokenizer_path="/path/to/omni-tokenizer",
        output_path="/path/to/output",
    )
"""

from .core import (
    add_audio_tokens,
    get_audio_vocab_size_from_tokenizer,
)

__all__ = [
    "add_audio_tokens",
    "get_audio_vocab_size_from_tokenizer",
]
