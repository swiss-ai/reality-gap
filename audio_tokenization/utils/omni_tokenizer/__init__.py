"""
Audio Omni-Tokenizer utilities.

Functions for creating audio omni-tokenizers by extending text tokenizers
with audio codec tokens.
"""

from .core import (
    create_audio_base_tokenizer,
    load_audio_tokenizer,
    load_audio_token_mapping,
    get_audio_token_id,
)

__all__ = [
    'create_audio_base_tokenizer',
    'load_audio_tokenizer',
    'load_audio_token_mapping',
    'get_audio_token_id',
]

