"""
Audio Tokenizers - Unified benchmark framework for audio tokenizers.

Easy imports:
    from audio_tokenizers import get_tokenizer, list_tokenizers
    
    # Get a tokenizer
    tokenizer = get_tokenizer('neucodec')
    
    # Or import directly
    from audio_tokenizers import NeuCodecTokenizer
"""

from .base.tokenizer_base import (
    BaseAudioTokenizer,
    get_tokenizer,
    list_tokenizers
)

# Import all implementations to trigger registration
from .implementations import *

# Make common functions available at package level
__all__ = [
    'BaseAudioTokenizer',
    'get_tokenizer',
    'list_tokenizers',
]