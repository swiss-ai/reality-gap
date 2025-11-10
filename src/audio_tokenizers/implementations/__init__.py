"""
Audio tokenizer implementations.

Each tokenizer here is automatically registered when imported.
"""

# Import all tokenizer implementations
# This triggers their registration via the metaclass
# Use try-except to handle missing dependencies gracefully

from .neucodec import NeuCodecTokenizer, DistilledNeuCodecTokenizer
from .tadicodec import TaDiCodecTokenizer
from .xcodec2 import XCodec2Tokenizer
from .cosyvoice2 import CosyVoice2Tokenizer
from .glm4voice import GLM4VoiceTokenizer
from .wavtokenizer import WavTokenizerWrapper

# Add more tokenizers as they're implemented:
# from .encodec import EncodecTokenizer
# from .dac import DACTokenizer
# from .soundstream import SoundStreamTokenizer

__all__ = [
    'NeuCodecTokenizer',
    'DistilledNeuCodecTokenizer',
    'TaDiCodecTokenizer',
    'XCodec2Tokenizer',
    'CosyVoice2Tokenizer',
    'GLM4VoiceTokenizer',
    'WavTokenizerWrapper',
]
