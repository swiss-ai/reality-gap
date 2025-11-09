"""
Audio tokenizer implementations.

Each tokenizer here is automatically registered when imported.
"""

# Import all tokenizer implementations
# This triggers their registration via the metaclass

from .neucodec import NeuCodecTokenizer, DistilledNeuCodecTokenizer
from .tadicodec import TaDiCodecTokenizer
from .cosyvoice2 import CosyVoice2Tokenizer

# Add more tokenizers as they're implemented:
from .glm4voice import GLM4VoiceTokenizer
# from .encodec import EncodecTokenizer
# from .dac import DACTokenizer
# from .soundstream import SoundStreamTokenizer

__all__ = [
    'NeuCodecTokenizer',
    'DistilledNeuCodecTokenizer',
    'TaDiCodecTokenizer',
    'CosyVoice2Tokenizer',
    'GLM4VoiceTokenizer',
]