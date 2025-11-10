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

# Optional tokenizers with external dependencies
try:
    from .cosyvoice2 import CosyVoice2Tokenizer
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import CosyVoice2Tokenizer: {e}. Skipping.")
    CosyVoice2Tokenizer = None

# Add more tokenizers as they're implemented:
try:
    from .glm4voice import GLM4VoiceTokenizer
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import GLM4VoiceTokenizer: {e}. Skipping.")
    GLM4VoiceTokenizer = None

# from .encodec import EncodecTokenizer
# from .dac import DACTokenizer
# from .soundstream import SoundStreamTokenizer

__all__ = [
    'NeuCodecTokenizer',
    'DistilledNeuCodecTokenizer',
    'TaDiCodecTokenizer',
    'XCodec2Tokenizer',
]

# Only add optional tokenizers to __all__ if they were successfully imported
if CosyVoice2Tokenizer is not None:
    __all__.append('CosyVoice2Tokenizer')
if GLM4VoiceTokenizer is not None:
    __all__.append('GLM4VoiceTokenizer')