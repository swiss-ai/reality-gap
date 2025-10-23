"""
NeuCodec tokenizer implementation using the unified base class.
Automatically registered and accessible via get_tokenizer('neucodec').

NeuCodec: Neural audio codec with high-quality 50 Hz tokenization
- Single-stream output (FSQ flattens multi-level quantization)
- No delay pattern needed - perfect for standard AR language models
"""

import torch

# Import base class
from ..base import BaseAudioTokenizer


class NeuCodecTokenizer(BaseAudioTokenizer):
    """NeuCodec tokenizer wrapper.

    Specifications:
    - Frame rate: 50 Hz (50 tokens/second)
    - Input sample rate: 16 kHz
    - Output sample rate: 24 kHz
    - Quantization: FSQ (Finite Scalar Quantization) with 8 levels
    - Token stream: Single stream (FSQ flattens the codebooks)
    - Codebook: 2^16 = 65,536 unique tokens (16-bit FSQ)
    """

    name = "neucodec"
    repo_path = "neucodec"
    default_checkpoint = "neuphonic/neucodec"
    default_sample_rate = 16000  # NeuCodec is designed for 16kHz input

    def _load_model(self):
        """Load NeuCodec model."""
        # Import after repo path is added to sys.path
        from neucodec.model import NeuCodec

        # Load model (auto-downloads if needed)
        self.model = NeuCodec.from_pretrained(self.checkpoint)

        # Get model's expected sample rate
        self._model_sample_rate = 16000  # NeuCodec always expects 16kHz
        self._output_sample_rate = getattr(self.model, 'sample_rate', 24000)  # Get from model if available

        # Set properties
        self._codebook_size = 2**16  # 16-bit FSQ = 65,536 tokens
        self._downsample_rate = 320  # 16000 / 50 = 320 samples per token
        self._frame_rate = 50  # 50 Hz token rate
    
    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to discrete tokens.

        Args:
            audio: Audio tensor, any shape accepted
                   NeuCodec handles normalization internally

        Returns:
            tokens: Discrete token IDs [1, 1, seq_len] - single stream
        """
        # NeuCodec's feature extractor requires CPU tensors
        audio = audio.cpu() if audio.is_cuda else audio

        # NeuCodec internally:
        # 1. Extracts features on CPU (numpy conversion)
        # 2. Moves to GPU for model processing
        # 3. Returns FSQ tokens as single stream
        tokens = self.model.encode_code(audio)

        return tokens  # [1, 1, seq_len]

    def decode_tokens(self, tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decode discrete tokens to audio.

        Args:
            tokens: Discrete token IDs [1, 1, seq_len]

        Returns:
            audio: Reconstructed audio [1, 1, T] at 24kHz
        """
        # Move tokens to model's device
        tokens = tokens.to(self.model.device)

        # Decode to audio
        audio = self.model.decode_code(tokens)

        return audio  # [1, 1, T] at 24kHz
    
    @property
    def codebook_size(self) -> int:
        """Vocabulary size: 65,536 tokens (16-bit FSQ)."""
        return self._codebook_size

    @property
    def downsample_rate(self) -> int:
        """Samples per token: 320 (16000 Hz / 50 Hz)."""
        return self._downsample_rate

    @property
    def output_sample_rate(self) -> int:
        """Output sample rate: 24 kHz."""
        return self._output_sample_rate

    @property
    def frame_rate(self) -> int:
        """Frame rate in Hz: 50 tokens/second."""
        return self._frame_rate


# Optional: Distilled version
class DistilledNeuCodecTokenizer(NeuCodecTokenizer):
    """Distilled NeuCodec - faster, smaller version."""
    
    name = "neucodec-distilled"
    default_checkpoint = "neuphonic/distill-neucodec"


# The tokenizers are now automatically registered!
# Usage:
# from tokenizer_base import get_tokenizer
# tokenizer = get_tokenizer('neucodec')