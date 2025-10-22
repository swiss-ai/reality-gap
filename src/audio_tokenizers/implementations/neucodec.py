"""
NeuCodec tokenizer implementation using the unified base class.
Automatically registered and accessible via get_tokenizer('neucodec').
"""

import torch

# Import base class
from ..base import BaseAudioTokenizer

# NeuCodec will be imported after repo path is added
class NeuCodecTokenizer(BaseAudioTokenizer):
    """NeuCodec audio tokenizer wrapper."""
    
    name = "neucodec"
    repo_path = "neucodec"
    default_checkpoint = "neuphonic/neucodec"
    default_sample_rate = 16000
    
    def _load_model(self):
        """Load NeuCodec model."""
        # Import here after repo path is added
        from neucodec.model import NeuCodec
        
        self.model = NeuCodec.from_pretrained(self.checkpoint)
        
        # Set model properties
        self._output_sample_rate = 24000  # NeuCodec upsamples to 24kHz
        self._codebook_size = 2**16  # 16-bit FSQ
        self._downsample_rate = 320  # 16kHz to 50Hz tokens
    
    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to NeuCodec tokens.

        Optimal workflow:
        - Feature extraction requires CPU (numpy conversion)
        - Model processing happens on GPU
        - Returns tokens on GPU for downstream processing
        """
        # Move to CPU only if needed (feature extractor requires numpy)
        if audio.is_cuda:
            audio_cpu = audio.cpu()
        else:
            audio_cpu = audio

        # NeuCodec handles: CPU feature extraction → GPU model processing
        tokens = self.model.encode_code(audio_cpu)

        # Return tokens on GPU for efficient downstream processing
        return tokens

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode NeuCodec tokens to audio.

        Tokens should be on model's device for decoding.
        Returns audio on GPU for efficient processing.
        """
        # Ensure tokens are on the correct device
        tokens = tokens.to(self.model.device)
        reconstructed = self.model.decode_code(tokens)

        # Return on GPU for downstream processing
        return reconstructed
    
    @property
    def codebook_size(self) -> int:
        return self._codebook_size
    
    @property
    def downsample_rate(self) -> int:
        return self._downsample_rate
    
    @property
    def output_sample_rate(self) -> int:
        return self._output_sample_rate


# Optional: Distilled version
class DistilledNeuCodecTokenizer(NeuCodecTokenizer):
    """Distilled NeuCodec - faster, smaller version."""
    
    name = "neucodec-distilled"
    default_checkpoint = "neuphonic/distill-neucodec"


# The tokenizers are now automatically registered!
# Usage:
# from tokenizer_base import get_tokenizer
# tokenizer = get_tokenizer('neucodec')