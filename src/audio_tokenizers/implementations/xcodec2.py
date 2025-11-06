"""
XCodec2 tokenizer implementation using the unified base class.
Automatically registered and accessible via get_tokenizer('xcodec2').

Notes based on tokenizers/xcodec2/usage_example.py:
- Input waveform expected as mono tensor of shape (1, T)
- Model path: "HKUSTAudio/xcodec2"
- encode_code / decode_code APIs are provided by XCodec2Model
"""

import torch

from ..base import BaseAudioTokenizer


class XCodec2Tokenizer(BaseAudioTokenizer):
    """XCodec2 tokenizer wrapper."""

    name = "xcodec2"
    repo_path = None  # installed via pip, not a local repo
    default_checkpoint = "HKUSTAudio/xcodec2"
    default_sample_rate = 16000

    def _load_model(self):
        """Load XCodec2 model and set properties."""
        from xcodec2.modeling_xcodec2 import XCodec2Model

        self.model = XCodec2Model.from_pretrained(self.checkpoint)
        self.model.to(self.device).eval()

        # Set properties - try to extract from model config, otherwise use defaults
        # Based on usage_example.py: input/output is 16kHz
        self._output_sample_rate = 16000
        
        # Try to get from model config if available
        if hasattr(self.model, 'config'):
            self._codebook_size = getattr(self.model.config, 'codebook_size', 0)
            self._downsample_rate = getattr(self.model.config, 'downsample_rate', 0)
            self._frame_rate = getattr(self.model.config, 'frame_rate', 0)
        else:
            # Defaults (will need to be adjusted based on actual XCodec2 specs)
            self._codebook_size = 0  # Unknown, will be set after first encode if needed
            self._downsample_rate = 0  # Unknown, will be calculated from token shape
            self._frame_rate = 0  # Unknown, will be calculated from downsample_rate

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to discrete tokens.

        Args:
            audio: Tensor from preprocess_audio, shape (B, C, T)
        Returns:
            tokens: Tensor shaped like (1, 1, T') from XCodec2 encode.
        """
        # XCodec2 expects (1, T) - convert from (B, C, T) format
        # Remove batch and channel dimensions
        if audio.dim() == 3:
            # (B, C, T) -> remove channel dimension -> (B, T)
            audio = audio.squeeze(1)  # Remove channel dimension
        
        # Ensure shape is (1, T) - XCodec2 expects this format
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # (T) -> (1, T)
        elif audio.dim() == 2 and audio.shape[0] > 1:
            # Multiple batches not supported
            raise ValueError(f"XCodec2 only supports single batch input, got batch size {audio.shape[0]}")
        
        audio = audio.to(self.model.device)

        with torch.no_grad():
            tokens = self.model.encode_code(input_waveform=audio)

        return tokens

    def decode_tokens(self, tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decode tokens back to waveform.

        Returns:
            audio: Tensor shaped like (1, 1, T') from XCodec2 decode.
        """
        tokens = tokens.to(self.model.device)
        with torch.no_grad():
            audio = self.model.decode_code(tokens).cpu()

        return audio

    @property
    def codebook_size(self) -> int:
        """Vocabulary size."""
        return self._codebook_size if hasattr(self, '_codebook_size') else 0

    @property
    def downsample_rate(self) -> int:
        """Samples per token."""
        return self._downsample_rate if hasattr(self, '_downsample_rate') else 0

    @property
    def output_sample_rate(self) -> int:
        """Output sample rate: 16 kHz."""
        return self._output_sample_rate if hasattr(self, '_output_sample_rate') else 16000

    @property
    def frame_rate(self) -> int:
        """Frame rate in Hz."""
        return self._frame_rate if hasattr(self, '_frame_rate') else 0
