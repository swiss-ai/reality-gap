"""
UniCodec tokenizer implementation using the unified base class.
Automatically registered and accessible via get_tokenizer('unicodec').

UniCodec: Unified audio codec supporting speech, music, and sound
- Single codebook with 16384 tokens
- 75 Hz frame rate
- Domain-adaptive with MoE strategy
"""

import torch
from huggingface_hub import hf_hub_download

from ..base import BaseAudioTokenizer


class UniCodecTokenizer(BaseAudioTokenizer):
    """UniCodec tokenizer wrapper.

    Specifications:
    - Frame rate: 75 Hz (75 tokens/second)
    - Input sample rate: 24 kHz
    - Output sample rate: 24 kHz
    - Quantization: Single codebook, 16384 tokens
    - Domains: speech (0), music (1), sound (2)
    """

    name = "unicodec"
    repo_path = "unicodec"
    default_checkpoint = "Yidiii/UniCodec_ckpt"
    default_sample_rate = 24000

    def __init__(self, checkpoint=None, device="cuda", sample_rate=None, domain="0", **kwargs):
        """
        Args:
            checkpoint: HuggingFace model ID or local path
            device: Device to use
            sample_rate: Input sample rate (must be 24000)
            domain: Audio domain - "0" (speech), "1" (music), "2" (sound)
        """
        self.domain = str(domain)
        super().__init__(checkpoint, device, sample_rate, **kwargs)

    def _load_model(self):
        """Load UniCodec model."""
        import sys
        from pathlib import Path

        # Add UniCodec-fix repo to path (use the unicodec package with fixes)
        unicodec_path = Path(__file__).parent.parent.parent / "repos" / "unicodec"
        if str(unicodec_path) not in sys.path:
            sys.path.insert(0, str(unicodec_path))

        from unicodec.decoder.pretrained import Unicodec

        # Get config and checkpoint paths
        config_path = unicodec_path / "configs" / "unicodec_frame75_10s_nq1_code16384_dim512_finetune.yaml"

        # Download checkpoint from HuggingFace
        if self.checkpoint == self.default_checkpoint:
            ckpt_path = hf_hub_download(repo_id=self.checkpoint, filename="unicode.ckpt")
        else:
            ckpt_path = self.checkpoint

        # Load model
        self.codec = Unicodec.from_pretrained0802(str(config_path), ckpt_path)
        self.codec = self.codec.to(self.device)
        self.codec.eval()

        # Set properties
        self._codebook_size = 16384
        self._downsample_rate = 320  # 24000 / 75
        self._frame_rate = 75
        self._output_sample_rate = 24000

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to discrete token indices.

        Args:
            audio: Audio tensor [batch, channels, samples] at 24kHz

        Returns:
            tokens: Token indices [batch, seq_len] (single codebook)
        """
        # Handle input shape
        if audio.dim() == 3:
            audio = audio.squeeze(0)  # Remove batch dim
        if audio.dim() == 2:
            audio = audio.squeeze(0)  # Remove channel dim

        audio = audio.to(self.device)
        bandwidth_id = torch.tensor([0]).to(self.device)

        # Encode - returns (features, discrete_codes)
        features, discrete_codes = self.codec.encode_infer(audio.unsqueeze(0), self.domain, bandwidth_id=bandwidth_id)

        # discrete_codes shape: [num_codebooks, batch, seq_len] -> [batch, seq_len]
        if discrete_codes.dim() == 3:
            tokens = discrete_codes.squeeze(0)  # [batch, seq_len]
        else:
            tokens = discrete_codes

        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        return tokens

    def decode_tokens(self, tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decode token indices to audio.

        Args:
            tokens: Token indices [batch, seq_len] or [num_codebooks, batch, seq_len]

        Returns:
            audio: Reconstructed audio [1, 1, T] at 24kHz
        """
        tokens = tokens.to(self.device)
        bandwidth_id = torch.tensor([0]).to(self.device)

        # Handle token shapes - need [num_codebooks, batch, seq_len]
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)  # [1, batch, seq_len]

        # Convert codes to features
        features = self.codec.codes_to_features(tokens)

        # Decode features to audio
        audio = self.codec.decode(features, bandwidth_id=bandwidth_id)

        # Reshape to [1, 1, T]
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(0)

        return audio

    @property
    def codebook_size(self) -> int:
        return self._codebook_size

    @property
    def downsample_rate(self) -> int:
        return self._downsample_rate

    @property
    def output_sample_rate(self) -> int:
        return self._output_sample_rate

    @property
    def frame_rate(self) -> int:
        return self._frame_rate
