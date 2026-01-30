"""
WavTokenizer implementation wrapper for benchmark-audio-tokenizer.

WavTokenizer is a SOTA discrete codec model achieving 40/75 tokens per second
for audio language modeling, with strong reconstruction results.
Paper: https://arxiv.org/abs/2408.16532

Two variants available:
- wavtokenizer-40: 40 tokens/second (large-600 model)
- wavtokenizer-75: 75 tokens/second (large-320 model)
"""

import os
import sys
import torch
import torchaudio
import numpy as np
from typing import Tuple, Dict, Optional, Any
import logging

# Add WavTokenizer repo to path
wavtokenizer_path = os.path.join(os.path.dirname(__file__), '..', '..', 'repos', 'wavtokenizer')
sys.path.insert(0, wavtokenizer_path)

from encoder.utils import convert_audio
from decoder.pretrained import WavTokenizer as OriginalWavTokenizer

from ..base import BaseAudioTokenizer

logger = logging.getLogger(__name__)


class WavTokenizerBase(BaseAudioTokenizer):
    """
    Base class for WavTokenizer variants.

    Key features:
    - Single codebook with 4096 codes
    - Supports speech, audio, and music
    """

    # To be overridden by subclasses
    name = "wavtokenizer"
    tokens_per_second = 40
    model_variant = "large-600"
    _downsample_rate = 600

    def __init__(self, device: str = "cuda", checkpoint: Optional[str] = None):
        """
        Initialize WavTokenizer.

        Args:
            device: Device to run the model on
            checkpoint: Optional path to local checkpoint
        """
        self.device = device
        self.checkpoint = checkpoint

        # Cache for resamplers (to avoid recreating and ensure correct device)
        self._resamplers = {}

        self._load_model()

    def _load_model(self):
        """Load the WavTokenizer model. To be implemented by subclasses."""
        raise NotImplementedError

    @property
    def sample_rate(self) -> int:
        """Input sample rate for the tokenizer."""
        return 24000  # WavTokenizer uses 24kHz

    @sample_rate.setter
    def sample_rate(self, value: int):
        """Setter for sample_rate (required by base class)."""
        if value != 24000:
            logger.warning(f"WavTokenizer uses fixed 24kHz sample rate, ignoring requested {value}Hz")

    @property
    def output_sample_rate(self) -> int:
        """Output sample rate for the decoder."""
        return 24000  # WavTokenizer outputs at 24kHz

    @output_sample_rate.setter
    def output_sample_rate(self, value: int):
        """Setter for output_sample_rate (for consistency)."""
        if value != 24000:
            logger.warning(f"WavTokenizer uses fixed 24kHz output rate, ignoring requested {value}Hz")

    @property
    def codebook_size(self) -> int:
        """Size of the codebook."""
        return 4096  # WavTokenizer uses 4096 codes

    @property
    def downsample_rate(self) -> int:
        """Downsampling rate from audio samples to tokens."""
        return self._downsample_rate

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to discrete tokens.

        Args:
            audio: Audio tensor (B, T) or (B, 1, T)

        Returns:
            tokens: Discrete token codes (B, N)
        """
        # Ensure correct shape
        if audio.dim() == 3:
            audio = audio.squeeze(1)  # Remove channel dimension

        # Move to device
        audio = audio.to(self.device)

        # bandwidth_id=0 is fixed (WavTokenizer uses single codebook)
        bandwidth_id = torch.tensor([0]).to(self.device)

        with torch.no_grad():
            _, discrete_codes = self.model.encode_infer(audio, bandwidth_id=bandwidth_id)

        # discrete_codes shape: (n_q, B, T) where n_q=1 for single quantizer
        # We want (B, T)
        if discrete_codes.dim() == 3:
            discrete_codes = discrete_codes.squeeze(0)

        return discrete_codes

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Decode tokens back to audio.

        Args:
            tokens: Token codes (B, N)

        Returns:
            audio: Audio tensor (B, 1, T)
        """
        # Move to device
        tokens = tokens.to(self.device)

        # Ensure tokens are integers
        tokens = tokens.long()

        # Add quantizer dimension if needed: (B, N) -> (1, B, N)
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)

        with torch.no_grad():
            # Convert codes to features
            features = self.model.codes_to_features(tokens)

            # Decode with bandwidth_id
            bandwidth_id = torch.tensor([0]).to(self.device)
            audio = self.model.decode(features, bandwidth_id=bandwidth_id)

        # Ensure output shape is (B, 1, T)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        return audio

    def encode(self, audio: torch.Tensor, sr: int = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Encode audio to discrete tokens.

        Args:
            audio: Audio tensor
            sr: Sample rate of input audio

        Returns:
            tokens: Discrete token tensor
            info: Dictionary with encoding information
        """
        if sr is None:
            sr = self.sample_rate

        # Convert to tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()

        # Resample if necessary (requires CPU for torchaudio.Resample)
        if sr != self.sample_rate:
            # Only move to CPU if resampling is needed
            audio = audio.cpu()
            # Use cached resampler or create new one
            resampler_key = f"{sr}_{self.sample_rate}"
            if resampler_key not in self._resamplers:
                self._resamplers[resampler_key] = torchaudio.transforms.Resample(sr, self.sample_rate)

            resampler = self._resamplers[resampler_key]

            # Ensure correct shape for resampling (B, C, T)
            if audio.dim() == 2:
                audio = audio.unsqueeze(1)  # Add channel dimension

            audio = resampler(audio)

            # Convert to mono if needed
            if audio.shape[1] > 1:
                audio = audio.mean(1, keepdim=True)

            # Remove channel dimension for WavTokenizer
            if audio.dim() == 3:
                audio = audio.squeeze(1)

        # Move to target device
        audio = audio.to(self.device)

        # Encode
        tokens = self.encode_audio(audio)

        info = {
            "num_tokens": tokens.numel(),
            "token_shape": list(tokens.shape),
            "sample_rate": self.sample_rate,
            "tokens_per_second": self.tokens_per_second
        }

        return tokens, info

    def decode(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Decode discrete tokens back to audio.

        Args:
            tokens: Discrete token tensor

        Returns:
            audio: Reconstructed audio tensor
            info: Dictionary with decoding information
        """
        # Decode
        audio = self.decode_tokens(tokens)

        # Remove channel dimension if single channel
        if audio.dim() == 3 and audio.shape[1] == 1:
            audio = audio.squeeze(1)

        info = {
            "output_sample_rate": self.output_sample_rate,
            "num_tokens": tokens.shape[-1]
        }

        return audio, info

    def reconstruct(self, audio: torch.Tensor, sr: int = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Encode and decode audio for reconstruction.

        Args:
            audio: Input audio tensor
            sr: Sample rate of input audio

        Returns:
            reconstructed: Reconstructed audio
            info: Combined encoding and decoding information
        """
        # Encode
        tokens, encode_info = self.encode(audio, sr)

        # Decode
        reconstructed, decode_info = self.decode(tokens)

        # Combine info
        info = {
            **encode_info,
            **decode_info,
            "num_tokens": tokens.numel(),
            "token_shape": list(tokens.shape)
        }

        return reconstructed, info


class WavTokenizer40(WavTokenizerBase):
    """
    WavTokenizer with 40 tokens per second (large-600 model).

    - 40 tokens/second
    - Single codebook with 4096 codes
    - Trained on 80,000 hours of speech, audio, and music
    """

    name = "wavtokenizer-40"
    tokens_per_second = 40
    model_variant = "large-600"
    _downsample_rate = 600  # 24000 / 40 = 600

    def _load_model(self):
        """Load the WavTokenizer large-600 model (40 tokens/s)."""
        try:
            # Model paths
            cache_dir = "/capstor/store/cscs/swissai/infra01/MLLM/wavtokenizer"
            os.makedirs(cache_dir, exist_ok=True)

            # Use large-600 model (best quality, 40 tokens/s)
            model_name = "wavtokenizer_large_unify_600_24k.ckpt"
            config_name = "wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"

            model_path = os.path.join(cache_dir, model_name)
            config_path = os.path.join(wavtokenizer_path, "configs", config_name)

            # Download from HuggingFace if not cached
            if not os.path.exists(model_path):
                logger.info(f"Downloading WavTokenizer large-600 (40 tokens/s) from HuggingFace...")

                from huggingface_hub import hf_hub_download
                downloaded_path = hf_hub_download(
                    repo_id="novateur/WavTokenizer-large-unify-40token",
                    filename=model_name,
                    cache_dir=cache_dir,
                    local_dir=cache_dir
                )
                logger.info(f"Downloaded to {downloaded_path}")

            # Load the model
            logger.info(f"Loading WavTokenizer large-600 (40 tokens/s) from {model_path}")

            self.model = OriginalWavTokenizer.from_pretrained0802(config_path, model_path)
            self.model = self.model.to(self.device)
            self.model.eval()

            # Compile encoder for ~1.3x speedup (fuses padding + conv operations)
            self.model.feature_extractor.encodec.encoder = torch.compile(
                self.model.feature_extractor.encodec.encoder
            )

            logger.info(f"WavTokenizer large-600 loaded successfully (40 tokens/s, compiled)")

        except Exception as e:
            logger.error(f"Error loading WavTokenizer: {e}")
            raise


class WavTokenizer75(WavTokenizerBase):
    """
    WavTokenizer with 75 tokens per second (large-320 model).

    NOT YET IMPLEMENTED: No unified 75Hz model available yet.
    Only speech-only model exists (novateur/WavTokenizer-large-speech-75token).

    - 75 tokens/second
    - Single codebook with 4096 codes
    """

    name = "wavtokenizer-75"
    tokens_per_second = 75
    model_variant = "large-320"
    _downsample_rate = 320  # 24000 / 75 = 320

    def _load_model(self):
        """Load the WavTokenizer large-320 model (75 tokens/s)."""
        raise NotImplementedError(
            "WavTokenizer 75Hz unified model not yet available. "
            "Only speech-only model exists: novateur/WavTokenizer-large-speech-75token"
        )


# Alias for backward compatibility
WavTokenizerWrapper = WavTokenizer40
