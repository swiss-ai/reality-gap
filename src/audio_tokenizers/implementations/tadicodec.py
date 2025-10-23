"""
TaDiCodec tokenizer implementation using the unified base class.
Automatically registered and accessible via get_tokenizer('tadicodec').

TaDiCodec: Text-aware Diffusion Speech Tokenizer (NeurIPS 2025)
Based on: https://github.com/AmphionTeam/TaDiCodec

Key advantages for language modeling:
- Single-layer codebook (no multi-stream complexity)
- Ultra-low frame rate (6.25 Hz = 160ms per token)
- Single stream of discrete tokens (perfect for standard AR)
- Codebook size: 16,384 (2^14)
"""

import torch

# Import base class
from ..base import BaseAudioTokenizer


class TaDiCodecTokenizer(BaseAudioTokenizer):
    """TaDiCodec tokenizer wrapper.

    Text-aware speech tokenizer with ultra-low frame rate (6.25 Hz).
    Uses text transcripts for better reconstruction quality.

    Specifications:
    - Frame rate: 6.25 Hz (8x less than NeuCodec)
    - Sample rate: 24 kHz (input/output)
    - Bitrate: 0.0875 kbps
    - Codebook: 16,384 tokens (single layer)
    """

    name = "tadicodec"
    repo_path = "tadicodec"
    default_checkpoint = "amphion/TaDiCodec"
    default_sample_rate = None  # Must be specified by user

    def __init__(self, sample_rate: int, **kwargs):
        """Initialize TaDiCodec tokenizer.

        Args:
            sample_rate: Input audio sample rate (required)
            **kwargs: Other arguments passed to base class
        """
        if sample_rate is None:
            raise ValueError("sample_rate must be specified for TaDiCodec tokenizer")
        super().__init__(sample_rate=sample_rate, **kwargs)

    def _load_model(self):
        """Load TaDiCodec tokenizer components."""
        # Import here after repo path is added to sys.path
        from models.tts.tadicodec.inference_tadicodec import TaDiCodecPipline

        # Load the full pipeline (auto-downloads from HuggingFace)
        self.pipeline = TaDiCodecPipline.from_pretrained(
            ckpt_dir=self.checkpoint,
            device=self.device,
            auto_download=True
        )

        # Store components for clean access
        self.model = self.pipeline.tadicodec  # Core encoder/decoder/quantizer
        self.mel_model = self.pipeline.mel_model  # Mel feature extractor
        self.vocoder = self.pipeline.vocoder_model  # Mel to waveform

        # Store configs
        self.model_cfg = self.pipeline.cfg.model.tadicodec
        self.preprocess_cfg = self.pipeline.cfg.preprocess

        # Get model's expected sample rate from config
        self._model_sample_rate = self.preprocess_cfg.sample_rate  # What TaDiCodec expects internally
        self._output_sample_rate = self.preprocess_cfg.sample_rate  # Same for output

        # self.sample_rate is the user's input sample rate (from __init__)

        # Calculate frame rate from config values
        hop_size = self.preprocess_cfg.hop_size  # 480
        downsample = self.model_cfg.down_sample_factor  # 8
        self._frame_rate = self._model_sample_rate / (hop_size * downsample)  # Model's rate / (hop * down)

        # Derive other properties
        self._downsample_rate = int(self._model_sample_rate / self._frame_rate)  # Samples per token at model rate
        self._n_quantizers = 1  # Single-layer codebook
        self._codebook_size = 2 ** self.model_cfg.vq_emb_dim  # 2^vq_emb_dim

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to discrete tokens.

        Args:
            audio: Audio tensor [T], [1, T], [C, T] or [1, C, T]
                   Processes single audio (batch=1) for text-audio alignment

        Returns:
            tokens: Discrete token IDs [1, seq_len] - single codebook stream
        """
        import math
        import torchaudio.functional as F

        # Normalize shape to [1, T]
        if audio.ndim == 3:
            audio = audio[0, 0, :]  # Take first channel of first batch
        elif audio.ndim == 2:
            audio = audio[0, :]  # Take first channel
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)  # [T] → [1, T]

        # Resample to 24kHz to ensure consistent 6.25 Hz frame rate
        # (Model accepts any rate but was trained on 24kHz)
        if self.sample_rate != self._model_sample_rate:
            audio = F.resample(audio, orig_freq=self.sample_rate, new_freq=self._model_sample_rate)

        # Move to device and ensure float
        audio = audio.to(self.device).float()

        with torch.no_grad():
            # Extract mel spectrogram
            mel = self.mel_model(audio)  # [1, n_mels, T]
            mel = mel.transpose(1, 2)  # [1, T, n_mels]

            # Normalize
            mel = (mel - self.preprocess_cfg.mel_mean) / math.sqrt(self.preprocess_cfg.mel_var)

            # Encode to tokens
            mel_mask = torch.ones(mel.shape[0], mel.shape[1], device=self.device)
            _, indices = self.model.encode(mel, mel_mask)

        return indices  # [1, seq_len]

    def decode_tokens(self, tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decode discrete tokens to audio.

        Args:
            tokens: Discrete token IDs [1, seq_len]
            **kwargs: Optional 'text' for text-aware decoding

        Returns:
            audio: Reconstructed audio [1, 1, T] at 24kHz
        """
        tokens = tokens.to(self.device)

        with torch.no_grad():
            # Convert token IDs to continuous embeddings
            vq_emb = self.model.index2vq(tokens)

            # Text conditioning (use space as default to avoid bugs)
            text = kwargs.get('text', ' ')
            text_ids = self.pipeline.tokenize_text(text)

            # Decode to mel spectrogram via diffusion
            mel = self.pipeline.decode(
                vq_emb=vq_emb,
                text_token_ids=text_ids,
                prompt_mel=None,
                n_timesteps=32,  # Diffusion steps
                cfg=1.0,  # No classifier-free guidance
                rescale_cfg=0.75
            )

            # Convert mel to waveform
            audio = self.vocoder(mel.transpose(1, 2))  # [1, 1, T]

        return audio

    @property
    def codebook_size(self) -> int:
        """Vocabulary size: 16,384 tokens."""
        return self._codebook_size

    @property
    def downsample_rate(self) -> int:
        """Samples per token: 3840 (24000 Hz / 6.25 Hz)."""
        return self._downsample_rate

    @property
    def output_sample_rate(self) -> int:
        """Output sample rate: 24 kHz."""
        return self._output_sample_rate

    @property
    def frame_rate(self) -> int:
        """Frame rate in Hz: 6.25 tokens/second."""
        return self._frame_rate


# The tokenizer is now automatically registered!
# Usage:
# from audio_tokenizers import get_tokenizer
# tokenizer = get_tokenizer('tadicodec')
