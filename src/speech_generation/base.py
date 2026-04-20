"""Abstract TTS backend interface and output types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List

import torch


@dataclass
class TTSOutput:
    """Output from a TTS backend."""

    speech_tokens: torch.Tensor  # (T,) int64, values in [0, codebook_size)
    codebook_size: int  # e.g. 6561 for CosyVoice2
    token_rate_hz: float  # e.g. 25.0
    audio: Optional[torch.Tensor] = None  # (samples,) float32 if rendered
    audio_sample_rate: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        return self.speech_tokens.numel() / self.token_rate_hz

    @property
    def num_tokens(self) -> int:
        return self.speech_tokens.numel()


class TTSBackend(ABC):
    """Abstract TTS backend that produces speech tokens from text.

    Subclass this to add new TTS engines (CosyVoice2, Parler-TTS, etc.).
    """

    @abstractmethod
    def load_model(self, device: str = "cuda") -> None:
        """Load model weights onto the specified device."""

    @abstractmethod
    def generate(
        self,
        text: str,
        reference_audio: Optional[torch.Tensor] = None,
        reference_audio_sr: Optional[int] = None,
        speaker_id: Optional[str] = None,
        render_audio: bool = False,
        **kwargs,
    ) -> TTSOutput:
        """Generate speech tokens from a single text input.

        Args:
            text: Normalized input text.
            reference_audio: Reference audio for voice cloning (1D tensor, mono).
            reference_audio_sr: Sample rate of reference audio.
            speaker_id: Named speaker for SFT-style backends.
            render_audio: If True, also synthesize waveform for sanity checking.

        Returns:
            TTSOutput with speech tokens and optional audio.
        """

    def generate_batch(
        self,
        texts: List[str],
        reference_audio: Optional[torch.Tensor] = None,
        reference_audio_sr: Optional[int] = None,
        render_audio: bool = False,
        **kwargs,
    ) -> List[TTSOutput]:
        """Generate speech tokens for a batch of texts.

        Default implementation processes sequentially. Override for true batching.
        """
        return [
            self.generate(
                text=t,
                reference_audio=reference_audio,
                reference_audio_sr=reference_audio_sr,
                render_audio=render_audio,
                **kwargs,
            )
            for t in texts
        ]

    @property
    @abstractmethod
    def codebook_size(self) -> int:
        """Size of the speech token vocabulary."""

    @property
    @abstractmethod
    def token_rate_hz(self) -> float:
        """Token rate in Hz (tokens per second of audio)."""
