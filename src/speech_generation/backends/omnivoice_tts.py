"""OmniVoice (k2-fsa) TTS backend: zero-shot multilingual TTS with voice cloning.

License: Apache 2.0. Supports 600+ languages incl. Polish.

Voice cloning requires BOTH a reference audio clip AND its transcript.
The transcript is passed through `ref_text` (kwargs in TTSBackend.generate),
since the base interface doesn't have a dedicated field for it.

Requires its own venv: `pip install omnivoice` pulls torch 2.8.0+cu128
which conflicts with cosyvoice2's torch.
"""

import logging
import tempfile
from typing import Optional

import torch

from ..base import TTSBackend, TTSOutput

logger = logging.getLogger(__name__)


class OmniVoiceTTSBackend(TTSBackend):
    """OmniVoice backend (Apache 2.0, voice cloning, ref_text required).

    Args:
        checkpoint: HF model ID (default: k2-fsa/OmniVoice).
        device: Target device.
        dtype: Inference dtype (float16 recommended).
    """

    OUTPUT_SR = 24000

    def __init__(
        self,
        checkpoint: str = "k2-fsa/OmniVoice",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.checkpoint = checkpoint
        self.device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        self.dtype = dtype
        self._model = None

    def load_model(self, device: Optional[str] = None) -> None:
        if device:
            self.device = device

        from omnivoice import OmniVoice

        self._model = OmniVoice.from_pretrained(
            self.checkpoint, device_map=self.device, dtype=self.dtype
        )
        logger.info("OmniVoice loaded: %s on %s", self.checkpoint, self.device)

    def generate(
        self,
        text: str,
        reference_audio: Optional[torch.Tensor] = None,
        reference_audio_sr: Optional[int] = None,
        speaker_id: Optional[str] = None,
        render_audio: bool = False,
        ref_text: Optional[str] = None,
        **kwargs,
    ) -> TTSOutput:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if reference_audio is None:
            raise ValueError("OmniVoice requires reference_audio (3–10s clip).")
        if not ref_text:
            raise ValueError(
                "OmniVoice requires ref_text (transcript of the reference clip). "
                "Pass it via kwargs from the harness."
            )

        ref_path = self._write_reference_to_tmp(reference_audio, reference_audio_sr)

        audio_np = self._model.generate(text=text, ref_audio=ref_path, ref_text=ref_text)
        # OmniVoice returns a numpy array shaped (1, samples) at 24 kHz
        audio = torch.tensor(audio_np[0], dtype=torch.float32)
        if audio.ndim > 1:
            audio = audio.squeeze()

        return TTSOutput(
            speech_tokens=torch.empty(0, dtype=torch.long),
            codebook_size=0,
            token_rate_hz=0.0,
            audio=audio if render_audio else None,
            audio_sample_rate=self.OUTPUT_SR if render_audio else None,
            metadata={
                "backend": "omnivoice",
                "checkpoint": self.checkpoint,
                "voice_cloning": True,
                "direct_tokens": False,
            },
        )

    def _write_reference_to_tmp(
        self, reference_audio: torch.Tensor, sr: Optional[int]
    ) -> str:
        import torchaudio

        if reference_audio.ndim == 1:
            reference_audio = reference_audio.unsqueeze(0)
        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        torchaudio.save(f.name, reference_audio.cpu(), sr or 16000)
        return f.name

    @property
    def codebook_size(self) -> int:
        return 0

    @property
    def token_rate_hz(self) -> float:
        return 0.0
