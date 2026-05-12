"""XTTS v2 (Coqui) TTS backend — REFERENCE-ONLY.

XTTS does not emit discrete tokens directly. The benchmark harness compares
backends on audio-level metrics (WER, MOS, speaker similarity); token-level
output is produced downstream by re-encoding with the selected codec.

Requires its own venv: `pip install TTS` conflicts with several other tokenizer
deps. Add a Makefile target (`make xtts`) before using on the cluster.
"""

import logging
from typing import Optional

import torch

from ..base import TTSBackend, TTSOutput

logger = logging.getLogger(__name__)


class XTTSTTSBackend(TTSBackend):
    """XTTS v2 backend for text -> waveform generation with voice cloning.

    Args:
        checkpoint: HF or Coqui model ID.
        device: Target device.
        language: ISO code passed to XTTS ("pl" for Polish).
    """

    OUTPUT_SR = 24000

    def __init__(
        self,
        checkpoint: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: str = "cuda",
        language: str = "pl",
    ):
        self.checkpoint = checkpoint
        self.device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        self.language = language
        self._tts = None

    def load_model(self, device: Optional[str] = None) -> None:
        if device:
            self.device = device

        from TTS.api import TTS

        self._tts = TTS(self.checkpoint).to(self.device)
        logger.info("XTTS backend loaded on %s (lang=%s)", self.device, self.language)

    def generate(
        self,
        text: str,
        reference_audio: Optional[torch.Tensor] = None,
        reference_audio_sr: Optional[int] = None,
        speaker_id: Optional[str] = None,
        render_audio: bool = False,
        **kwargs,
    ) -> TTSOutput:
        if self._tts is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if reference_audio is None:
            raise ValueError(
                "XTTS requires reference_audio for voice cloning (3–10s mono clip)."
            )

        ref_path = self._write_reference_to_tmp(reference_audio, reference_audio_sr)

        wav = self._tts.tts(
            text=text,
            speaker_wav=ref_path,
            language=kwargs.get("language", self.language),
        )
        audio = torch.tensor(wav, dtype=torch.float32)
        if audio.ndim > 1:
            audio = audio.squeeze()

        return TTSOutput(
            speech_tokens=torch.empty(0, dtype=torch.long),
            codebook_size=0,
            token_rate_hz=0.0,
            audio=audio if render_audio else None,
            audio_sample_rate=self.OUTPUT_SR if render_audio else None,
            metadata={
                "backend": "xtts_v2",
                "language": self.language,
                "direct_tokens": False,
            },
        )

    def _write_reference_to_tmp(
        self, reference_audio: torch.Tensor, sr: Optional[int]
    ) -> str:
        """XTTS API takes a path; dump the tensor to a temp WAV."""
        import tempfile

        import torchaudio

        if reference_audio.ndim == 1:
            reference_audio = reference_audio.unsqueeze(0)
        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        torchaudio.save(f.name, reference_audio.cpu(), sr or 16000)
        return f.name

    @property
    def codebook_size(self) -> int:
        return 0  # not a tokenizing backend

    @property
    def token_rate_hz(self) -> float:
        return 0.0
