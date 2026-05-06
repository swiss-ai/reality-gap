"""MMS-TTS (Meta) backend: Polish-trained monolingual TTS.

Single fixed speaker per language. No voice cloning — reference_audio is ignored.
Useful as a baseline to confirm a TTS that was actually trained on Polish.

Runs in the cosyvoice2 venv (transformers is already installed there).
"""

import logging
from typing import Optional

import torch

from ..base import TTSBackend, TTSOutput

logger = logging.getLogger(__name__)


class MMSTTSBackend(TTSBackend):
    """MMS-TTS backend (Meta facebook/mms-tts-<lang>).

    Args:
        language: ISO-639-3 code embedded in the model ID (e.g. 'pol' for Polish).
        device: Target device.
    """

    OUTPUT_SR = 16000  # MMS-TTS native rate

    def __init__(self, language: str = "pol", device: str = "cuda"):
        self.language = language
        self.device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        self.checkpoint = f"facebook/mms-tts-{language}"
        self._model = None
        self._tokenizer = None

    def load_model(self, device: Optional[str] = None) -> None:
        if device:
            self.device = device

        from transformers import AutoTokenizer, VitsModel

        self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self._model = VitsModel.from_pretrained(self.checkpoint).to(self.device).eval()
        logger.info("MMS-TTS loaded: %s on %s", self.checkpoint, self.device)

    def generate(
        self,
        text: str,
        reference_audio: Optional[torch.Tensor] = None,
        reference_audio_sr: Optional[int] = None,
        speaker_id: Optional[str] = None,
        render_audio: bool = False,
        **kwargs,
    ) -> TTSOutput:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        inputs = self._tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            audio = self._model(**inputs).waveform.squeeze(0).float().cpu()

        return TTSOutput(
            speech_tokens=torch.empty(0, dtype=torch.long),
            codebook_size=0,
            token_rate_hz=0.0,
            audio=audio if render_audio else None,
            audio_sample_rate=self.OUTPUT_SR if render_audio else None,
            metadata={
                "backend": "mms_tts",
                "language": self.language,
                "checkpoint": self.checkpoint,
                "voice_cloning": False,
                "direct_tokens": False,
            },
        )

    @property
    def codebook_size(self) -> int:
        return 0

    @property
    def token_rate_hz(self) -> float:
        return 0.0
