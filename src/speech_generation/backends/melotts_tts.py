"""MeloTTS (MyShell) TTS backend — MIT, multilingual, built-in voices.

Like Qwen2.5-Omni: NO zero-shot cloning. Uses built-in language-specific
speakers. For zh the only built-in voice is "ZH" (mixed-style).
"""

import logging
import tempfile
from typing import Optional

import torch

from ..base import TTSBackend, TTSOutput

logger = logging.getLogger(__name__)


_LANG_MAP = {
    "zh": "ZH",
    "en": "EN",
    "es": "ES",
    "fr": "FR",
    "ja": "JP",
    "ko": "KR",
}


class MeloTTSBackend(TTSBackend):
    """MeloTTS backend (MIT, MyShell, built-in voices only)."""

    def __init__(
        self,
        language: str = "ZH",
        device: str = "cuda",
        speed: float = 1.0,
    ):
        self.language = language
        self.device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        self.speed = speed
        self._model = None
        self._speaker_id = None
        self._sample_rate: int = 44100  # MeloTTS native rate

    def load_model(self, device: Optional[str] = None) -> None:
        if device:
            self.device = device

        from melo.api import TTS

        # Map our pl/zh/en codes to MeloTTS's uppercase format.
        lang = _LANG_MAP.get(self.language.lower(), self.language.upper())
        self._model = TTS(language=lang, device=self.device)
        # Default speaker per language. ZH has one built-in speaker.
        self._speaker_id = next(iter(self._model.hps.data.spk2id.values()))
        logger.info("MeloTTS loaded: lang=%s on %s", lang, self.device)

    def generate(
        self,
        text: str,
        reference_audio: Optional[torch.Tensor] = None,
        reference_audio_sr: Optional[int] = None,
        speaker_id: Optional[str] = None,
        render_audio: bool = False,
        language: Optional[str] = None,
        **kwargs,
    ) -> TTSOutput:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # MeloTTS.tts_to_file writes wav; we go through tmp file rather than
        # piping the array because the API doesn't return audio directly.
        out_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        self._model.tts_to_file(text, self._speaker_id, out_tmp, speed=self.speed)

        import soundfile as sf
        wav, sr = sf.read(out_tmp)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        audio = torch.as_tensor(wav, dtype=torch.float32)

        return TTSOutput(
            speech_tokens=torch.empty(0, dtype=torch.long),
            codebook_size=0,
            token_rate_hz=0.0,
            audio=audio if render_audio else None,
            audio_sample_rate=sr if render_audio else None,
            metadata={
                "backend": "melotts",
                "language": self.language,
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
