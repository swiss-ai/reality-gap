"""Piper TTS backend — MIT-licensed, Polish-native, lightweight.

Voice: `pl_PL-gosia-medium` (default). Piper is fixed-voice per model; no
voice cloning. The benchmark harness's reference_audio arg is ignored.

Install:
    pip install piper-tts
And download voice file (.onnx + .onnx.json):
    https://huggingface.co/rhasspy/piper-voices/tree/main/pl/pl_PL/gosia/medium

License: MIT (Piper code + Polish voice).
"""

import logging
from pathlib import Path
from typing import Optional

import torch

from ..base import TTSBackend, TTSOutput

logger = logging.getLogger(__name__)


class PiperTTSBackend(TTSBackend):
    """Piper backend (MIT, fixed-voice Polish-native).

    Args:
        voice_path: Path to the .onnx voice file (with sibling .onnx.json).
        device: cpu / cuda — Piper uses ONNXRuntime; CUDA needs onnxruntime-gpu.
        speaker_id: Multi-speaker voice only — ignored for single-speaker voices.
        length_scale: Speech rate (1.0 = normal, <1 faster, >1 slower).
        noise_scale: Audio variation. Default 0.667.
        noise_w: Stochastic duration prediction noise. Default 0.8.
    """

    def __init__(
        self,
        voice_path: str = "voices/pl_PL-gosia-medium.onnx",
        device: str = "cuda",
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
    ):
        self.voice_path = voice_path
        self.device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w = noise_w
        self._voice = None
        self._sample_rate: Optional[int] = None

    def load_model(self, device: Optional[str] = None) -> None:
        if device:
            self.device = device

        from piper import PiperVoice

        voice_path = Path(self.voice_path)
        if not voice_path.exists():
            raise FileNotFoundError(
                f"Voice file not found at {voice_path}. Download from "
                f"https://huggingface.co/rhasspy/piper-voices/tree/main/pl/pl_PL/gosia/medium"
            )

        use_cuda = self.device != "cpu"
        self._voice = PiperVoice.load(str(voice_path), use_cuda=use_cuda)
        self._sample_rate = int(self._voice.config.sample_rate)
        logger.info(
            "Piper loaded: %s on %s (sr=%d, use_cuda=%s)",
            voice_path.name, self.device, self._sample_rate, use_cuda,
        )

    def generate(
        self,
        text: str,
        reference_audio: Optional[torch.Tensor] = None,
        reference_audio_sr: Optional[int] = None,
        speaker_id: Optional[str] = None,
        render_audio: bool = False,
        **kwargs,
    ) -> TTSOutput:
        if self._voice is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        from piper import SynthesisConfig

        syn = SynthesisConfig(
            length_scale=kwargs.get("length_scale", self.length_scale),
            noise_scale=kwargs.get("noise_scale", self.noise_scale),
            noise_w_scale=kwargs.get("noise_w", self.noise_w),
        )

        # Piper streams int16 PCM bytes; concat and normalize to float32 in [-1, 1].
        pcm_bytes = b"".join(
            chunk.audio_int16_bytes for chunk in self._voice.synthesize(text, syn_config=syn)
        )
        import numpy as np

        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio = torch.from_numpy(audio_int16.astype("float32") / 32768.0)

        return TTSOutput(
            speech_tokens=torch.empty(0, dtype=torch.long),
            codebook_size=0,
            token_rate_hz=0.0,
            audio=audio if render_audio else None,
            audio_sample_rate=self._sample_rate if render_audio else None,
            metadata={
                "backend": "piper",
                "voice": Path(self.voice_path).name,
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
