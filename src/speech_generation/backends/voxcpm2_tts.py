"""VoxCPM2 (OpenBMB) TTS backend — Apache 2.0, 30 languages incl. Polish, voice cloning.

Two integration paths exist for this model:

  1. Direct Python (`pip install voxcpm`) — what this backend uses.
     Suitable for the standard benchmark harness, one call per sentence.

  2. vLLM-omni serving (https://github.com/swiss-ai/model-launch). Better for
     the multi-GPU / batched-throughput story the supervisor flagged. Wire
     up later via a separate `voxcpm2_vllm_tts.py` backend that POSTs to
     a running vLLM-omni endpoint, once we have the server running.

Voice cloning takes a reference WAV file path. This backend writes the
reference tensor to a tmp file (same pattern as xtts/omnivoice).
"""

import logging
import tempfile
from typing import Optional

import torch

from ..base import TTSBackend, TTSOutput

logger = logging.getLogger(__name__)


class VoxCPM2TTSBackend(TTSBackend):
    """VoxCPM2 backend (Apache 2.0, multilingual + voice cloning).

    Args:
        checkpoint: HF model ID (default: openbmb/VoxCPM2).
        device: Target device.
        load_denoiser: Whether to load the optional denoiser submodel.
        cfg_value: Classifier-free guidance scale at inference.
        inference_timesteps: Number of diffusion timesteps.
    """

    def __init__(
        self,
        checkpoint: str = "openbmb/VoxCPM2",
        device: str = "cuda",
        load_denoiser: bool = False,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
    ):
        self.checkpoint = checkpoint
        self.device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        self.load_denoiser = load_denoiser
        self.cfg_value = cfg_value
        self.inference_timesteps = inference_timesteps
        self._model = None
        self._sample_rate: Optional[int] = None

    def load_model(self, device: Optional[str] = None) -> None:
        if device:
            self.device = device

        from voxcpm import VoxCPM

        self._model = VoxCPM.from_pretrained(
            self.checkpoint, load_denoiser=self.load_denoiser
        )
        try:
            self._model.to(self.device)
        except AttributeError:
            # Some VoxCPM versions don't expose .to() on the wrapper; sub-models
            # are typically already on CUDA by default. Ignored if unavailable.
            pass

        self._sample_rate = int(getattr(self._model.tts_model, "sample_rate", 16000))
        logger.info(
            "VoxCPM2 loaded: %s on %s (sr=%d)", self.checkpoint, self.device, self._sample_rate
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
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        gen_kwargs = {
            "text": text,
            "cfg_value": kwargs.get("cfg_value", self.cfg_value),
            "inference_timesteps": kwargs.get(
                "inference_timesteps", self.inference_timesteps
            ),
        }

        if reference_audio is not None:
            ref_path = self._write_reference_to_tmp(
                reference_audio, reference_audio_sr
            )
            gen_kwargs["reference_wav_path"] = ref_path

        audio_np = self._model.generate(**gen_kwargs)
        audio = torch.as_tensor(audio_np, dtype=torch.float32)
        if audio.ndim > 1:
            audio = audio.squeeze()

        return TTSOutput(
            speech_tokens=torch.empty(0, dtype=torch.long),
            codebook_size=0,
            token_rate_hz=0.0,
            audio=audio if render_audio else None,
            audio_sample_rate=self._sample_rate if render_audio else None,
            metadata={
                "backend": "voxcpm2",
                "checkpoint": self.checkpoint,
                "voice_cloning": reference_audio is not None,
                "cfg_value": gen_kwargs["cfg_value"],
                "inference_timesteps": gen_kwargs["inference_timesteps"],
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
