"""IndexTTS (BiliBili) TTS backend — Apache 2.0, zh-native, voice cloning."""

import logging
import tempfile
from typing import Optional

import torch

from ..base import TTSBackend, TTSOutput

logger = logging.getLogger(__name__)


class IndexTTSBackend(TTSBackend):
    """IndexTTS backend (Apache 2.0, Chinese-native, zero-shot voice cloning).

    IndexTTS expects a reference WAV file path (not a tensor). We write tensor
    to a tmp file (same pattern as VoxCPM2).
    """

    def __init__(
        self,
        checkpoint: str = "IndexTeam/IndexTTS-1.5",
        device: str = "cuda",
    ):
        self.checkpoint = checkpoint
        self.device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        self._model = None
        self._sample_rate: int = 24000  # IndexTTS native rate
        self._ref_cache: dict[int, str] = {}

    def load_model(self, device: Optional[str] = None) -> None:
        if device:
            self.device = device

        # IndexTTS downloads model + config from HF on first call.
        from indextts.infer import IndexTTS

        # IndexTTS API needs both model_dir and cfg_path. Pull from HF hub
        # snapshot to a local dir, then point both at it.
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(self.checkpoint)
        cfg_path = f"{model_dir}/config.yaml"

        self._model = IndexTTS(
            model_dir=model_dir,
            cfg_path=cfg_path,
        )
        logger.info("IndexTTS loaded: %s on %s", self.checkpoint, self.device)

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
        if reference_audio is None:
            raise ValueError("IndexTTS requires reference_audio for voice cloning.")

        ref_path = self._write_reference_to_tmp(reference_audio, reference_audio_sr)
        out_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

        # IndexTTS.infer writes to a file path; load it back.
        self._model.infer(ref_path, text, out_tmp)

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
                "backend": "indextts",
                "checkpoint": self.checkpoint,
                "voice_cloning": True,
                "direct_tokens": False,
            },
        )

    def _write_reference_to_tmp(self, ref: torch.Tensor, sr: Optional[int]) -> str:
        key = id(ref)
        if key in self._ref_cache:
            return self._ref_cache[key]
        import torchaudio
        if ref.ndim == 1:
            ref = ref.unsqueeze(0)
        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        torchaudio.save(f.name, ref.cpu(), sr or 16000)
        self._ref_cache[key] = f.name
        return f.name

    @property
    def codebook_size(self) -> int:
        return 0

    @property
    def token_rate_hz(self) -> float:
        return 0.0
