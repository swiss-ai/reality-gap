"""F5-TTS backend — MIT-licensed, diffusion-based, voice cloning.

Default checkpoint: `Sticzu/marek-f5tts-polish` (HF Hub, MIT licensed) — a
Polish fine-tune of the original F5-TTS, single male voice "Marek". Pass
`--checkpoint` to override with another HF repo ID, a local checkpoint path,
or a built-in F5-TTS model name (e.g. "F5TTS_v1_Base" for English/Chinese).

Always requires Polish reference audio + transcript for voice cloning.

License: MIT (code + Sticzu Polish fine-tune weights). Verify license if
swapping in a different community checkpoint.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

import torch

from ..base import TTSBackend, TTSOutput

logger = logging.getLogger(__name__)

# Built-in F5-TTS model names (resolved internally by the library — not HF repos).
_BUILTIN_MODELS = {"F5TTS", "F5TTS_v1_Base", "F5-TTS", "E2TTS_Base", "E2-TTS"}

# Default Polish fine-tune.
DEFAULT_HF_REPO = "Sticzu/marek-f5tts-polish"
DEFAULT_CKPT_FILE = "model_205500.pt"
DEFAULT_VOCAB_FILE = "vocab.txt"


class F5TTSBackend(TTSBackend):
    """F5-TTS backend (MIT, voice cloning, Polish fine-tune by default).

    Args:
        checkpoint: One of three forms:
            * HF repo id (e.g. "Sticzu/marek-f5tts-polish") — downloads
              ``ckpt_file`` and ``vocab_file`` from the Hub.
            * Local path to a .pt / .safetensors file — used directly.
            * Built-in name (e.g. "F5TTS_v1_Base") — resolved by F5-TTS
              library, English/Chinese only.
        ckpt_file: When ``checkpoint`` is a HF repo, the filename inside it.
        vocab_file: When ``checkpoint`` is a HF repo, the vocab filename.
        model_type: F5-TTS architecture type. Default "F5TTS".
        device: cpu / cuda.
        nfe_step / cfg_strength / sway_sampling_coef: diffusion inference knobs.
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_HF_REPO,
        ckpt_file: str = DEFAULT_CKPT_FILE,
        vocab_file: str = DEFAULT_VOCAB_FILE,
        model_type: str = "F5TTS",
        device: str = "cuda",
        nfe_step: int = 32,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,
    ):
        self.checkpoint = checkpoint
        self.ckpt_file = ckpt_file
        self.vocab_file = vocab_file
        self.model_type = model_type
        self.device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        self.nfe_step = nfe_step
        self.cfg_strength = cfg_strength
        self.sway_sampling_coef = sway_sampling_coef
        self._f5 = None
        self._sample_rate: Optional[int] = None

    def load_model(self, device: Optional[str] = None) -> None:
        if device:
            self.device = device

        from f5_tts.api import F5TTS

        ckpt_path, vocab_path = self._resolve_checkpoint()

        if ckpt_path is None:
            # Built-in model: F5-TTS library resolves the checkpoint itself.
            self._f5 = F5TTS(model=self.checkpoint, device=self.device)
        else:
            self._f5 = F5TTS(
                model_type=self.model_type,
                ckpt_file=str(ckpt_path),
                vocab_file=str(vocab_path) if vocab_path else "",
                device=self.device,
            )
        self._sample_rate = int(getattr(self._f5, "target_sample_rate", 24000))
        logger.info(
            "F5-TTS loaded: %s on %s (sr=%d)",
            self.checkpoint, self.device, self._sample_rate,
        )

    def _resolve_checkpoint(self) -> tuple[Optional[Path], Optional[Path]]:
        """Resolve ``self.checkpoint`` to (ckpt_path, vocab_path).

        Returns (None, None) for built-in model names (let F5-TTS handle it).
        """
        if self.checkpoint in _BUILTIN_MODELS:
            return None, None

        ckpt_as_path = Path(self.checkpoint)
        if ckpt_as_path.exists() and ckpt_as_path.is_file():
            # Local .pt path; assume vocab.txt sibling unless overridden.
            sibling_vocab = ckpt_as_path.with_name(self.vocab_file)
            return ckpt_as_path, sibling_vocab if sibling_vocab.exists() else None

        # Treat as HF Hub repo id (org/repo).
        from huggingface_hub import hf_hub_download

        logger.info("Downloading F5-TTS checkpoint from HF Hub: %s", self.checkpoint)
        ckpt_path = Path(
            hf_hub_download(repo_id=self.checkpoint, filename=self.ckpt_file)
        )
        vocab_path = Path(
            hf_hub_download(repo_id=self.checkpoint, filename=self.vocab_file)
        )
        return ckpt_path, vocab_path

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
        if self._f5 is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if reference_audio is None:
            raise ValueError("F5-TTS requires reference_audio (3–10s clip).")
        if not ref_text:
            raise ValueError(
                "F5-TTS requires ref_text (transcript of the reference clip)."
            )

        ref_path = self._write_reference_to_tmp(reference_audio, reference_audio_sr)

        wav, sr, _ = self._f5.infer(
            ref_file=ref_path,
            ref_text=ref_text,
            gen_text=text,
            nfe_step=kwargs.get("nfe_step", self.nfe_step),
            cfg_strength=kwargs.get("cfg_strength", self.cfg_strength),
            sway_sampling_coef=kwargs.get(
                "sway_sampling_coef", self.sway_sampling_coef
            ),
            remove_silence=False,
        )
        audio = torch.as_tensor(wav, dtype=torch.float32)
        if audio.ndim > 1:
            audio = audio.squeeze()

        return TTSOutput(
            speech_tokens=torch.empty(0, dtype=torch.long),
            codebook_size=0,
            token_rate_hz=0.0,
            audio=audio if render_audio else None,
            audio_sample_rate=int(sr) if render_audio else None,
            metadata={
                "backend": "f5_tts",
                "checkpoint": self.checkpoint,
                "voice_cloning": True,
                "direct_tokens": False,
                "nfe_step": kwargs.get("nfe_step", self.nfe_step),
                "cfg_strength": kwargs.get("cfg_strength", self.cfg_strength),
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
