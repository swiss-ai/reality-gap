"""Qwen2.5-Omni-7B TTS backend — Apache 2.0, multilingual, built-in voices.

Unlike VoxCPM2/CosyVoice2/F5 which support zero-shot voice cloning from a
reference WAV, Qwen2.5-Omni uses **built-in named voices** picked via the
`speaker=` argument. The reference_audio passed by the benchmark harness is
IGNORED for this backend.

Built-in voices (as of model release):
    "Chelsie" — female (default)
    "Ethan"   — male
"""

import logging
from typing import Optional

import torch

from ..base import TTSBackend, TTSOutput

logger = logging.getLogger(__name__)


# Map per-language to a built-in voice. Chelsie is the official zh voice.
_DEFAULT_VOICE_PER_LANG = {
    "zh": "Chelsie",
    "en": "Chelsie",
}


class QwenOmniTTSBackend(TTSBackend):
    """Qwen2.5-Omni-7B backend (Apache 2.0, built-in voices only).

    Args:
        checkpoint: HF model ID (default: Qwen/Qwen2.5-Omni-7B).
        device: Target device.
        speaker: Built-in voice name. Default picks per-language.
        return_audio: must be True for TTS.
    """

    def __init__(
        self,
        checkpoint: str = "Qwen/Qwen2.5-Omni-7B",
        device: str = "cuda",
        speaker: Optional[str] = None,
        torch_dtype: str = "auto",
    ):
        self.checkpoint = checkpoint
        self.device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        self.speaker = speaker
        self.torch_dtype = torch_dtype
        self._model = None
        self._processor = None
        # Qwen2.5-Omni audio decoder output rate is 24 kHz.
        self._sample_rate: int = 24000

    def load_model(self, device: Optional[str] = None) -> None:
        if device:
            self.device = device

        # NGC 24.11 ships torch 2.6.0a0+df5bbc09d1.nv24.11 — already patched
        # against CVE-2025-32434, but transformers' check_torch_load_is_safe()
        # reads torch version via importlib.metadata.version("torch") (NOT
        # torch.__version__), parses the alpha as <2.6 per PEP 440, and blocks
        # Qwen2.5-Omni's load_speakers(). Monkey-patch the check itself before
        # importing/calling from_pretrained — NVIDIA's backport already
        # contains the fix, so the check is purely cosmetic for our setup.
        import transformers.utils.import_utils as _imp_utils
        _imp_utils.check_torch_load_is_safe = lambda: None

        from transformers import (
            Qwen2_5OmniForConditionalGeneration,
            Qwen2_5OmniProcessor,
        )

        self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.checkpoint,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
        )
        self._processor = Qwen2_5OmniProcessor.from_pretrained(self.checkpoint)
        logger.info(
            "Qwen2.5-Omni loaded: %s on %s (sr=%d, speaker=%s)",
            self.checkpoint, self.device, self._sample_rate,
            self.speaker or "lang-default",
        )

    def generate(
        self,
        text: str,
        reference_audio: Optional[torch.Tensor] = None,
        reference_audio_sr: Optional[int] = None,
        speaker_id: Optional[str] = None,
        render_audio: bool = False,
        language: str = "zh",
        **kwargs,
    ) -> TTSOutput:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Pick voice. Priority: explicit speaker_id arg > __init__ speaker >
        # per-language default.
        voice = (
            speaker_id
            or self.speaker
            or _DEFAULT_VOICE_PER_LANG.get(language, "Chelsie")
        )

        # Conversation template: system prompt + user-supplied text.
        # The system prompt fixes the assistant persona so the model just
        # vocalizes the user text rather than answering it as dialog.
        system_prompt = (
            "You are a text-to-speech assistant. Read the user's text aloud "
            "verbatim in a natural voice. Do not add commentary."
        )
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": text}]},
        ]
        prompt = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False,
        )
        inputs = self._processor(
            text=prompt, return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            text_ids, audio = self._model.generate(
                **inputs,
                speaker=voice,
                return_audio=True,
            )

        audio = audio.reshape(-1).detach().to(torch.float32).cpu()

        return TTSOutput(
            speech_tokens=torch.empty(0, dtype=torch.long),
            codebook_size=0,
            token_rate_hz=0.0,
            audio=audio if render_audio else None,
            audio_sample_rate=self._sample_rate if render_audio else None,
            metadata={
                "backend": "qwen_omni",
                "checkpoint": self.checkpoint,
                "voice": voice,
                "voice_cloning": False,  # Qwen uses built-in voices only.
                "language": language,
                "direct_tokens": False,
            },
        )

    @property
    def codebook_size(self) -> int:
        return 0

    @property
    def token_rate_hz(self) -> float:
        return 0.0
