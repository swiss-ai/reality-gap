"""Parler-TTS Mini Multilingual backend — Apache 2.0, prompt-controlled.

Voice is controlled by a TEXT DESCRIPTION ("A female speaker, slow tempo,
clear audio"), not by a reference audio clip. The reference_audio harness arg
is ignored. A default Polish-suitable description is built into this backend
and can be overridden via kwargs.

License: Apache 2.0.
"""

import logging
from typing import Optional

import torch

from ..base import TTSBackend, TTSOutput

logger = logging.getLogger(__name__)


DEFAULT_DESCRIPTION = (
    "A female speaker delivers slightly animated speech with clear audio "
    "quality and minimal background noise."
)


class ParlerTTSBackend(TTSBackend):
    """Parler-TTS backend (Apache 2.0, text-description-conditioned).

    Args:
        checkpoint: HF model ID (default: parler-tts-mini-multilingual-v1.1).
        device: cpu / cuda.
        default_description: Default voice-style prompt; can be overridden
            per-call via kwargs["description"].
    """

    def __init__(
        self,
        checkpoint: str = "parler-tts/parler-tts-mini-multilingual-v1.1",
        device: str = "cuda",
        default_description: str = DEFAULT_DESCRIPTION,
    ):
        self.checkpoint = checkpoint
        self.device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        self.default_description = default_description
        self._model = None
        self._description_tokenizer = None
        self._prompt_tokenizer = None
        self._sample_rate: Optional[int] = None

    def load_model(self, device: Optional[str] = None) -> None:
        if device:
            self.device = device

        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        self._model = (
            ParlerTTSForConditionalGeneration.from_pretrained(self.checkpoint)
            .to(self.device)
            .eval()
        )
        self._description_tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        # Parler-TTS uses separate tokenizers for description vs prompt; the
        # prompt tokenizer name is stored on the model's config.
        prompt_tok_name = getattr(
            self._model.config, "prompt_tokenizer_name", self.checkpoint
        )
        self._prompt_tokenizer = AutoTokenizer.from_pretrained(prompt_tok_name)

        self._sample_rate = int(self._model.config.sampling_rate)
        logger.info(
            "Parler-TTS loaded: %s on %s (sr=%d)",
            self.checkpoint, self.device, self._sample_rate,
        )

    def generate(
        self,
        text: str,
        reference_audio: Optional[torch.Tensor] = None,
        reference_audio_sr: Optional[int] = None,
        speaker_id: Optional[str] = None,
        render_audio: bool = False,
        description: Optional[str] = None,
        **kwargs,
    ) -> TTSOutput:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        desc = description or self.default_description

        desc_ids = self._description_tokenizer(desc, return_tensors="pt").input_ids.to(
            self.device
        )
        prompt_ids = self._prompt_tokenizer(text, return_tensors="pt").input_ids.to(
            self.device
        )

        with torch.no_grad():
            generation = self._model.generate(
                input_ids=desc_ids, prompt_input_ids=prompt_ids
            )

        audio = generation.cpu().float().squeeze()

        return TTSOutput(
            speech_tokens=torch.empty(0, dtype=torch.long),
            codebook_size=0,
            token_rate_hz=0.0,
            audio=audio if render_audio else None,
            audio_sample_rate=self._sample_rate if render_audio else None,
            metadata={
                "backend": "parler_tts",
                "checkpoint": self.checkpoint,
                "description": desc,
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
