from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_AUDIO_SRC = REPO_ROOT.parent / "src"

if str(BENCHMARK_AUDIO_SRC) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_AUDIO_SRC))

from audio_tokenizers.implementations.wavtokenizer import WavTokenizerWrapper


@dataclass
class AudioTokenizerConfig:
    """Configuration container for the audio tokenizer wrapper."""

    device: str = "cuda"
    special_token_order: Sequence[str] = field(
        default_factory=lambda: (
            "bos",
            "eos",
            "audio_start",
            "audio_end",
            "meta_start",
            "meta_end",
            "token_start",
        )
    )
    dataset_label: str = "librispeech"


class AudioTokenizer:
    """Adapter that combines the WavTokenizer wrapper with our special-token layout."""

    def __init__(self, config: AudioTokenizerConfig | None = None):
        self.config = config or AudioTokenizerConfig()
        self.backend = WavTokenizerWrapper(device=self.config.device)
        self.special_token_ids = {
            name: idx for idx, name in enumerate(self.config.special_token_order)
        }
        self.audio_token_offset = len(self.special_token_ids)
        self.audio_vocab_size = self.backend.codebook_size
        self.vocab_size = self.audio_token_offset + self.audio_vocab_size

    def tokenize_audio(self, audio_array: np.ndarray, sampling_rate: int) -> Dict:
        """Encode audio into a token sequence with BOS/AUDIO/META wrappers."""
        mono_audio = self._prepare_audio(audio_array)
        audio_tensor = torch.from_numpy(mono_audio).unsqueeze(0)
        tokens, encode_info = self.backend.encode(audio_tensor, sr=sampling_rate)

        if tokens.dim() == 2:
            tokens = tokens.squeeze(0)

        audio_codes = tokens.cpu().numpy().astype(np.int64).tolist()
        shifted_codes = [code + self.audio_token_offset for code in audio_codes]

        sequence = self._wrap_with_special_tokens(shifted_codes)

        token_info = {
            "num_audio_tokens": len(audio_codes),
            "num_total_tokens": len(sequence),
            "audio_token_offset": self.audio_token_offset,
            "special_tokens": self.special_token_ids,
            "wavtokenizer": {
                "tokens_per_second": self.backend.tokens_per_second,
                "codebook_size": self.backend.codebook_size,
                "sample_rate": self.backend.sample_rate,
            },
            **encode_info,
        }

        return {"tokens": sequence, "info": token_info}

    def detokenize(self, token_sequence: Sequence[int]) -> np.ndarray:
        """Convert a special-token sequence back into audio."""
        audio_tokens = self.extract_audio_tokens(token_sequence)
        token_tensor = torch.tensor(audio_tokens, dtype=torch.int64).unsqueeze(0)
        audio, _ = self.backend.decode(token_tensor)
        if audio.dim() > 1:
            audio = audio.squeeze(0)
        return audio.cpu().numpy()

    def extract_audio_tokens(self, sequence: Sequence[int]) -> List[int]:
        """Extract pure codec IDs (without offsets) from a sequence."""
        token_start = self.special_token_ids["token_start"]
        audio_end = self.special_token_ids["audio_end"]
        collecting = False
        audio_tokens: List[int] = []

        for token in sequence:
            if token == token_start:
                collecting = True
                continue
            if token == audio_end:
                break
            if collecting:
                if token < self.audio_token_offset:
                    # Unknown meta token, skip
                    continue
                audio_tokens.append(token - self.audio_token_offset)

        return audio_tokens

    def describe_special_tokens(self) -> List[Dict[str, int]]:
        """Return the special-token names and IDs."""
        return [
            {"name": name, "id": idx}
            for name, idx in self.special_token_ids.items()
        ]

    def _prepare_audio(self, audio_array: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio_array, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        return audio

    def _wrap_with_special_tokens(self, shifted_codes: List[int]) -> List[int]:
        bos = self.special_token_ids["bos"]
        audio_start = self.special_token_ids["audio_start"]
        meta_start = self.special_token_ids["meta_start"]
        meta_end = self.special_token_ids["meta_end"]
        token_start = self.special_token_ids["token_start"]
        audio_end = self.special_token_ids["audio_end"]
        eos = self.special_token_ids["eos"]

        sequence = [
            bos,
            audio_start,
            meta_start,
            meta_end,
            token_start,
            *shifted_codes,
            audio_end,
            eos,
        ]
        return sequence

