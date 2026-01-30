"""Audio-only tokenizer using WavTokenizer.

Follows the same pattern as vision_tokenization/vokenizers/emu/image_only.py:
- Loads omni_tokenizer to get special token IDs
- Gets audio token offset by querying: convert_tokens_to_ids("<|audio token 0|>")
"""

import sys
from pathlib import Path
from typing import Optional, Union

import torch
import numpy as np

# Add the src directory to path
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from audio_tokenizers.implementations.wavtokenizer import WavTokenizer40


class WavTokenizerAudioOnly:
    """Audio-only tokenizer using WavTokenizer-40.

    Wraps WavTokenizer40 and adds encapsulation with
    <|audio_start|> and <|audio_end|> tokens.

    Output: [audio_start] [audio_tokens...] [audio_end]

    The audio token offset is determined by querying the omni_tokenizer
    for the first audio token, just like vision does:
        first_audio_token = tokenizer.convert_tokens_to_ids("<|audio token 0|>")
    """

    def __init__(
        self,
        omni_tokenizer_path: str,
        device: str = "cuda",
        **kwargs,
    ):
        self.device = device
        self.omni_tokenizer_path = omni_tokenizer_path

        # Lazy load
        self._wavtokenizer = None
        self._omni_tokenizer = None
        self._bos_id = None
        self._eos_id = None
        self._audio_start_id = None
        self._audio_end_id = None
        self._audio_token_offset = None

    @property
    def wavtokenizer(self) -> WavTokenizer40:
        if self._wavtokenizer is None:
            self._wavtokenizer = WavTokenizer40(device=self.device)
        return self._wavtokenizer

    @property
    def omni_tokenizer(self):
        """Load and cache omni_tokenizer."""
        if self._omni_tokenizer is None:
            from transformers import AutoTokenizer
            self._omni_tokenizer = AutoTokenizer.from_pretrained(
                self.omni_tokenizer_path, trust_remote_code=True, use_fast=True
            )
        return self._omni_tokenizer

    def _cache_special_tokens(self):
        """Cache special token IDs from omni_tokenizer (same pattern as vision)."""
        # BOS and EOS tokens
        assert self.omni_tokenizer.bos_token is not None, "BOS token must be defined"
        assert self.omni_tokenizer.eos_token is not None, "EOS token must be defined"
        self._bos_id = self.omni_tokenizer.bos_token_id
        self._eos_id = self.omni_tokenizer.eos_token_id

        # Audio structure tokens
        self._audio_start_id = self.omni_tokenizer.convert_tokens_to_ids("<|audio_start|>")
        self._audio_end_id = self.omni_tokenizer.convert_tokens_to_ids("<|audio_end|>")

        # Audio token offset - query the first audio token just like vision does:
        # vision: first_vision_token = self.text_tokenizer.convert_tokens_to_ids("<|visual token 000000|>")
        # audio: first_audio_token = self.omni_tokenizer.convert_tokens_to_ids("<|audio token 0|>")
        first_audio_token = self.omni_tokenizer.convert_tokens_to_ids("<|audio token 0|>")

        # Check if audio tokens exist in the omni_tokenizer
        if first_audio_token == self.omni_tokenizer.unk_token_id:
            raise ValueError(
                f"Audio tokens not found in omni_tokenizer at {self.omni_tokenizer_path}. "
                "Make sure the omni_tokenizer has audio tokens added "
                "(e.g., <|audio token 0|> through <|audio token N|>)."
            )

        self._audio_token_offset = first_audio_token

    @property
    def bos_id(self) -> int:
        if self._bos_id is None:
            self._cache_special_tokens()
        return self._bos_id

    @property
    def eos_id(self) -> int:
        if self._eos_id is None:
            self._cache_special_tokens()
        return self._eos_id

    @property
    def audio_start_id(self) -> int:
        if self._audio_start_id is None:
            self._cache_special_tokens()
        return self._audio_start_id

    @property
    def audio_end_id(self) -> int:
        if self._audio_end_id is None:
            self._cache_special_tokens()
        return self._audio_end_id

    @property
    def audio_token_offset(self) -> int:
        """Offset for audio tokens, determined from omni_tokenizer."""
        if self._audio_token_offset is None:
            self._cache_special_tokens()
        return self._audio_token_offset

    def tokenize(
        self,
        audio: Union[np.ndarray, torch.Tensor, bytes],
        sample_rate: int,
        text: Optional[str] = None,
    ) -> torch.Tensor:
        """Tokenize audio with encapsulation.

        Returns:
            Token tensor: [BOS, audio_start, audio_tokens..., audio_end, EOS]
        """
        # Convert to tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()

        # Ensure batch dimension for single samples
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # (T,) -> (1, T)

        # Encode (handles resampling internally)
        tokens, _ = self.wavtokenizer.encode(audio, sr=sample_rate)
        tokens = tokens.flatten()
        num_audio_tokens = tokens.numel()

        # Pre-allocate output tensor for efficiency (same pattern as vision)
        # Structure: BOS + audio_start + audio_tokens + audio_end + EOS
        total_size = 1 + 1 + num_audio_tokens + 1 + 1  # BOS + audio_start + tokens + audio_end + EOS
        device = tokens.device
        output = torch.empty(total_size, dtype=torch.long, device=device)

        # Fill tokens using slicing (no Python list operations)
        output[0] = self.bos_id
        output[1] = self.audio_start_id
        # Apply offset and copy audio tokens
        output[2:2 + num_audio_tokens] = tokens + self.audio_token_offset
        output[-2] = self.audio_end_id
        output[-1] = self.eos_id

        return output

    def tokenize_batch(
        self,
        audios: list,
        sample_rate: int,
    ) -> list:
        """Tokenize a batch of audio samples efficiently on GPU.

        All audio samples should have the same length for efficient batching.
        Use bucket filtering to ensure same-length samples.

        Args:
            audios: List of audio arrays (np.ndarray or torch.Tensor), all same length
            sample_rate: Sample rate of all audio samples

        Returns:
            List of token tensors, each: [BOS, audio_start, audio_tokens..., audio_end, EOS]
        """
        if not audios:
            return []

        batch_size = len(audios)

        # Stack numpy arrays first (fast CPU operation), then single transfer to GPU
        first = audios[0]
        if isinstance(first, np.ndarray):
            # Stack all numpy arrays at once, then move to GPU in single transfer
            stacked = np.stack(audios, axis=0)  # (B, T) numpy
            batch_audio = torch.from_numpy(stacked).float().to(self.device, non_blocking=True)
        else:
            # Already tensors - stack and move
            batch_audio = torch.stack([a if a.dim() == 1 else a.squeeze(0) for a in audios], dim=0)
            if batch_audio.device != self.device:
                batch_audio = batch_audio.to(self.device, non_blocking=True)

        # Encode entire batch at once (handles resampling internally)
        batch_tokens, _ = self.wavtokenizer.encode(batch_audio, sr=sample_rate)
        # batch_tokens shape: (B, N) where N is number of audio tokens per sample

        # Get dimensions
        num_audio_tokens = batch_tokens.shape[1]
        device = batch_tokens.device

        # Pre-allocate output for entire batch
        # Structure per sample: BOS + audio_start + audio_tokens + audio_end + EOS
        total_size = 1 + 1 + num_audio_tokens + 1 + 1
        batch_output = torch.empty(batch_size, total_size, dtype=torch.long, device=device)

        # Fill structure tokens (same for all samples)
        batch_output[:, 0] = self.bos_id
        batch_output[:, 1] = self.audio_start_id
        batch_output[:, -2] = self.audio_end_id
        batch_output[:, -1] = self.eos_id

        # Apply offset and copy audio tokens for all samples at once
        batch_output[:, 2:2 + num_audio_tokens] = batch_tokens + self.audio_token_offset

        # Return as list of tensors (one per sample)
        return [batch_output[i] for i in range(batch_size)]
