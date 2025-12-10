#!/usr/bin/env python3
"""
Audio-only tokenizer with core functionality.
Supports WavTokenizer and other audio tokenizers via Omni-Tokenizer system.
"""

import torch
import json
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from transformers import AutoTokenizer
import sys
import os

# Import BaseTokenizer from vision_tokenization
# Add parent directory of vision_tokenization to sys.path so it can be imported as a package
_current_file = Path(__file__).resolve()
_repo_root = _current_file.parents[3]
_vision_tokenization_dir = _repo_root / "src" / "repos" / "benchmark-image-tokenzier"
_vision_tokenization_path = _vision_tokenization_dir / "vision_tokenization"
_vision_tokenization_parent = _vision_tokenization_path.parent

# Add parent directory to sys.path so vision_tokenization can be imported as a package
if str(_vision_tokenization_parent) not in sys.path:
    sys.path.insert(0, str(_vision_tokenization_parent))

# Now import with full module path
from vision_tokenization.vokenizers.base import BaseTokenizer

# Import audio tokenizer utilities
# We need to import from audio_tokenization.utils.omni_tokenizer, not vision_tokenization
_audio_tokenization_dir = _current_file.parents[2]
# Make sure audio_tokenization is in sys.path BEFORE vision_tokenization parent
# so that utils.omni_tokenizer resolves to audio_tokenization version
if str(_audio_tokenization_dir) not in sys.path:
    # Insert at position 0 to ensure it's found before vision_tokenization
    sys.path.insert(0, str(_audio_tokenization_dir))

# Import directly from core.py to avoid __init__.py import issues
import importlib.util
_omni_tokenizer_core_path = _audio_tokenization_dir / "utils" / "omni_tokenizer" / "core.py"
spec = importlib.util.spec_from_file_location("audio_omni_tokenizer_core", _omni_tokenizer_core_path)
audio_omni_tokenizer_core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audio_omni_tokenizer_core)
load_audio_token_mapping = audio_omni_tokenizer_core.load_audio_token_mapping
get_audio_token_id = audio_omni_tokenizer_core.get_audio_token_id

# Import WavTokenizerWrapper
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from audio_tokenizers.implementations.wavtokenizer import WavTokenizerWrapper


class AudioOnlyTokenizer(BaseTokenizer):
    """
    Audio tokenizer for audio-only sequences.
    Provides direct audio tokenization with audio special tokens.
    Supports WavTokenizer and other audio tokenizers via Omni-Tokenizer system.
    """

    def __init__(
        self,
        text_tokenizer_path: str,
        device: str = "cuda"
    ):
        """
        Initialize with text tokenizer that has audio tokens and audio tokenizer.

        Args:
            text_tokenizer_path: Path to text tokenizer with audio tokens (Omni-Tokenizer)
            device: Device for audio tokenizer (default: "cuda")
        """
        # Store device
        self.device = device

        # Load tokenizer with trust_remote_code for custom tokenizer class
        # Use fast tokenizer for better performance
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            text_tokenizer_path,
            trust_remote_code=True,
            use_fast=True
        )

        # Load audio tokenizer config from tokenizer_config.json
        config_path = Path(text_tokenizer_path) / "tokenizer_config.json"
        with open(config_path, 'r') as f:
            tokenizer_config = json.load(f)

        if 'audio_tokenizer' not in tokenizer_config:
            raise ValueError(
                f"No audio_tokenizer config found in {config_path}. "
                f"Make sure the omni-tokenizer was created with audio tokenizer info."
            )

        audio_config = tokenizer_config['audio_tokenizer']
        audio_tokenizer_type = audio_config['type']
        audio_tokenizer_path = audio_config.get('path', None)

        print(f"Loading audio tokenizer: {audio_tokenizer_type}")

        # Dynamically load the correct audio tokenizer class
        if audio_tokenizer_type == 'WavTokenizer':
            # WavTokenizer auto-loads from HuggingFace, path is optional
            self.audio_tokenizer = WavTokenizerWrapper(device=self.device)
        else:
            raise ValueError(
                f"Unsupported audio tokenizer type: {audio_tokenizer_type}. "
                f"Supported types: WavTokenizer"
            )

        # Load audio token mapping
        self.audio_token_mapping = load_audio_token_mapping(text_tokenizer_path)
        self.audio_vocab_size = self.audio_token_mapping['audio_vocab_size']
        self.audio_token_ids = self.audio_token_mapping['audio_token_ids']

        # Cache frequently used token IDs
        self._cache_special_tokens()

    def _cache_special_tokens(self):
        """Cache special token IDs to avoid repeated lookups."""
        # Structure tokens
        assert self.text_tokenizer.bos_token is not None, "BOS token must be defined"
        assert self.text_tokenizer.eos_token is not None, "EOS token must be defined"

        self.bos_id = self.text_tokenizer.bos_token_id
        self.eos_id = self.text_tokenizer.eos_token_id

        # Audio special tokens
        self.audio_start_id = self.text_tokenizer.convert_tokens_to_ids("<|audio_start|>")
        self.audio_token_start_id = self.text_tokenizer.convert_tokens_to_ids("<|audio_token_start|>")
        self.audio_end_id = self.text_tokenizer.convert_tokens_to_ids("<|audio_end|>")

        # Verify tokens exist
        unk_id = self.text_tokenizer.unk_token_id
        if unk_id is not None:
            assert self.audio_start_id != unk_id, "<|audio_start|> not found in vocabulary"
            assert self.audio_token_start_id != unk_id, "<|audio_token_start|> not found in vocabulary"
            assert self.audio_end_id != unk_id, "<|audio_end|> not found in vocabulary"

    def _map_audio_indices_to_token_ids(self, audio_indices: torch.Tensor) -> torch.Tensor:
        """
        Map audio tokenizer indices to Omni-Tokenizer token IDs.

        Args:
            audio_indices: Tensor of audio indices from audio tokenizer [T]

        Returns:
            Tensor of token IDs in Omni-Tokenizer vocabulary [T]
        """
        # Convert to numpy for efficient indexing
        if isinstance(audio_indices, torch.Tensor):
            audio_indices_np = audio_indices.cpu().numpy()
        else:
            audio_indices_np = np.asarray(audio_indices)

        # Map each index to token ID using the mapping
        token_ids = []
        for idx in audio_indices_np.flatten():
            token_id = get_audio_token_id(int(idx), mapping=self.audio_token_mapping)
            token_ids.append(token_id)

        return torch.tensor(token_ids, dtype=torch.long)

    def encapsulate_audio(
        self,
        audio_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Directly tokenize audio-only data without intermediate text conversion.

        Args:
            audio_indices: Tensor of audio indices from audio tokenizer [T]

        Returns:
            Token IDs ready for model input
        """
        num_tokens = audio_indices.numel()

        # Map audio indices to Omni-Tokenizer token IDs
        audio_token_ids = self._map_audio_indices_to_token_ids(audio_indices)

        # Pre-allocate output tensor for efficiency
        # Structure: BOS + audio_start + audio_token_start + audio_tokens + audio_end + EOS
        total_size = (
            1 +  # BOS
            1 +  # audio_start
            1 +  # audio_token_start
            num_tokens +  # audio tokens
            1 +  # audio_end
            1    # EOS
        )

        # Pre-allocate the entire output tensor
        output = torch.empty(total_size, dtype=torch.long)

        # Fill in the tokens
        output[0] = self.bos_id
        output[1] = self.audio_start_id
        output[2] = self.audio_token_start_id
        output[3:3+num_tokens] = audio_token_ids
        output[3+num_tokens] = self.audio_end_id
        output[3+num_tokens+1] = self.eos_id

        return output

    @torch.inference_mode()
    def tokenize_audio(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sampling_rate: int = 16000
    ) -> torch.Tensor:
        """
        Complete pipeline: Audio array → audio indices → encapsulated tokens.

        Args:
            audio: Audio array (numpy array or torch tensor, float32)
            sampling_rate: Sample rate of input audio (default: 16000)

        Returns:
            Token sequence with audio structure tokens (BOS, audio_start, tokens, audio_end, EOS)
        """
        assert self.audio_tokenizer is not None, "Audio tokenizer required for processing audio"

        # Step 1: Ensure audio is in correct format for WavTokenizerWrapper
        # WavTokenizerWrapper.encode() expects numpy array or torch tensor
        # Convert to numpy array and ensure it's 1D (flatten if needed)
        if isinstance(audio, torch.Tensor):
            audio_np = audio.cpu().numpy().flatten().astype(np.float32)
        else:
            audio_np = np.asarray(audio, dtype=np.float32).flatten()
        
        # Step 2: Encode audio to discrete tokens
        # WavTokenizerWrapper.encode() handles resampling and returns (tokens, info)
        # Note: There's a bug in WavTokenizerWrapper.encode() where it checks audio.shape[1]
        # even when audio is 1D. We work around this by ensuring the audio is passed correctly.
        # The wrapper should handle 1D arrays, but if sr == sample_rate, it doesn't reshape properly.
        # We'll use encode_audio() directly after preprocessing to avoid the bug.
        
        # Convert to tensor and ensure batch dimension
        audio_tensor = torch.from_numpy(audio_np).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension: (T) -> (1, T)
        
        # Handle resampling if needed
        if sampling_rate != self.audio_tokenizer.sample_rate:
            import torchaudio
            resampler = torchaudio.transforms.Resample(sampling_rate, self.audio_tokenizer.sample_rate)
            audio_tensor = resampler(audio_tensor)
        
        # Encode using encode_audio() directly
        tokens = self.audio_tokenizer.encode_audio(audio_tensor)
        
        # Create info dict
        info = {
            "num_tokens": tokens.numel(),
            "token_shape": list(tokens.shape),
            "sample_rate": self.audio_tokenizer.sample_rate,
            "tokens_per_second": self.audio_tokenizer.tokens_per_second
        }

        # Step 2: Flatten tokens if needed (WavTokenizer returns [1, T] or [T])
        if tokens.dim() > 1:
            tokens = tokens.flatten()

        # Step 3: Encapsulate with audio structure tokens
        return self.encapsulate_audio(tokens)

    def translate_audio_to_text(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sampling_rate: int = 16000
    ) -> str:
        """
        Translate an audio array to text representation with special tokens.

        Args:
            audio: Audio array (numpy array or torch tensor, float32)
            sampling_rate: Sample rate of input audio (default: 16000)

        Returns:
            Text string with audio special tokens like:
            '<|audio_start|><|audio_token_start|><|audio token 000042|>...<|audio_end|>'
        """
        # First tokenize the audio to get token IDs
        token_ids = self.tokenize_audio(audio, sampling_rate)
        token_ids_no_eos_bos = token_ids[1:-1]  # Remove BOS and EOS for text conversion
        # Decode to text using the text tokenizer
        text = self.text_tokenizer.decode(token_ids_no_eos_bos.tolist(), skip_special_tokens=False)

        return text

    def tokenize(self, image=None, text=None, **kwargs) -> torch.Tensor:
        """
        Unified tokenization interface for audio-only mode.

        Args:
            image: Ignored for audio-only tokenization (kept for BaseTokenizer compatibility)
            text: Ignored for audio-only tokenization (kept for BaseTokenizer compatibility)
            **kwargs: Additional arguments:
                - audio: Audio array to tokenize (required) - numpy array or torch tensor
                - sampling_rate: Sample rate of input audio (default: 16000)

        Returns:
            Tokenized audio as tensor
        """
        # Audio is required for audio-only tokenizer
        audio = kwargs.get('audio', None)
        sampling_rate = kwargs.get('sampling_rate', 16000)
        
        if audio is None:
            raise ValueError(
                "Audio input is required for AudioOnlyTokenizer. "
                "Call with: tokenizer(audio=audio_array, sampling_rate=16000)"
            )
        
        # Ignore image and text parameters, only process audio
        return self.tokenize_audio(audio, sampling_rate)

