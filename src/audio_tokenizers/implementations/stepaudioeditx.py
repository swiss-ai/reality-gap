"""
Step-Audio-EditX tokenizer implementation wrapper for benchmark-audio-tokenizer.

Step-Audio-EditX is a dual-codebook audio tokenizer developed by Stepfun AI.
It uses two quantizers (vq02 @ 50Hz and vq06 @ 75Hz) for audio representation.

Paper/Repo: https://github.com/stepfun-ai/Step-Audio-EditX
"""

import os
import sys
import torch
import torchaudio
import numpy as np
from typing import Tuple, Dict, Optional, Any, List
import logging

# Add Step-Audio-EditX repo to path
stepaudioeditx_path = os.path.join(os.path.dirname(__file__), '..', '..', 'repos', 'stepaudioeditx')
sys.path.insert(0, stepaudioeditx_path)

from tokenizer import StepAudioTokenizer
from stepvocoder.cosyvoice2.cli.cosyvoice import CosyVoice
from model_loader import ModelSource

from ..base import BaseAudioTokenizer

logger = logging.getLogger(__name__)


class StepAudioEditXWrapper(BaseAudioTokenizer):
    """
    Wrapper for Step-Audio-EditX dual-codebook tokenizer.

    Key features:
    - Dual codebook: vq02 (50Hz, 1024 codes) + vq06 (25Hz, 4096 codes)
    - Combined tokenization at ~41.6Hz  (2 vq02 + 3 vq06 per 5-token group)
    - Uses FunASR encoder + ONNX-based acoustic tokenizer
    - CosyVoice2 decoder for reconstruction
    """

    name = "stepaudioeditx"

    def __init__(self, device: str = "cuda", checkpoint: Optional[str] = None):
        """
        Initialize Step-Audio-EditX tokenizer.

        Args:
            device: Device to run the model on
            checkpoint: Optional path to local checkpoint directory
        """
        self.device = device
        self.checkpoint = checkpoint

        # Cache for resamplers
        self._resamplers = {}

        self._load_model()

    def _load_model(self):
        """Load the Step-Audio-EditX tokenizer and decoder models."""
        try:
            # Model paths
            cache_dir = "/capstor/store/cscs/swissai/infra01/MLLM/stepaudioeditx"
            os.makedirs(cache_dir, exist_ok=True)

            # Download tokenizer from HuggingFace if not cached
            tokenizer_path = os.path.join(cache_dir, "Step-Audio-Tokenizer")
            if not os.path.exists(tokenizer_path):
                logger.info(f"Downloading Step-Audio-Tokenizer from HuggingFace...")
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id="stepfun-ai/Step-Audio-Tokenizer",
                    local_dir=tokenizer_path,
                    cache_dir=cache_dir
                )
                logger.info(f"Downloaded to {tokenizer_path}")

            # Download Step-Audio-EditX model (contains decoder) if not cached
            model_path = os.path.join(cache_dir, "Step-Audio-EditX")
            if not os.path.exists(model_path):
                logger.info(f"Downloading Step-Audio-EditX model (with decoder) from HuggingFace...")
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id="stepfun-ai/Step-Audio-EditX",
                    local_dir=model_path,
                    cache_dir=cache_dir
                )
                logger.info(f"Downloaded to {model_path}")

            # Load CosyVoice decoder path first (needed for ONNX file)
            vocoder_path = os.path.join(model_path, "CosyVoice-300M-25Hz")
            if not os.path.exists(vocoder_path):
                raise FileNotFoundError(f"CosyVoice decoder not found at {vocoder_path}")

            # Load tokenizer
            # IMPORTANT: Use ONNX model from CosyVoice decoder, not from Step-Audio-Tokenizer
            # The tokenizers have different codebook sizes!
            logger.info(f"Loading Step-Audio-EditX tokenizer")
            logger.info(f"  FunASR model from: {tokenizer_path}")
            logger.info(f"  Speech tokenizer (ONNX) from: {vocoder_path}")

            # Copy CosyVoice ONNX to tokenizer path if needed
            cosy_onnx = os.path.join(vocoder_path, "speech_tokenizer_v1.onnx")
            tokenizer_onnx = os.path.join(tokenizer_path, "dengcunqin",
                                         "speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online",
                                         "speech_tokenizer_v1.onnx")

            if not os.path.exists(tokenizer_onnx):
                import shutil
                os.makedirs(os.path.dirname(tokenizer_onnx), exist_ok=True)
                logger.info(f"Copying CosyVoice ONNX model to tokenizer directory...")
                shutil.copy(cosy_onnx, tokenizer_onnx)

            # Use local mode to avoid re-downloading
            self.tokenizer = StepAudioTokenizer(
                encoder_path=tokenizer_path,
                model_source=ModelSource.LOCAL,
                funasr_model_id="dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online"
            )

            logger.info(f"Loading CosyVoice decoder from {vocoder_path}")
            self.decoder = CosyVoice(vocoder_path)

            logger.info(f"Step-Audio-EditX loaded successfully")
            logger.info(f"  - Dual codebook: vq02 (50Hz, 1024 codes) + vq06 (75Hz, 4096 codes)")
            logger.info(f"  - Combined rate: ~41.6 tokens/second")
            logger.info(f"  - Decoder: CosyVoice-300M-25Hz")
            logger.info(f"  - Token ranges: vq02=[0,1023], vq06=[1024,5119]")

        except Exception as e:
            logger.error(f"Error loading Step-Audio-EditX: {e}")
            raise

    @property
    def sample_rate(self) -> int:
        """Input sample rate for the tokenizer."""
        return 16000  # Step-Audio-EditX uses 16kHz input

    @sample_rate.setter
    def sample_rate(self, value: int):
        """Setter for sample_rate (required by base class)."""
        if value != 16000:
            logger.warning(f"Step-Audio-EditX uses fixed 16kHz sample rate, ignoring requested {value}Hz")

    @property
    def output_sample_rate(self) -> int:
        """Output sample rate for the decoder."""
        return 22050  # CosyVoice outputs at 22.05kHz

    @output_sample_rate.setter
    def output_sample_rate(self, value: int):
        """Setter for output_sample_rate (for consistency)."""
        if value != 22050:
            logger.warning(f"Step-Audio-EditX uses fixed 22.05kHz output rate, ignoring requested {value}Hz")

    @property
    def codebook_size(self) -> int:
        """Size of VQ02 codebook (VQ06 has 4096)."""
        return 1024  # VQ02 has 1024 codes, VQ06 has 4096 codes

    @property
    def downsample_rate(self) -> int:
        """Downsampling rate from audio samples to tokens."""
        # Combined rate is approximately 41.6 tokens/sec
        # 16000 / 41.6 ≈ 384
        return 384

    def encode_audio(self, audio: torch.Tensor, sr: int) -> List[int]:
        """
        Encode audio to discrete tokens using dual codebook.

        Args:
            audio: Audio tensor (B, T) or (B, 1, T)
            sr: Sample rate

        Returns:
            flattened_tokens: Flattened interleaved token sequence for decoder (list of ints)
                             Format: [vq02, vq02, vq06+1024, vq06+1024, vq06+1024, ...]
                             (vq02 in 0-1023, vq06 in 1024-2047)
        """
        # Ensure correct shape (B, T) for tokenizer
        if audio.dim() == 3:
            audio = audio.squeeze(1)  # Remove channel dimension

        # Move to CPU for preprocessing (tokenizer handles device internally)
        audio = audio.cpu()

        # Get dual-codebook tokens
        # wav2token returns: (flattened_tokens_llm, vq02_ori, vq06_ori)
        # We need vq02_ori and vq06_ori for decoder (NOT the LLM-offset version)
        _, vq02_ori, vq06_ori = self.tokenizer.wav2token(
            audio,
            sr,
            enable_trim=False,  # Don't trim for benchmark consistency
            energy_norm=True
        )

        # Create decoder-compatible interleaved sequence
        # Follow merge_vq0206_to_token_str logic but without string conversion
        vq06_offset = [x + 1024 for x in vq06_ori]  # Offset vq06 by 1024

        flattened_tokens = []
        i = 0  # vq02 index
        j = 0  # vq06 index

        # Interleave: 2 vq02 + 3 vq06 per group
        chunk_nums = min(len(vq06_offset) // 3, len(vq02_ori) // 2)
        for idx in range(chunk_nums):
            flattened_tokens.extend([vq02_ori[i], vq02_ori[i+1]])  # 2 vq02
            flattened_tokens.extend([vq06_offset[j], vq06_offset[j+1], vq06_offset[j+2]])  # 3 vq06
            i += 2
            j += 3

        return flattened_tokens

    def decode_tokens(self, flattened_tokens: List[int], speaker_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode flattened interleaved tokens back to audio using CosyVoice.

        Args:
            flattened_tokens: Flattened token sequence [vq02, vq02, vq06, vq06, vq06, ...]
            speaker_embedding: Optional speaker embedding tensor (1, 192). If None, uses zero embedding.

        Returns:
            audio: Audio tensor (B, 1, T)
        """
        # CosyVoice token2wav_nonstream requires:
        # - token: flattened interleaved sequence as tensor
        # - prompt_token: minimal prompt for _reshape
        # - prompt_feat: minimal mel features
        # - embedding: speaker embedding (zero for average voice, or extracted from audio)

        # Convert to tensor
        token = torch.tensor(flattened_tokens, dtype=torch.long).unsqueeze(0)  # (1, T)

        # Create minimal prompt (CosyVoice requires non-empty prompt_token for _reshape)
        # Use padding tokens: [PAD, PAD, 0, 0, 0] where PAD=1024 for vq02 padding
        prompt_token = torch.tensor([1024, 1024, 1024, 1024, 1024], dtype=torch.long).unsqueeze(0)  # (1, 5)
        prompt_feat = torch.zeros((1, 2, 80), dtype=torch.float32)  # (1, 2, 80) - 2 frames for prompt

        # Use provided speaker embedding or zero embedding (average voice)
        if speaker_embedding is None:
            embedding = torch.zeros((1, 192), dtype=torch.float32)  # (1, 192) - campplus embedding dim
        else:
            embedding = speaker_embedding

        # Decode using CosyVoice
        with torch.no_grad():
            audio_output = self.decoder.token2wav_nonstream(
                token,
                prompt_token,
                prompt_feat,
                embedding
            )

        # audio_output is tensor (1, T)
        audio = audio_output.cpu()

        # Ensure output shape is (B, 1, T)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # (1, 1, T)

        return audio

    def encode(self, audio: torch.Tensor, sr: int = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Encode audio to discrete tokens.

        Args:
            audio: Audio tensor
            sr: Sample rate of input audio

        Returns:
            tokens: Flattened interleaved token tensor
            info: Dictionary with encoding information
        """
        if sr is None:
            sr = self.sample_rate

        # Convert to tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()

        # Ensure audio is on CPU for resampling
        audio = audio.cpu()

        # Resample if necessary
        if sr != self.sample_rate:
            # Use cached resampler or create new one
            resampler_key = f"{sr}_{self.sample_rate}"
            if resampler_key not in self._resamplers:
                self._resamplers[resampler_key] = torchaudio.transforms.Resample(sr, self.sample_rate)

            resampler = self._resamplers[resampler_key]

            # Ensure correct shape for resampling (B, C, T)
            if audio.dim() == 2:
                audio = audio.unsqueeze(1)  # Add channel dimension

            audio = resampler(audio)

            # Convert to mono if needed
            if audio.shape[1] > 1:
                audio = audio.mean(1, keepdim=True)

            # Remove channel dimension for tokenizer
            if audio.dim() == 3:
                audio = audio.squeeze(1)

        # Extract speaker embedding for voice cloning
        speaker_embedding = self.decoder.frontend.extract_spk_embedding(audio, self.sample_rate)

        # Encode to dual codebook
        flattened_tokens = self.encode_audio(audio, self.sample_rate)

        # Convert flattened tokens to tensor
        tokens = torch.tensor(flattened_tokens, dtype=torch.long).unsqueeze(0)  # (1, T)

        # Calculate individual token counts (every 5 tokens = 2 vq02 + 3 vq06)
        num_groups = len(flattened_tokens) // 5
        num_vq02 = num_groups * 2
        num_vq06 = num_groups * 3

        info = {
            "num_tokens": len(flattened_tokens),
            "token_shape": list(tokens.shape),
            "num_tokens_vq02": num_vq02,
            "num_tokens_vq06": num_vq06,
            "sample_rate": self.sample_rate,
            "vq02_rate": 50,  # Hz
            "vq06_rate": 25,  # Hz
            "interleave_pattern": "2 vq02 + 3 vq06 per group",
            "speaker_embedding": speaker_embedding  # Store for reconstruction
        }

        return tokens, info

    def decode(self, tokens: torch.Tensor, speaker_embedding: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Decode discrete tokens back to audio.

        Args:
            tokens: Flattened interleaved token tensor (B, T)
            speaker_embedding: Optional speaker embedding for voice cloning

        Returns:
            audio: Reconstructed audio tensor
            info: Dictionary with decoding information
        """
        # Convert to list for decoder
        flattened_tokens = tokens.squeeze(0).tolist()

        # Decode with speaker embedding
        audio = self.decode_tokens(flattened_tokens, speaker_embedding=speaker_embedding)

        # Remove channel dimension if single channel
        if audio.dim() == 3 and audio.shape[1] == 1:
            audio = audio.squeeze(1)

        info = {
            "output_sample_rate": self.output_sample_rate,
            "num_tokens": len(flattened_tokens)
        }

        return audio, info

    def reconstruct(self, audio: torch.Tensor, sr: int = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Encode and decode audio for reconstruction with voice cloning.

        Args:
            audio: Input audio tensor
            sr: Sample rate of input audio

        Returns:
            reconstructed: Reconstructed audio
            info: Combined encoding and decoding information
        """
        # Encode (extracts speaker embedding automatically)
        tokens_dict, encode_info = self.encode(audio, sr)

        # Extract speaker embedding from encode_info
        speaker_embedding = encode_info.get("speaker_embedding", None)

        # Decode with speaker embedding for voice cloning
        reconstructed, decode_info = self.decode(tokens_dict, speaker_embedding=speaker_embedding)

        # Combine info (remove speaker_embedding from final info as it's a tensor)
        info = {**encode_info, **decode_info}
        if "speaker_embedding" in info:
            del info["speaker_embedding"]  # Don't include tensor in metadata

        return reconstructed, info
