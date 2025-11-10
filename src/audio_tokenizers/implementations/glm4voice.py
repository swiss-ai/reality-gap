"""
GLM-4-Voice tokenizer wrapper with CAMPPlus speaker encoder support.

GLM-4-Voice uses a Whisper-based tokenizer with Vector Quantization that converts
continuous speech to discrete tokens at ~12.5 tokens per second.

This implementation includes CAMPPlus speaker encoder for voice cloning during
reconstruction, allowing different voices to be applied to the decoded audio.

Note: The decoder model is located at: /capstor/store/cscs/swissai/infra01/MLLM/glm-4-voice-decoder-hf
      The CAMPPlus model is located at: /capstor/store/cscs/swissai/infra01/MLLM/cosyvoice-campplus/campplus.onnx
"""

import os
import sys
import torch
import torch.nn as nn
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from transformers import WhisperFeatureExtractor
import logging

# Add GLM-4-Voice paths to system
glm4voice_path = Path(__file__).parent.parent.parent / "repos" / "glm4voice"
sys.path.insert(0, str(glm4voice_path))
sys.path.insert(0, str(glm4voice_path / "cosyvoice"))
sys.path.insert(0, str(glm4voice_path / "third_party" / "Matcha-TTS"))

from ..base.tokenizer_base import BaseAudioTokenizer

logger = logging.getLogger(__name__)


class CAMPPlusSpeakerEncoder:
    """
    CAMPPlus speaker encoder for extracting 192-dim speaker embeddings.
    Uses the actual CAMPPlus ONNX model from CosyVoice.
    """

    def __init__(self, model_path: str = "/capstor/store/cscs/swissai/infra01/MLLM/cosyvoice-campplus/campplus.onnx"):
        """Initialize CAMPPlus speaker encoder."""
        try:
            import onnxruntime
            self.model_path = model_path

            if os.path.exists(model_path):
                option = onnxruntime.SessionOptions()
                option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
                option.intra_op_num_threads = 1

                self.session = onnxruntime.InferenceSession(
                    model_path,
                    sess_options=option,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                )
                logger.info(f"CAMPPlus speaker encoder loaded from {model_path}")
            else:
                logger.warning(f"CAMPPlus model not found at {model_path}")
                self.session = None
        except Exception as e:
            logger.warning(f"Could not load CAMPPlus model: {e}")
            self.session = None

    def extract_embedding(self, audio: torch.Tensor, sr: int = 16000) -> torch.Tensor:
        """
        Extract speaker embedding from audio.

        Args:
            audio: Audio tensor (samples,) or (1, samples)
            sr: Sample rate (should be 16000)

        Returns:
            Speaker embedding tensor (1, 192)
        """
        if self.session is None:
            # Return zero embedding if model not available
            return torch.zeros(1, 192)

        try:
            # Ensure audio is tensor
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio).float()

            # Ensure correct shape
            if audio.dim() == 2:
                audio = audio.squeeze(0)
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)

            # Resample to 16kHz if needed
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                audio = resampler(audio)

            # Extract 80-dim Fbank features (CAMPPlus input)
            feat = kaldi.fbank(
                audio,
                num_mel_bins=80,
                dither=0,
                sample_frequency=16000
            )

            # Normalize features
            feat = feat - feat.mean(dim=0, keepdim=True)

            # Run through CAMPPlus model
            embedding = self.session.run(
                None,
                {self.session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()}
            )[0].flatten()

            # Convert to tensor
            embedding = torch.tensor(embedding).unsqueeze(0)  # (1, 192)

            return embedding

        except Exception as e:
            logger.warning(f"Error extracting embedding: {e}")
            return torch.zeros(1, 192)


class GLM4VoiceTokenizer(BaseAudioTokenizer):
    """
    Wrapper for GLM-4-Voice audio tokenizer.

    GLM-4-Voice uses a Whisper-based encoder with Vector Quantization
    to convert speech to discrete tokens, achieving ~12.5 tokens/second.

    Note on voice cloning:
    While GLM-4-Voice supports speaker embeddings through CAMPPlus, the effect
    is very subtle (~2-3% signal difference). The model appears to have limited
    voice cloning capabilities - it wasn't specifically trained for strong voice
    preservation. The embeddings do affect the output but the perceptual difference
    is minimal. For this reason, we default to not using speaker embeddings.
    """

    name = "glm4voice"

    def __init__(self, device: str = "cuda", checkpoint: Optional[str] = None, use_speaker_encoder: bool = False):
        """
        Initialize GLM-4-Voice tokenizer.

        Args:
            device: Device to run the model on
            checkpoint: Optional path to local checkpoint
            use_speaker_encoder: Whether to use CAMPPlus for speaker embeddings (default: False,
                                 as the effect is minimal and not worth the computational cost)
        """
        self.device = device
        self.checkpoint = checkpoint
        self.use_speaker_encoder = use_speaker_encoder

        # Model paths
        if checkpoint:
            self.tokenizer_path = checkpoint
            self.decoder_path = os.path.join(checkpoint, "decoder")
        else:
            # Use HuggingFace models
            self.tokenizer_path = "THUDM/glm-4-voice-tokenizer"
            # Check for decoder in capstor
            self.decoder_path = "/capstor/store/cscs/swissai/infra01/MLLM/glm-4-voice-decoder-hf"
            if not os.path.exists(self.decoder_path):
                self.decoder_path = None

        # Initialize base class (will call _load_model)
        super().__init__(device=device)

    def _load_model(self) -> None:
        """Load the GLM-4-Voice tokenizer model."""
        try:
            # Import GLM-4-Voice modules
            from speech_tokenizer.modeling_whisper import WhisperVQEncoder
            from speech_tokenizer.utils import extract_speech_token

            # Store imports
            self.WhisperVQEncoder = WhisperVQEncoder
            self.extract_speech_token = extract_speech_token

            # Load Whisper feature extractor
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.tokenizer_path)

            # Load Whisper VQ encoder (store as self.model for base class)
            self.model = self.WhisperVQEncoder.from_pretrained(self.tokenizer_path)
            self.model.to(self.device)
            self.model.eval()

            # Try to load decoder if available
            self.audio_decoder = None
            self._try_load_decoder()

            # Initialize speaker encoder
            if self.use_speaker_encoder:
                self.speaker_encoder = CAMPPlusSpeakerEncoder()
            else:
                self.speaker_encoder = None

            logger.info(f"GLM-4-Voice tokenizer initialized on {self.device}")
            if self.speaker_encoder and self.speaker_encoder.session:
                logger.info("CAMPPlus speaker encoder loaded for voice cloning")

        except Exception as e:
            logger.error(f"Error loading GLM-4-Voice tokenizer: {e}")
            raise

    def _try_load_decoder(self):
        """Try to load the decoder model if available."""
        try:
            from flow_inference import AudioDecoder

            if self.decoder_path and os.path.exists(self.decoder_path):
                flow_config = os.path.join(self.decoder_path, "config.yaml")
                flow_checkpoint = os.path.join(self.decoder_path, 'flow.pt')
                hift_checkpoint = os.path.join(self.decoder_path, 'hift.pt')

                if all(os.path.exists(p) for p in [flow_config, flow_checkpoint, hift_checkpoint]):
                    self.audio_decoder = AudioDecoder(
                        config_path=flow_config,
                        flow_ckpt_path=flow_checkpoint,
                        hift_ckpt_path=hift_checkpoint,
                        device=self.device
                    )
                    logger.info("GLM-4-Voice decoder loaded successfully")
                else:
                    logger.warning("Decoder files not found. Decoding will not be available.")
            else:
                logger.warning("Decoder not available. Only encoding will work.")
                self.audio_decoder = None

        except Exception as e:
            logger.warning(f"Could not load decoder: {e}. Only encoding will work.")
            self.audio_decoder = None

    @property
    def sample_rate(self) -> int:
        """Input sample rate for the tokenizer."""
        return 16000  # Whisper uses 16kHz

    @sample_rate.setter
    def sample_rate(self, value: int):
        """Setter for sample_rate (required by base class)."""
        # GLM-4-Voice is fixed at 16kHz, so we just log if different
        if value != 16000:
            logger.warning(f"GLM-4-Voice uses fixed 16kHz sample rate, ignoring requested {value}Hz")

    @property
    def output_sample_rate(self) -> int:
        """Output sample rate for the decoder."""
        return 22050  # GLM-4-Voice decoder outputs at 22.05kHz

    @output_sample_rate.setter
    def output_sample_rate(self, value: int):
        """Setter for output_sample_rate (for consistency)."""
        if value != 22050:
            logger.warning(f"GLM-4-Voice uses fixed 22.05kHz output rate, ignoring requested {value}Hz")

    @property
    def codebook_size(self) -> int:
        """Size of the tokenizer's codebook."""
        # GLM-4-Voice uses a discrete codebook
        # Based on the paper, they use a finite vocabulary
        return 4096  # Default VQ codebook size

    @property
    def downsample_rate(self) -> int:
        """Temporal downsampling factor."""
        # GLM-4-Voice achieves ~12.5 tokens per second
        # With 16kHz input, this is approximately 16000/12.5 = 1280
        return 1280

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode audio tensor to discrete tokens.

        Args:
            audio: Audio tensor (B, C, T) at self.sample_rate

        Returns:
            Token codes (B, N) where N is number of tokens
        """
        # Ensure audio is in correct format
        if audio.dim() == 2 and audio.shape[0] == 1:
            # (1, T) -> (T,)
            audio = audio.squeeze(0)
        elif audio.dim() == 3:
            # (B, C, T) -> (B, T)
            if audio.shape[1] == 1:
                audio = audio.squeeze(1)

        # Process single sample (no batch processing needed)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Process the first sample only (GLM-4-Voice doesn't support batching well)
        audio_tensor = audio[0:1]  # Keep as tensor with shape (1, T)

        # Encode with Whisper VQ encoder
        with torch.no_grad():
            # Get speech tokens using the correct signature
            # extract_speech_token expects: model, feature_extractor, utts
            # where utts is a list of (audio_tensor, sample_rate) tuples
            # The audio should be a tensor, not numpy
            utts = [(audio_tensor, self.sample_rate)]
            # extract_speech_token returns a list of token lists
            all_speech_tokens = self.extract_speech_token(
                self.model,
                self.feature_extractor,
                utts
            )

            # Get the first (and only) utterance's tokens
            speech_tokens = all_speech_tokens[0] if all_speech_tokens else []

            # Convert list to tensor
            speech_tokens = torch.tensor(speech_tokens, dtype=torch.long, device=self.device)

            # Add batch dimension if needed
            if speech_tokens.dim() == 0:
                speech_tokens = speech_tokens.unsqueeze(0)
            if speech_tokens.dim() == 1:
                speech_tokens = speech_tokens.unsqueeze(0)

        return speech_tokens

    def decode_tokens(self, tokens: torch.Tensor, speaker_embedding: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Decode tokens back to audio.

        Args:
            tokens: Token codes (B, N)
            speaker_embedding: Optional speaker embedding (1, 192)
            **kwargs: Additional arguments

        Returns:
            Audio tensor (B, 1, T) at output_sample_rate
        """
        if self.audio_decoder is None:
            # Return zeros if decoder not available
            logger.warning("Decoder not available. Returning zero audio.")
            # Estimate output length based on tokens
            output_len = tokens.shape[-1] * self.downsample_rate * self.output_sample_rate // self.sample_rate
            return torch.zeros(1, 1, output_len, device=self.device)

        # Use provided embedding or default zeros
        if speaker_embedding is None:
            speaker_embedding = torch.zeros(1, 192).to(self.device)
        else:
            speaker_embedding = speaker_embedding.to(self.device)

        # Ensure tokens are 2D
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        # Process single sample only
        token_sample = tokens[0].unsqueeze(0) if tokens.shape[0] > 0 else tokens

        with torch.no_grad():
            # Use AudioDecoder's token2wav with speaker embedding
            import uuid
            this_uuid = str(uuid.uuid1())

            # Decode tokens to audio with speaker embedding
            audio, _ = self.audio_decoder.token2wav(
                token_sample,
                uuid=this_uuid,
                embedding=speaker_embedding,
                finalize=True
            )

            # Get CPU tensor
            audio = audio.cpu()

            # Ensure audio is (1, T)
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            # Add channel dimension to make (B=1, C=1, T)
            if audio.dim() == 2:
                audio = audio.unsqueeze(1)

        return audio

    def encode(self, audio: torch.Tensor, sr: int = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Encode audio to discrete tokens.

        Args:
            audio: Audio tensor or numpy array
            sr: Sample rate of input audio

        Returns:
            tokens: Discrete token tensor
            info: Dictionary with encoding information
        """
        if sr is None:
            sr = self.sample_rate

        # Convert numpy to tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()

        # Move to device if needed
        if audio.device != self.device:
            audio = audio.to(self.device)

        # Resample if necessary
        if sr != self.sample_rate:
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate).to(audio.device)
            audio = resampler(audio)
            if audio.shape[0] == 1:
                audio = audio.squeeze(0)

        # Ensure correct shape for encode_audio
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)  # (B=1, C=1, T)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(1)  # (B, C=1, T)

        # Encode
        tokens = self.encode_audio(audio)

        # Calculate statistics
        audio_duration = audio.shape[-1] / self.sample_rate
        tokens_per_second = tokens.shape[-1] / audio_duration if audio_duration > 0 else 0

        info = {
            "tokens_per_second": tokens_per_second,
            "compression_ratio": audio.shape[-1] / tokens.shape[-1] if tokens.shape[-1] > 0 else 0,
            "num_tokens": tokens.shape[-1]
        }

        return tokens, info

    def decode(self, tokens: torch.Tensor, reference_audio: Optional[torch.Tensor] = None, reference_sr: int = 16000) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Decode discrete tokens back to audio.

        Args:
            tokens: Discrete token tensor
            reference_audio: Optional reference audio for voice cloning
            reference_sr: Sample rate of reference audio

        Returns:
            audio: Reconstructed audio tensor
            info: Dictionary with decoding information
        """
        # Extract speaker embedding if reference audio provided
        speaker_embedding = None
        if reference_audio is not None and self.speaker_encoder is not None:
            speaker_embedding = self.speaker_encoder.extract_embedding(reference_audio, reference_sr)
            info_voice = "cloned from reference"
        else:
            info_voice = "default voice"

        # Decode
        audio = self.decode_tokens(tokens, speaker_embedding=speaker_embedding)

        # Remove channel dimension if single channel
        if audio.dim() == 3 and audio.shape[1] == 1:
            audio = audio.squeeze(1)

        info = {
            "output_sample_rate": self.output_sample_rate,
            "num_tokens": tokens.shape[-1],
            "voice": info_voice
        }

        return audio, info

    def reconstruct(self, audio: torch.Tensor, sr: int = None, reference_audio: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Encode and decode audio for reconstruction with optional voice cloning.

        Args:
            audio: Input audio tensor
            sr: Sample rate of input audio
            reference_audio: Optional reference audio for voice cloning (if None, uses input audio itself)

        Returns:
            reconstructed: Reconstructed audio
            info: Combined encoding and decoding information
        """
        # Encode
        tokens, encode_info = self.encode(audio, sr)

        # Use input audio as reference if not provided
        if reference_audio is None:
            reference_audio = audio
            reference_sr = sr if sr is not None else self.sample_rate
        else:
            reference_sr = sr if sr is not None else self.sample_rate

        # Decode with voice cloning
        reconstructed, decode_info = self.decode(tokens, reference_audio=reference_audio, reference_sr=reference_sr)

        # Combine info and add token metadata
        info = {
            **encode_info,
            **decode_info,
            "num_tokens": tokens.numel(),
            "token_shape": list(tokens.shape)
        }

        return reconstructed, info