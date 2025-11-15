"""
CosyVoice 2 Speech Tokenizer with full encode/decode support.

Architecture:
- Codebook: 6561 tokens (81^2)
- Token rate: 25 Hz
- Mel rate: 50 Hz (2x upsampling)
- Input: 16 kHz audio
- Output: 24 kHz audio
"""

import sys
import torch
import numpy as np
import onnxruntime
import whisper
from pathlib import Path
from typing import Optional

from ..base import BaseAudioTokenizer


class CosyVoice2Tokenizer(BaseAudioTokenizer):
    """
    CosyVoice 2 Speech Tokenizer with full reconstruction support.
    """

    name = "cosyvoice2"
    repo_path = "cosyvoice"
    default_checkpoint = "iic/CosyVoice2-0.5B"
    default_sample_rate = 16000

    def __init__(self,
                 checkpoint: Optional[str] = None,
                 device: str = "cuda",
                 sample_rate: Optional[int] = None,
                 load_decoder: bool = True,
                 **kwargs):
        """
        Initialize CosyVoice 2 tokenizer.

        Args:
            checkpoint: Model checkpoint (default: iic/CosyVoice2-0.5B)
            device: Device to run on
            sample_rate: Input sample rate (must be 16000)
            load_decoder: Whether to load decoder models for reconstruction
        """
        self.load_decoder = load_decoder
        super().__init__(checkpoint, device, sample_rate, **kwargs)

    def _load_model(self) -> None:
        """Load CosyVoice 2 models."""
        # Add CosyVoice to path
        cosyvoice_path = Path(__file__).parent.parent.parent / "repos" / "cosyvoice"
        if str(cosyvoice_path) not in sys.path:
            sys.path.insert(0, str(cosyvoice_path))

        from modelscope import snapshot_download

        # Download or locate model
        if self.checkpoint and Path(self.checkpoint).exists():
            self.model_dir = Path(self.checkpoint)
        else:
            # Use ModelScope for complete model files
            print(f"Downloading {self.checkpoint} from ModelScope...")
            self.model_dir = Path(snapshot_download(self.checkpoint))

        print(f"Model directory: {self.model_dir}")

        # Load ONNX tokenizer for encoding (CosyVoice 2 uses v2)
        tokenizer_path = self.model_dir / "speech_tokenizer_v2.onnx"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Speech tokenizer v2 not found in {self.model_dir}")

        print(f"Loading tokenizer from: {tokenizer_path}")

        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CUDAExecutionProvider"] if self.device != "cpu" else ["CPUExecutionProvider"]

        self.tokenizer_session = onnxruntime.InferenceSession(
            str(tokenizer_path),
            sess_options=option,
            providers=providers
        )

        # Load decoder models if requested
        self.cosyvoice = None
        if self.load_decoder:
            try:
                print("Loading CosyVoice 2 decoder models...")
                self._load_cosyvoice_model()
                print("✓ Decoder models loaded successfully")
            except ImportError as e:
                print(f"Warning: Could not load decoder models due to missing dependency: {e}")
                print("Please install missing dependencies from requirements-cosyvoice2.txt")
                print("Encoding will work, but decoding will not be available.")
                self.cosyvoice = None
            except FileNotFoundError as e:
                print(f"Warning: Could not load decoder models due to missing file: {e}")
                print("Please ensure the model checkpoint is complete.")
                print("Encoding will work, but decoding will not be available.")
                self.cosyvoice = None
            except Exception as e:
                print(f"Error: Failed to load decoder models: {e}")
                import traceback
                traceback.print_exc()
                raise

        # Set properties
        self._codebook_size = 6561  # 81^2
        self._downsample_rate = 640  # 16000 / 25
        self._output_sample_rate = 24000

    def _load_cosyvoice_model(self):
        """Load the full CosyVoice2 model for decoding."""
        import sys
        import os

        # Add necessary paths
        cosyvoice_path = Path(__file__).parent.parent.parent / "repos" / "cosyvoice"
        matcha_path = cosyvoice_path / "third_party" / "Matcha-TTS"

        for path in [str(cosyvoice_path), str(matcha_path)]:
            if path not in sys.path:
                sys.path.insert(0, path)

        from hyperpyyaml import load_hyperpyyaml
        from cosyvoice.cli.model import CosyVoice2Model
        from cosyvoice.utils.class_utils import get_model_type

        # Load config with proper override for empty qwen_pretrain_path
        config_path = self.model_dir / "cosyvoice2.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Override the empty qwen_pretrain_path as CosyVoice2 does
        overrides = {
            'qwen_pretrain_path': str(self.model_dir / 'CosyVoice-BlankEN')
        }

        print(f"Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            configs = load_hyperpyyaml(f, overrides=overrides)

        # Verify it's a CosyVoice2 model
        assert get_model_type(configs) == CosyVoice2Model, 'Model is not CosyVoice2!'

        # Initialize the model
        self.model = CosyVoice2Model(
            configs['llm'],
            configs['flow'],
            configs['hift'],
            fp16=(self.device != "cpu")
        )

        # Load model weights
        self.model.load(
            str(self.model_dir / "llm.pt"),
            str(self.model_dir / "flow.pt"),
            str(self.model_dir / "hift.pt")
        )

        print("✓ CosyVoice2 decoder models loaded successfully")

        # Store for compatibility
        self.cosyvoice = self


    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to tokens at 25 Hz."""
        # Assume single audio input [1, channels, samples]
        if audio.ndim == 3:
            audio = audio[0]  # Remove batch dim

        # Convert to mono if needed
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0)
        else:
            audio = audio[0]

        # Convert to numpy
        audio_np = audio.cpu().numpy()
        tokens = []

        # Process in 30-second chunks for long audio
        for i in range(0, len(audio_np), 30 * 16000):
            chunk = audio_np[i:i + 30 * 16000]

            # Extract mel features using whisper
            chunk_tensor = torch.from_numpy(chunk).float().to(self.device)
            mel = whisper.log_mel_spectrogram(chunk_tensor, n_mels=128)

            # Add batch dim if needed
            if mel.ndim == 2:
                mel = mel.unsqueeze(0)

            # Run ONNX tokenizer
            outputs = self.tokenizer_session.run(
                None,
                {
                    self.tokenizer_session.get_inputs()[0].name: mel.cpu().numpy(),
                    self.tokenizer_session.get_inputs()[1].name: np.array([mel.shape[2]], dtype=np.int32)
                }
            )

            tokens.extend(outputs[0].flatten().tolist())

        # Return as tensor with batch dim for compatibility
        return torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)

    def decode_tokens(self, tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decode tokens to audio using CosyVoice2Model.

        IMPORTANT: Current implementation uses ZERO embeddings for reconstruction,
        producing a generic/default voice. For actual voice cloning, you need to:
        1. Provide reference audio (3-10 seconds of target speaker)
        2. Extract speaker embedding from reference: model.extract_speaker_embedding(ref_audio)
        3. Pass embedding to token2wav instead of zeros

        TODO: Add support for reference_audio parameter to enable voice cloning:
            decode_tokens(tokens, reference_audio=audio_sample)
        """
        if self.model is None:
            raise RuntimeError(
                "Decoder models not loaded. Initialize with load_decoder=True "
                "to enable audio reconstruction."
            )

        # Ensure correct shape [1, sequence_length]
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)

        # Remove padding (0 tokens)
        mask = tokens != 0
        if mask.any():
            length = mask.sum().item()
            tokens = tokens[:, :length]

        tokens = tokens.to(self.device)

        # Create a unique session ID
        import uuid as uuid_lib
        session_uuid = str(uuid_lib.uuid4())

        # Initialize cache for this session
        self.model.hift_cache_dict[session_uuid] = None

        # Use first 10 tokens as prompt (or less if sequence is shorter)
        prompt_len = min(10, tokens.shape[1])
        prompt_token = tokens[:, :prompt_len]

        # Create minimal inputs for token2wav
        # NOTE: Using zero embeddings produces generic/default voice
        # For voice cloning, replace with actual speaker embeddings from reference audio
        embedding = torch.zeros(1, 192).to(self.device)  # Speaker embedding (ZERO = no voice cloning!)
        prompt_feat = torch.zeros(1, prompt_len * 2, 80).to(self.device)  # Mel features

        with torch.no_grad():
            # Call token2wav to generate audio - it returns a tensor directly, not a generator
            audio = self.model.token2wav(
                token=tokens,
                prompt_token=prompt_token,
                prompt_feat=prompt_feat,
                embedding=embedding,
                token_offset=0,
                uuid=session_uuid,
                stream=False,
                finalize=True,
                speed=kwargs.get('speed', 1.0)
            )

        # Clean up cache after generation
        if session_uuid in self.model.hift_cache_dict:
            del self.model.hift_cache_dict[session_uuid]

        # Ensure output shape [1, 1, samples]
        if audio.ndim == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.ndim == 2:
            audio = audio.unsqueeze(0)

        return audio

    @property
    def codebook_size(self) -> int:
        return self._codebook_size

    @property
    def downsample_rate(self) -> int:
        return self._downsample_rate

    @property
    def output_sample_rate(self) -> int:
        return self._output_sample_rate

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"checkpoint='{self.checkpoint}', "
                f"device='{self.device}', "
                f"sample_rate={self.sample_rate}, "
                f"output_sample_rate={self.output_sample_rate}, "
                f"decoder_loaded={self.cosyvoice is not None})")