"""CosyVoice2 TTS backend: text -> speech tokens (6561 codebook, 25 Hz).

Two token extraction strategies:
  Option A (primary): Intercept LLM-generated tokens before flow+hift.
  Option B (fallback): Full TTS -> re-encode audio with S3Tokenizer ONNX.

Cross-lingual mode is used for Polish (reference audio for voice cloning).

Requires the CosyVoice repo on sys.path and the cosyvoice2 venv.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..base import TTSBackend, TTSOutput

logger = logging.getLogger(__name__)

# Repo paths (same layout as audio_tokenizers)
REPOS_DIR = Path(__file__).parent.parent.parent / "repos"


class CosyVoice2TTSBackend(TTSBackend):
    """CosyVoice2 TTS backend for text -> speech token generation.

    Args:
        checkpoint: ModelScope ID or local path (default: iic/CosyVoice2-0.5B).
        device: Target device.
        mode: Inference mode — "cross_lingual" (default for Polish) or "zero_shot".
        extract_tokens_directly: If True, use Option A (intercept LLM tokens).
            If False, use Option B (full TTS + re-encode). Default True.
    """

    CODEBOOK_SIZE = 6561  # 81^2 FSQ
    TOKEN_RATE_HZ = 25.0
    INPUT_SR = 16000
    OUTPUT_SR = 24000

    def __init__(
        self,
        checkpoint: str = "iic/CosyVoice2-0.5B",
        device: str = "cuda",
        mode: str = "cross_lingual",
        extract_tokens_directly: bool = True,
    ):
        self.checkpoint = checkpoint
        self.device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        self.mode = mode
        self.extract_tokens_directly = extract_tokens_directly
        self._cosyvoice = None  # High-level CosyVoice2 wrapper
        self._onnx_session = None  # S3Tokenizer ONNX for Option B fallback

    def load_model(self, device: Optional[str] = None) -> None:
        """Load CosyVoice2 model (frontend + LLM + flow + hift)."""
        if device:
            self.device = device

        self._setup_paths()
        self._load_cosyvoice()
        logger.info("CosyVoice2 TTS backend loaded on %s", self.device)

    def _setup_paths(self) -> None:
        """Add CosyVoice repo and dependencies to sys.path."""
        cosyvoice_path = REPOS_DIR / "cosyvoice"
        matcha_path = cosyvoice_path / "third_party" / "Matcha-TTS"

        for path in [cosyvoice_path, matcha_path]:
            path_str = str(path)
            if path.exists() and path_str not in sys.path:
                sys.path.insert(0, path_str)
                logger.debug("Added %s to sys.path", path_str)

    def _load_cosyvoice(self) -> None:
        """Load the high-level CosyVoice2 model (includes frontend)."""
        from modelscope import snapshot_download

        # Resolve model directory
        if Path(self.checkpoint).exists():
            model_dir = str(self.checkpoint)
        else:
            logger.info("Downloading %s from ModelScope...", self.checkpoint)
            model_dir = snapshot_download(self.checkpoint)

        # Import and instantiate the high-level CosyVoice2 class
        # This loads: frontend (BPE tokenizer) + model (LLM + flow + hift)
        from cosyvoice.cli.cosyvoice import CosyVoice2 as CosyVoiceModel

        self._cosyvoice = CosyVoiceModel(model_dir, load_jit=False, load_trt=False)
        logger.info("CosyVoice2 model loaded from %s", model_dir)

        # For Option B fallback, also prepare the ONNX tokenizer
        if not self.extract_tokens_directly:
            self._load_onnx_tokenizer(Path(model_dir))

    def _load_onnx_tokenizer(self, model_dir: Path) -> None:
        """Load S3Tokenizer ONNX for Option B (re-encode audio -> tokens)."""
        import onnxruntime

        tokenizer_path = model_dir / "speech_tokenizer_v2.onnx"
        if not tokenizer_path.exists():
            logger.warning("ONNX tokenizer not found at %s, Option B unavailable", tokenizer_path)
            return

        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = (
            ["CUDAExecutionProvider"] if self.device != "cpu" else ["CPUExecutionProvider"]
        )
        self._onnx_session = onnxruntime.InferenceSession(
            str(tokenizer_path), sess_options=option, providers=providers
        )
        logger.info("ONNX S3Tokenizer loaded for fallback re-encoding")

    def generate(
        self,
        text: str,
        reference_audio: Optional[torch.Tensor] = None,
        reference_audio_sr: Optional[int] = None,
        speaker_id: Optional[str] = None,
        render_audio: bool = False,
        prompt_text: str = "",
        **kwargs,
    ) -> TTSOutput:
        """Generate speech tokens from text.

        Args:
            text: Normalized text to synthesize.
            reference_audio: Reference audio tensor (1D, mono) for voice cloning.
            reference_audio_sr: Sample rate of reference audio.
            speaker_id: Not used for cross-lingual mode.
            render_audio: Also render WAV for sanity checking.
            prompt_text: Transcript of reference audio (needed for zero_shot mode).
        """
        if self._cosyvoice is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Resample reference audio to 16kHz if needed
        ref_audio_16k = self._prepare_reference(reference_audio, reference_audio_sr)

        if self.extract_tokens_directly:
            return self._generate_option_a(
                text, ref_audio_16k, prompt_text, render_audio, **kwargs
            )
        else:
            return self._generate_option_b(
                text, ref_audio_16k, prompt_text, render_audio, **kwargs
            )

    def _generate_option_a(
        self,
        text: str,
        ref_audio_16k: torch.Tensor,
        prompt_text: str,
        render_audio: bool,
        **kwargs,
    ) -> TTSOutput:
        """Option A: Intercept LLM speech tokens before flow+hift.

        Calls the CosyVoice2 inference method and collects both
        the generated audio and extracts tokens from it via the
        S3Tokenizer. If the CosyVoice2 internal API exposes tokens
        directly, those are used instead.
        """
        # Run inference via CosyVoice2's high-level API
        # The inference methods are generators yielding {'tts_speech': tensor}
        audio_chunks = []

        if self.mode == "cross_lingual":
            generator = self._cosyvoice.inference_cross_lingual(
                text, ref_audio_16k, stream=False, speed=kwargs.get("speed", 1.0)
            )
        elif self.mode == "zero_shot":
            generator = self._cosyvoice.inference_zero_shot(
                text, prompt_text, ref_audio_16k, stream=False, speed=kwargs.get("speed", 1.0)
            )
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        for result in generator:
            audio_chunks.append(result["tts_speech"])

        if not audio_chunks:
            raise RuntimeError(f"CosyVoice2 produced no output for: {text!r}")

        full_audio = torch.cat(audio_chunks, dim=-1)  # (1, samples) at 24kHz

        # Extract tokens from the generated audio via S3Tokenizer
        # This is technically Option B applied after Option A's audio generation,
        # but it guarantees token-audio alignment. Once we verify the internal
        # token interception API on the cluster, this can be replaced with direct
        # LLM token capture.
        # TODO: Replace with direct LLM token interception once verified on cluster.
        # The CosyVoice2Model.tts() generator may yield token tensors directly
        # (check self._cosyvoice.model.tts() signature).
        speech_tokens = self._audio_to_tokens(full_audio, sample_rate=self.OUTPUT_SR)

        return TTSOutput(
            speech_tokens=speech_tokens,
            codebook_size=self.CODEBOOK_SIZE,
            token_rate_hz=self.TOKEN_RATE_HZ,
            audio=full_audio.squeeze() if render_audio else None,
            audio_sample_rate=self.OUTPUT_SR if render_audio else None,
            metadata={
                "backend": "cosyvoice2",
                "mode": self.mode,
                "extraction": "option_a",
            },
        )

    def _generate_option_b(
        self,
        text: str,
        ref_audio_16k: torch.Tensor,
        prompt_text: str,
        render_audio: bool,
        **kwargs,
    ) -> TTSOutput:
        """Option B: Full TTS -> re-encode audio with S3Tokenizer ONNX."""
        # Generate audio via CosyVoice2
        audio_chunks = []

        if self.mode == "cross_lingual":
            generator = self._cosyvoice.inference_cross_lingual(
                text, ref_audio_16k, stream=False, speed=kwargs.get("speed", 1.0)
            )
        elif self.mode == "zero_shot":
            generator = self._cosyvoice.inference_zero_shot(
                text, prompt_text, ref_audio_16k, stream=False, speed=kwargs.get("speed", 1.0)
            )
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        for result in generator:
            audio_chunks.append(result["tts_speech"])

        if not audio_chunks:
            raise RuntimeError(f"CosyVoice2 produced no output for: {text!r}")

        full_audio = torch.cat(audio_chunks, dim=-1)  # (1, samples) at 24kHz

        # Re-encode to tokens via S3Tokenizer
        speech_tokens = self._audio_to_tokens(full_audio, sample_rate=self.OUTPUT_SR)

        return TTSOutput(
            speech_tokens=speech_tokens,
            codebook_size=self.CODEBOOK_SIZE,
            token_rate_hz=self.TOKEN_RATE_HZ,
            audio=full_audio.squeeze() if render_audio else None,
            audio_sample_rate=self.OUTPUT_SR if render_audio else None,
            metadata={
                "backend": "cosyvoice2",
                "mode": self.mode,
                "extraction": "option_b",
            },
        )

    def _audio_to_tokens(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Encode audio to speech tokens via the S3Tokenizer ONNX model.

        Resamples to 16kHz first (S3Tokenizer expects 16kHz input),
        then runs the ONNX model to get discrete tokens.
        """
        import torchaudio
        import whisper

        # Ensure 1D
        if audio.ndim > 1:
            audio = audio.squeeze()

        # Resample 24kHz -> 16kHz for the tokenizer
        if sample_rate != self.INPUT_SR:
            audio_cpu = audio.cpu()
            resampler = torchaudio.transforms.Resample(sample_rate, self.INPUT_SR)
            audio_16k = resampler(audio_cpu.unsqueeze(0)).squeeze(0)
        else:
            audio_16k = audio.cpu() if audio.is_cuda else audio

        # Lazy-load ONNX session if not already loaded
        if self._onnx_session is None:
            from modelscope import snapshot_download
            model_dir = Path(
                self.checkpoint
                if Path(self.checkpoint).exists()
                else snapshot_download(self.checkpoint)
            )
            self._load_onnx_tokenizer(model_dir)

        if self._onnx_session is None:
            raise RuntimeError("ONNX S3Tokenizer could not be loaded")

        audio_np = audio_16k.numpy()

        # Process in 30-second chunks (S3Tokenizer limit)
        all_tokens = []
        chunk_samples = 30 * self.INPUT_SR
        for i in range(0, len(audio_np), chunk_samples):
            chunk = audio_np[i : i + chunk_samples]

            # Minimum length for whisper mel spectrogram
            if len(chunk) < 400:
                chunk = np.pad(chunk, (0, 400 - len(chunk)))

            mel = whisper.log_mel_spectrogram(
                torch.from_numpy(chunk).float(), n_mels=128
            )
            if mel.ndim == 2:
                mel = mel.unsqueeze(0)

            outputs = self._onnx_session.run(
                None,
                {
                    self._onnx_session.get_inputs()[0].name: mel.numpy(),
                    self._onnx_session.get_inputs()[1].name: np.array(
                        [mel.shape[2]], dtype=np.int32
                    ),
                },
            )
            all_tokens.extend(outputs[0].flatten().tolist())

        return torch.tensor(all_tokens, dtype=torch.long)

    def _prepare_reference(
        self,
        reference_audio: Optional[torch.Tensor],
        reference_audio_sr: Optional[int],
    ) -> torch.Tensor:
        """Resample reference audio to 16kHz mono tensor."""
        if reference_audio is None:
            raise ValueError(
                "reference_audio is required for cross-lingual/zero-shot mode. "
                "Provide a 3-10 second clip of the target speaker."
            )

        import torchaudio

        audio = reference_audio
        if audio.ndim > 1:
            audio = audio.mean(dim=0) if audio.shape[0] <= 8 else audio.mean(dim=-1)

        if reference_audio_sr and reference_audio_sr != self.INPUT_SR:
            resampler = torchaudio.transforms.Resample(reference_audio_sr, self.INPUT_SR)
            audio = resampler(audio.unsqueeze(0)).squeeze(0)

        return audio

    @property
    def codebook_size(self) -> int:
        return self.CODEBOOK_SIZE

    @property
    def token_rate_hz(self) -> float:
        return self.TOKEN_RATE_HZ
