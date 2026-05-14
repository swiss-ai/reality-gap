"""VoxCPM2 backend that talks to a vllm-omni serving endpoint.

Stage-3 batched path. Counterpart to the Direct-Python backend in
voxcpm2_tts.py — same model, different inference mechanism.

The vllm-omni server is launched separately via
`scripts/launch_vllm_voxcpm2.slurm`. This backend POSTs to that
server's endpoint and reconstructs the audio from the response.

UNVERIFIED API — the docs page documenting VoxCPM2's request format
in vllm-omni was 404 at research time. The HTTP shape below is a
best-guess against vLLM's OpenAI-compat audio API conventions. When
the launcher proves the actual API on first run, update _post_audio()
to match.
"""

import base64
import io
import logging
import time
from pathlib import Path
from typing import Optional

import torch

from ..base import TTSBackend, TTSOutput

logger = logging.getLogger(__name__)


class VoxCPM2VLLMTTSBackend(TTSBackend):
    """VoxCPM2 via vllm-omni HTTP server.

    Args:
        endpoint: Base URL of the running vllm-omni server, e.g.
            "http://nid001234:8000". Set via VLLM_ENDPOINT env var or
            passed explicitly.
        model: Model name to send in the request. Must match the
            --model arg the server was launched with.
        timeout_s: Per-request HTTP timeout. Server-side warmup +
            sequence packing means cold starts can take ~30 s.
    """

    def __init__(
        self,
        endpoint: str,
        model: str = "openbmb/VoxCPM2",
        timeout_s: float = 60.0,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self._session = None
        self._sample_rate: Optional[int] = None
        # Cache base64-encoded reference audio so we don't re-encode on every
        # request; analogous to the id()-based cache in voxcpm2_tts.py.
        self._ref_cache: dict[int, str] = {}

    def load_model(self, device: Optional[str] = None) -> None:
        # "Loading" the model is really just a health check against the
        # remote server. The server holds the GPU; this client is CPU-only.
        import requests

        self._session = requests.Session()
        try:
            r = self._session.get(f"{self.endpoint}/health", timeout=10)
            r.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"vllm-omni server not reachable at {self.endpoint}. "
                f"Did you sbatch scripts/launch_vllm_voxcpm2.slurm and wait "
                f"for the endpoint URL in the log? Underlying error: {e}"
            ) from e

        # VoxCPM2 native is 48 kHz; the server should expose this via /v1/models
        # or similar. Until we verify, assume 48 k. The bridge script
        # resamples to --target-sr (default 24 k) downstream so this only
        # matters for accurate stats reporting.
        self._sample_rate = 48000
        logger.info("VoxCPM2-vLLM backend ready at %s (sr=%d)",
                    self.endpoint, self._sample_rate)

    def generate(
        self,
        text: str,
        reference_audio: Optional[torch.Tensor] = None,
        reference_audio_sr: Optional[int] = None,
        speaker_id: Optional[str] = None,
        render_audio: bool = False,
        **kwargs,
    ) -> TTSOutput:
        if self._session is None:
            raise RuntimeError("load_model() not called. Run it before generate().")

        ref_b64 = self._encode_reference(reference_audio, reference_audio_sr)
        audio_np, sr = self._post_audio(text, ref_b64)

        audio = torch.from_numpy(audio_np).to(torch.float32)
        if audio.ndim > 1:
            audio = audio.squeeze()

        return TTSOutput(
            speech_tokens=torch.empty(0, dtype=torch.long),
            codebook_size=0,
            token_rate_hz=0.0,
            audio=audio if render_audio else None,
            audio_sample_rate=sr if render_audio else None,
            metadata={
                "backend": "voxcpm2_vllm",
                "endpoint": self.endpoint,
                "model": self.model,
                "voice_cloning": reference_audio is not None,
                "direct_tokens": False,
            },
        )

    def generate_batch(
        self,
        texts: list[str],
        reference_audio: Optional[torch.Tensor] = None,
        reference_audio_sr: Optional[int] = None,
        render_audio: bool = False,
        **kwargs,
    ) -> list[TTSOutput]:
        """True batched generation — the whole point of this backend.

        Posts a single request with N texts. The vllm-omni server batches
        them internally (continuous batching, CUDA graph, shared mem pool).

        UNVERIFIED: the batched-request schema may not be `input: list[str]`.
        Update _post_audio_batch() once the real API is confirmed.
        """
        ref_b64 = self._encode_reference(reference_audio, reference_audio_sr)
        results = self._post_audio_batch(texts, ref_b64)

        outputs = []
        for audio_np, sr in results:
            audio = torch.from_numpy(audio_np).to(torch.float32)
            if audio.ndim > 1:
                audio = audio.squeeze()
            outputs.append(TTSOutput(
                speech_tokens=torch.empty(0, dtype=torch.long),
                codebook_size=0,
                token_rate_hz=0.0,
                audio=audio if render_audio else None,
                audio_sample_rate=sr if render_audio else None,
                metadata={
                    "backend": "voxcpm2_vllm",
                    "endpoint": self.endpoint,
                    "model": self.model,
                    "batched": True,
                    "direct_tokens": False,
                },
            ))
        return outputs

    # ── HTTP plumbing ─────────────────────────────────────────────────
    # The two _post_* methods are the ONLY places the actual vllm-omni
    # API shape lives. When we confirm the real schema on first run,
    # edit here.

    def _post_audio(self, text: str, ref_b64: Optional[str]) -> tuple:
        """POST a single text → return (audio_np, sample_rate).

        ASSUMED SCHEMA (vLLM OpenAI-compat-style):
            POST {endpoint}/v1/audio/generations
            {
              "model": "...",
              "input": "...",
              "voice": "<base64 wav>",   # optional
              "response_format": "wav"
            }
            response: WAV bytes (audio/wav content type), OR
                      JSON with base64-encoded "audio" field.
        """
        import numpy as np
        import soundfile as sf

        payload = {
            "model": self.model,
            "input": text,
            "response_format": "wav",
        }
        if ref_b64:
            payload["voice"] = ref_b64

        r = self._session.post(
            f"{self.endpoint}/v1/audio/generations",
            json=payload,
            timeout=self.timeout_s,
        )
        r.raise_for_status()

        # Two response styles to handle until verified:
        ct = r.headers.get("content-type", "")
        if "audio/wav" in ct or "audio/x-wav" in ct:
            buf = io.BytesIO(r.content)
        else:
            # JSON with base64 audio
            data = r.json()
            audio_b64 = data.get("audio") or data["data"][0]["audio"]
            buf = io.BytesIO(base64.b64decode(audio_b64))

        audio_np, sr = sf.read(buf, dtype="float32")
        return audio_np, sr

    def _post_audio_batch(self, texts: list[str], ref_b64: Optional[str]) -> list[tuple]:
        """POST a batch of texts → list of (audio_np, sample_rate).

        Falls back to sequential _post_audio if the server rejects the
        batched schema. Once we verify a batched endpoint exists, drop
        the fallback.
        """
        try:
            return self._post_audio_batch_native(texts, ref_b64)
        except Exception as e:
            logger.warning("Batched endpoint failed (%s); falling back to sequential", e)
            return [self._post_audio(t, ref_b64) for t in texts]

    def _post_audio_batch_native(self, texts: list[str], ref_b64: Optional[str]) -> list[tuple]:
        import base64 as b64m
        import soundfile as sf

        payload = {
            "model": self.model,
            "input": texts,             # ASSUMED: list[str] for batched
            "response_format": "wav",
        }
        if ref_b64:
            payload["voice"] = ref_b64

        r = self._session.post(
            f"{self.endpoint}/v1/audio/generations",
            json=payload,
            timeout=self.timeout_s * max(len(texts), 1),
        )
        r.raise_for_status()
        data = r.json()
        out = []
        for item in data["data"]:
            audio_bytes = b64m.b64decode(item["audio"])
            audio_np, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            out.append((audio_np, sr))
        return out

    def _encode_reference(
        self,
        reference_audio: Optional[torch.Tensor],
        sr: Optional[int],
    ) -> Optional[str]:
        if reference_audio is None:
            return None
        key = id(reference_audio)
        cached = self._ref_cache.get(key)
        if cached is not None:
            return cached

        import soundfile as sf

        if reference_audio.ndim > 1:
            reference_audio = reference_audio.mean(dim=0)
        buf = io.BytesIO()
        sf.write(buf, reference_audio.detach().cpu().numpy(), sr or 16000,
                 format="WAV", subtype="PCM_16")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        self._ref_cache[key] = encoded
        return encoded

    @property
    def codebook_size(self) -> int:
        return 0

    @property
    def token_rate_hz(self) -> float:
        return 0.0
