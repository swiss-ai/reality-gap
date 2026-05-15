"""VoxCPM2 backend that talks to a vllm-omni serving endpoint.

Stage-3 batched path. Counterpart to the Direct-Python backend in
voxcpm2_tts.py — same model, different inference mechanism.

The vllm-omni server is launched separately via
`scripts/launch_vllm_voxcpm2.slurm`. This backend POSTs to that
server's endpoint and reconstructs the audio from the response.

API surface (confirmed against /openapi.json on 2026-05-15):

  POST /v1/audio/voices            multipart/form-data
      audio_sample (file)
      name (string)
      consent (string)
      ref_text (string, optional)
  GET  /v1/audio/voices
  POST /v1/audio/speech            application/json
      {model, input, voice, instructions, ...}
      returns audio/wav bytes (default) or audio/* depending on Accept
  POST /v1/audio/speech/batch      application/json
      {model, items: [SpeechBatchItem], voice, ...}
      returns JSON

Reference audio is uploaded once via POST /v1/audio/voices and reused by
`name` in all subsequent /v1/audio/speech calls. We cache the voice name
by id() of the reference tensor — same pattern as the Direct-Python
backend's tmp-file cache.
"""

import base64
import hashlib
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
            "http://172.28.26.8:8000".
        model: Model name to send in the request. Must match the
            --model arg the server was launched with.
        timeout_s: Per-request HTTP timeout. Server-side warmup +
            sequence packing means cold starts can take ~30 s.
        consent: Consent string passed when uploading reference voices.
            The server requires it but accepts any non-empty string for
            user-provided clips.
    """

    def __init__(
        self,
        endpoint: str,
        model: str = "openbmb/VoxCPM2",
        timeout_s: float = 60.0,
        consent: str = "user-provided reference audio",
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.consent = consent
        self._session = None
        self._sample_rate: Optional[int] = None
        # Cache registered-voice names by id(reference_audio) so we don't
        # re-upload the same reference WAV for every call.
        self._voice_cache: dict[int, str] = {}

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

        # VoxCPM2 native is 48 kHz per the model card. The bridge script
        # resamples to --target-sr (default 24 k) downstream so this only
        # matters for accurate stats reporting on the client side.
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

        # Upload reference voice once (cached by tensor identity) and use
        # its registered name in the speech request.
        voice_name = self._ensure_voice_uploaded(reference_audio, reference_audio_sr)
        audio_np, sr = self._post_speech(text, voice_name)

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
                "voice": voice_name,
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
        """True batched generation via /v1/audio/speech/batch.

        BatchSpeechRequest has `items: list[SpeechBatchItem]` and a shared
        `voice`. Per-item overrides are documented; we keep it simple and
        send N items with one shared voice.
        """
        voice_name = self._ensure_voice_uploaded(reference_audio, reference_audio_sr)
        results = self._post_speech_batch(texts, voice_name)

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
                    "voice": voice_name,
                    "batched": True,
                    "direct_tokens": False,
                },
            ))
        return outputs

    # ── HTTP plumbing ─────────────────────────────────────────────────

    def _ensure_voice_uploaded(
        self,
        reference_audio: Optional[torch.Tensor],
        sr: Optional[int],
    ) -> Optional[str]:
        """Register the reference voice on first sight, return its name.

        Cached by tensor identity — synthesize_to_shar loads the reference
        once outside the per-utt loop, so id(tensor) is stable across calls.
        """
        if reference_audio is None:
            return None
        key = id(reference_audio)
        cached = self._voice_cache.get(key)
        if cached is not None:
            return cached

        import soundfile as sf

        ra = reference_audio
        if ra.ndim > 1:
            ra = ra.mean(dim=0)
        buf = io.BytesIO()
        sf.write(buf, ra.detach().cpu().numpy(), sr or 16000,
                 format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()

        # Stable name from the audio content — re-running the synth job with
        # the same reference re-uses the same voice rather than littering the
        # server with one-shot uploads.
        voice_name = "voice_" + hashlib.sha1(wav_bytes).hexdigest()[:12]

        # multipart/form-data per the openapi schema. `name` and `consent`
        # are required; `ref_text` is optional and improves quality in ICL
        # mode but we don't have transcripts of the reference WAV.
        files = {"audio_sample": (f"{voice_name}.wav", wav_bytes, "audio/wav")}
        data = {"name": voice_name, "consent": self.consent}

        logger.info("Uploading reference voice as %s (%d bytes)",
                    voice_name, len(wav_bytes))
        r = self._session.post(
            f"{self.endpoint}/v1/audio/voices",
            files=files, data=data, timeout=self.timeout_s,
        )
        # If the server already knows this voice (we picked a stable hash),
        # it may return 400 "already exists" — that's fine, name still works.
        if r.status_code >= 400 and "already" in r.text.lower():
            logger.info("Voice %s already registered server-side", voice_name)
        else:
            r.raise_for_status()

        self._voice_cache[key] = voice_name
        return voice_name

    def _post_speech(self, text: str, voice_name: Optional[str]) -> tuple:
        """POST /v1/audio/speech. Returns (audio_np float32, sample_rate int).

        Per the OpenAICreateSpeechRequest schema: required `input`; optional
        `model`, `voice`. Response is audio bytes by default (audio/*).
        """
        import soundfile as sf

        payload = {"model": self.model, "input": text}
        if voice_name:
            payload["voice"] = voice_name

        r = self._session.post(
            f"{self.endpoint}/v1/audio/speech",
            json=payload, timeout=self.timeout_s,
        )
        r.raise_for_status()

        # Per /openapi.json, both audio/* and application/json are possible.
        # Server's default is audio bytes; handle both shapes.
        ct = r.headers.get("content-type", "")
        if ct.startswith("audio/"):
            buf = io.BytesIO(r.content)
        else:
            # JSON wrapper — likely base64-encoded audio field
            data = r.json()
            audio_b64 = data.get("audio") or (data.get("data") or [{}])[0].get("audio")
            if audio_b64 is None:
                raise RuntimeError(f"unexpected JSON response (no audio field): {data!r}")
            buf = io.BytesIO(base64.b64decode(audio_b64))

        audio_np, sr = sf.read(buf, dtype="float32")
        return audio_np, sr

    def _post_speech_batch(
        self, texts: list[str], voice_name: Optional[str]
    ) -> list[tuple]:
        """POST /v1/audio/speech/batch. Returns list of (audio_np, sr).

        BatchSpeechRequest: {model, items: [SpeechBatchItem], voice, ...}
        SpeechBatchItem shape is best-guess `{input: str}` until verified —
        if the server 422s, the error response will name the missing field.
        """
        import soundfile as sf

        payload = {
            "model": self.model,
            "items": [{"input": t} for t in texts],
        }
        if voice_name:
            payload["voice"] = voice_name

        r = self._session.post(
            f"{self.endpoint}/v1/audio/speech/batch",
            json=payload, timeout=self.timeout_s * max(len(texts), 1),
        )
        r.raise_for_status()
        data = r.json()

        # Confirmed response shape: items wrapped in some top-level key.
        # Each item: {index, status, audio_data: <base64>, media_type, error}.
        # Key is `audio_data`, NOT `audio`.
        if isinstance(data, list):
            items_out = data
        else:
            items_out = (
                data.get("items")
                or data.get("results")
                or data.get("data")
                or data.get("outputs")
                or []
            )
            # Last-resort: any list of dicts at the top level.
            if not items_out:
                for v in data.values():
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        items_out = v
                        break
        if not items_out:
            raise RuntimeError(f"batched response had no items field: {data!r}")

        # Reorder by index — server may return out of order under batching.
        items_out = sorted(items_out, key=lambda it: it.get("index", 0))

        out = []
        for item in items_out:
            status = item.get("status")
            if status and status != "success":
                raise RuntimeError(
                    f"batched item {item.get('index')} failed: "
                    f"{item.get('error') or 'unknown error'}"
                )
            audio_b64 = item.get("audio_data") or item.get("audio")
            if audio_b64 is None:
                raise RuntimeError(f"batched item had no audio_data field: {item!r}")
            audio_np, sr = sf.read(io.BytesIO(base64.b64decode(audio_b64)),
                                    dtype="float32")
            out.append((audio_np, sr))
        return out

    @property
    def codebook_size(self) -> int:
        return 0

    @property
    def token_rate_hz(self) -> float:
        return 0.0
