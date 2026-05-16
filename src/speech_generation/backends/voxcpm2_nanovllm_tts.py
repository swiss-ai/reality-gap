"""VoxCPM2 via nano-vllm-voxcpm (in-process continuous batching).

Counterpart to voxcpm2_tts.py (Direct-Python) and voxcpm2_vllm_tts.py
(HTTP/vllm-omni). nano-vllm's VoxCPM2Engine genuinely fuses concurrent
in-flight generations into the same forward pass for this codec — pl_50
smoke at concurrency=32 hit wall_RTF 0.018 (~10x over Direct-Python's
0.185). vllm-omni couldn't.

Container: /capstor/store/cscs/swissai/infra01/container-images/nanovllm-voxcpm2-cuda13.sqsh
SLURM env: nanovllm-voxcpm2-colocated.toml

Concurrency model:
  - VoxCPM.from_pretrained() returns AsyncVoxCPM2ServerPool when called
    inside an asyncio loop. We hold one loop + pool for the backend's life.
  - generate_batch() dispatches all texts via asyncio.gather() — the pool's
    engine batches them with max_num_seqs=512.
  - For real throughput, call generate_batch with batch_size matching the
    sweet spot found in the pl_50 smoke (~32).

Voice cloning:
  - One pool.add_prompt(wav_bytes, "wav", ref_transcript) at first call;
    prompt_id is cached by (id(reference_audio), reference_text) so the
    same ref tensor + transcript re-uses the encoded latents.
  - Reference TRANSCRIPT is REQUIRED — that's what nano-vllm uses for
    in-context-learning of the voice. Pass via `reference_text=` /
    `ref_text=` kwarg on generate / generate_batch.
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Optional

import numpy as np
import torch

from ..base import TTSBackend, TTSOutput

logger = logging.getLogger(__name__)


class VoxCPM2NanoVLLMTTSBackend(TTSBackend):
    """VoxCPM2 served via nano-vllm-voxcpm in-process pool."""

    def __init__(
        self,
        model_path: str = "/capstor/store/cscs/swissai/infra01/hf_models/models/openbmb/VoxCPM2",
        device: str = "cuda",
        gpu_memory_utilization: float = 0.9,
        max_num_seqs: int = 512,
        max_num_batched_tokens: int = 16384,
        max_model_len: int = 4096,
        inference_timesteps: int = 10,
    ):
        self.model_path = model_path
        self.device = device
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_model_len = max_model_len
        self.inference_timesteps = inference_timesteps

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._pool = None
        self._sample_rate: Optional[int] = None
        # Cache prompt_id by (id(ref_tensor), ref_text). The bridge script
        # loads the reference once and reuses the same tensor across calls,
        # so id() is stable.
        self._prompt_cache: dict[tuple[int, str], str] = {}

    def load_model(self, device: Optional[str] = None) -> None:
        from nanovllm_voxcpm import VoxCPM

        if device:
            self.device = device

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def _init():
            # from_pretrained checks for a running loop and returns
            # AsyncVoxCPM2ServerPool when one is found — which is the case
            # here because we're awaiting inside run_until_complete.
            pool = VoxCPM.from_pretrained(
                self.model_path,
                devices=[0],
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_num_seqs=self.max_num_seqs,
                max_num_batched_tokens=self.max_num_batched_tokens,
                max_model_len=self.max_model_len,
                inference_timesteps=self.inference_timesteps,
            )
            await pool.wait_for_ready()
            info = await pool.get_model_info()
            return pool, info

        self._pool, info = self._loop.run_until_complete(_init())
        self._sample_rate = int(info["output_sample_rate"])
        logger.info(
            "nano-vllm-voxcpm ready: %s (sr=%d, max_num_seqs=%d, max_batched_tokens=%d)",
            self.model_path, self._sample_rate, self.max_num_seqs, self.max_num_batched_tokens,
        )

    def _ensure_prompt(
        self,
        reference_audio: Optional[torch.Tensor],
        reference_audio_sr: Optional[int],
        reference_text: Optional[str],
    ) -> str:
        if reference_audio is None or not reference_text:
            raise ValueError(
                "voxcpm2_nanovllm requires reference_audio AND reference_text "
                "(the transcript of the reference clip — needed for ICL voice cloning)."
            )
        key = (id(reference_audio), reference_text)
        cached = self._prompt_cache.get(key)
        if cached is not None:
            return cached

        import soundfile as sf

        ra = reference_audio
        if ra.ndim > 1:
            ra = ra.mean(dim=0)
        buf = io.BytesIO()
        sf.write(buf, ra.detach().cpu().numpy(), reference_audio_sr or 16000,
                 format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()

        async def _add():
            return await self._pool.add_prompt(wav_bytes, "wav", reference_text)

        prompt_id = self._loop.run_until_complete(_add())
        self._prompt_cache[key] = prompt_id
        logger.info("Registered reference prompt: id=%s (text=%r…)",
                    prompt_id, reference_text[:60])
        return prompt_id

    def generate(
        self,
        text: str,
        reference_audio: Optional[torch.Tensor] = None,
        reference_audio_sr: Optional[int] = None,
        speaker_id: Optional[str] = None,
        render_audio: bool = False,
        **kwargs,
    ) -> TTSOutput:
        # synthesize_to_shar's sequential branch passes the ref transcript as
        # `ref_text=`; benchmark_tts uses `reference_text=`. Accept both.
        ref_text = kwargs.get("ref_text") or kwargs.get("reference_text")
        outs = self.generate_batch(
            texts=[text],
            reference_audio=reference_audio,
            reference_audio_sr=reference_audio_sr,
            render_audio=render_audio,
            ref_text=ref_text,
        )
        return outs[0]

    def generate_batch(
        self,
        texts: list[str],
        reference_audio: Optional[torch.Tensor] = None,
        reference_audio_sr: Optional[int] = None,
        render_audio: bool = False,
        **kwargs,
    ) -> list[TTSOutput]:
        if self._pool is None or self._loop is None:
            raise RuntimeError("Backend not loaded — call load_model() first.")

        ref_text = kwargs.get("ref_text") or kwargs.get("reference_text")
        prompt_id = self._ensure_prompt(reference_audio, reference_audio_sr, ref_text)

        async def _gen_one(t: str) -> Optional[np.ndarray]:
            chunks: list[np.ndarray] = []
            try:
                async for chunk in self._pool.generate(t, prompt_id=prompt_id):
                    chunks.append(chunk)
            except Exception:
                logger.exception("nano-vllm generate failed for text=%r", t[:80])
                return None
            if not chunks:
                return None
            return np.concatenate(chunks).astype(np.float32, copy=False)

        async def _gen_all():
            return await asyncio.gather(*[_gen_one(t) for t in texts])

        audios = self._loop.run_until_complete(_gen_all())

        outputs: list[TTSOutput] = []
        sr = self._sample_rate
        for wav in audios:
            if wav is None or wav.size == 0:
                outputs.append(TTSOutput(
                    speech_tokens=torch.empty(0, dtype=torch.long),
                    codebook_size=0, token_rate_hz=0.0,
                    audio=None, audio_sample_rate=None,
                    metadata={"backend": "voxcpm2_nanovllm", "failed": True},
                ))
                continue
            audio = torch.from_numpy(wav)
            outputs.append(TTSOutput(
                speech_tokens=torch.empty(0, dtype=torch.long),
                codebook_size=0, token_rate_hz=0.0,
                audio=audio if render_audio else None,
                audio_sample_rate=sr if render_audio else None,
                metadata={
                    "backend": "voxcpm2_nanovllm",
                    "model_path": self.model_path,
                    "max_num_seqs": self.max_num_seqs,
                    "batched": len(texts) > 1,
                    "direct_tokens": False,
                },
            ))
        return outputs

    def __del__(self):
        # Best-effort teardown — server subprocess(es) get killed by stop().
        # During interpreter shutdown some attrs may already be gone, so guard.
        loop = getattr(self, "_loop", None)
        pool = getattr(self, "_pool", None)
        if loop is None or pool is None:
            return
        try:
            if not loop.is_closed():
                loop.run_until_complete(pool.stop())
                loop.close()
        except Exception:
            pass

    @property
    def codebook_size(self) -> int:
        return 0

    @property
    def token_rate_hz(self) -> float:
        return 0.0
