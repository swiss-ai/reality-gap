#!/usr/bin/env python3
"""Synthesize Polish audio via a chosen TTS backend and write Lhotse Shar.

This is the **bridge script** in the synthetic-Polish pipeline:

    JSON test-set (id, text) → TTS backend → audio + cuts → Lhotse Shar
       → audio_tokenization/tokenize.py → parquets
       → build_interleaved_indexed.py  → Megatron .bin/.idx

The downstream steps (parquet + interleave) are existing scripts; this script
fills the gap by producing Lhotse Shar that those tools can consume.

Input formats:
  - JSON list (same schema as data/tts_bench/pl_50.json):
      [{"id": "...", "text": "...", ["category": "...", "source_id": "..."]}, ...]
  - Lhotse Shar dir (transcripts only — synthesized audio replaces the real)

Usage:
    python scripts/synthesize_to_shar.py \\
        --backend voxcpm2 \\
        --input data/tts_bench/pl_voxpopuli.json \\
        --output-shar results/synthetic_shar/voxcpm2/pl_voxpopuli \\
        --reference-audio outputs/reference_audio/pl_speaker_0.wav

    # Reference text (for backends that need ref_text like OmniVoice):
    python scripts/synthesize_to_shar.py ... \\
        --reference-text "tekst referencyjny"

Output:
    <output-shar>/
      shar_index.json
      worker_00/cuts.000000.jsonl.gz
      worker_00/recording.000000.tar
      ...
"""

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
import torchaudio

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Backend loader (mirrors benchmark_tts.py BACKENDS) ───────────────
def load_backend(name: str, device: str, checkpoint: Optional[str],
                 vllm_endpoint: Optional[str] = None):
    """Instantiate one of the commercially-licensed TTS backends."""
    if name == "voxcpm2":
        from speech_generation.backends.voxcpm2_tts import VoxCPM2TTSBackend
        b = VoxCPM2TTSBackend(checkpoint=checkpoint or "openbmb/VoxCPM2", device=device)
    elif name == "voxcpm2_vllm":
        from speech_generation.backends.voxcpm2_vllm_tts import VoxCPM2VLLMTTSBackend
        if not vllm_endpoint:
            raise ValueError("voxcpm2_vllm backend requires --vllm-endpoint")
        b = VoxCPM2VLLMTTSBackend(endpoint=vllm_endpoint,
                                  model=checkpoint or "openbmb/VoxCPM2")
    elif name == "piper":
        from speech_generation.backends.piper_tts import PiperTTSBackend
        b = PiperTTSBackend(voice_path=checkpoint or "voices/pl_PL-gosia-medium.onnx", device=device)
    elif name == "parler":
        from speech_generation.backends.parler_tts import ParlerTTSBackend
        b = ParlerTTSBackend(
            checkpoint=checkpoint or "parler-tts/parler-tts-mini-multilingual-v1.1",
            device=device,
        )
    elif name == "f5":
        from speech_generation.backends.f5_tts import F5TTSBackend
        b = F5TTSBackend(checkpoint=checkpoint or "Sticzu/marek-f5tts-polish", device=device)
    elif name == "cosyvoice2":
        from speech_generation.backends.cosyvoice2_tts import CosyVoice2TTSBackend
        b = CosyVoice2TTSBackend(checkpoint=checkpoint or "iic/CosyVoice2-0.5B", device=device)
    else:
        raise ValueError(f"Unsupported backend: {name}")
    b.load_model()
    return b


# ── Input loader ─────────────────────────────────────────────────────
def load_input(path: Path) -> list[dict]:
    """JSON list of {id, text, ...}. Other input formats can be added later."""
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(items).__name__}")
    return items


def load_reference(ref_audio_path: Optional[str]) -> tuple[Optional[torch.Tensor], Optional[int]]:
    if not ref_audio_path:
        return None, None
    wav, sr = torchaudio.load(ref_audio_path)
    if wav.ndim > 1:
        wav = wav.mean(dim=0)
    return wav, sr


# ── Shar writing ─────────────────────────────────────────────────────
def _write_cut(writer, sid: str, text: str, audio, sr: int, target_sr: int):
    """Resample if needed, encode to WAV, build MonoCut, write to SharWriter."""
    from lhotse import MonoCut, Recording, SupervisionSegment

    if sr != target_sr:
        audio = torchaudio.transforms.Resample(sr, target_sr)(audio.unsqueeze(0)).squeeze(0)
        sr = target_sr

    duration = audio.numel() / sr
    buf = io.BytesIO()
    sf.write(buf, audio.detach().cpu().numpy(), sr,
             format="WAV", subtype="PCM_16")
    recording = Recording.from_bytes(data=buf.getvalue(), recording_id=sid)
    supervision = SupervisionSegment(
        id=sid, recording_id=sid, start=0.0, duration=duration,
        channel=0, text=text, language="pl",
    )
    cut = MonoCut(
        id=sid, start=0.0, duration=duration, channel=0,
        recording=recording, supervisions=[supervision],
    )
    writer.write(cut)
    return duration


def synthesize_and_write_shar(
    backend,
    items: list[dict],
    output_shar: Path,
    reference_audio: Optional[torch.Tensor],
    reference_audio_sr: Optional[int],
    reference_text: Optional[str],
    shard_size: int,
    target_sr: int,
    batch_size: int = 1,
) -> dict:
    """Synthesize each utterance and emit a Lhotse Shar dataset.

    When batch_size > 1, items are chunked and dispatched via
    backend.generate_batch() — this is the path that actually leverages
    vllm-omni's continuous batching. batch_size=1 keeps the per-utt loop
    (Direct-Python and other non-batched backends).

    Each entry becomes a Cut with one recording + one supervision (text).
    Cuts are streamed into a SharWriter so memory stays bounded for large runs.
    """
    from lhotse.shar import SharWriter

    output_shar.mkdir(parents=True, exist_ok=True)

    stats = {
        "n_input": len(items),
        "n_succeeded": 0,
        "n_failed": 0,
        "total_audio_seconds": 0.0,
        "total_generation_seconds": 0.0,
        "batch_size": batch_size,
    }

    with SharWriter(
        str(output_shar),
        fields={"recording": "wav"},
        shard_size=shard_size,
    ) as writer:
        if batch_size <= 1:
            # Sequential per-utt path — used for Direct-Python backends or
            # when explicitly running batch_size=1 for comparison.
            for i, item in enumerate(items):
                sid = item["id"]
                text = item["text"]
                t0 = time.time()
                try:
                    out = backend.generate(
                        text=text,
                        reference_audio=reference_audio,
                        reference_audio_sr=reference_audio_sr,
                        render_audio=True,
                        ref_text=reference_text,
                    )
                    elapsed = time.time() - t0
                    if out.audio is None:
                        raise RuntimeError("Backend returned no audio.")
                    duration = _write_cut(
                        writer, sid, text, out.audio,
                        out.audio_sample_rate or target_sr, target_sr,
                    )
                    stats["n_succeeded"] += 1
                    stats["total_audio_seconds"] += duration
                    stats["total_generation_seconds"] += elapsed
                    if (i + 1) % 25 == 0 or i == 0:
                        logger.info(
                            "[%d/%d] %s ok (%.2fs gen, %.2fs audio, rtf=%.3f)",
                            i + 1, len(items), sid, elapsed, duration,
                            elapsed / max(duration, 1e-6),
                        )
                except Exception as e:
                    stats["n_failed"] += 1
                    logger.error("[%d/%d] %s FAILED: %s",
                                 i + 1, len(items), sid, e, exc_info=False)
            return stats

        # Batched path — chunk items, dispatch each chunk in one request.
        for batch_start in range(0, len(items), batch_size):
            batch = items[batch_start:batch_start + batch_size]
            texts = [it["text"] for it in batch]
            t0 = time.time()
            try:
                outs = backend.generate_batch(
                    texts=texts,
                    reference_audio=reference_audio,
                    reference_audio_sr=reference_audio_sr,
                    render_audio=True,
                )
                batch_elapsed = time.time() - t0
                batch_audio_s = 0.0
                # Write each cut. Per-utt elapsed is amortized batch_elapsed/N
                # for stats purposes; the real metric is batch RTF which we
                # log per-batch below.
                for it, out in zip(batch, outs):
                    sid = it["id"]
                    text = it["text"]
                    if out.audio is None:
                        stats["n_failed"] += 1
                        logger.error("[batch %d-%d] %s: no audio",
                                     batch_start, batch_start + len(batch), sid)
                        continue
                    duration = _write_cut(
                        writer, sid, text, out.audio,
                        out.audio_sample_rate or target_sr, target_sr,
                    )
                    stats["n_succeeded"] += 1
                    batch_audio_s += duration
                stats["total_audio_seconds"] += batch_audio_s
                stats["total_generation_seconds"] += batch_elapsed
                logger.info(
                    "[batch %d-%d/%d] %d ok in %.2fs gen, %.2fs audio, batch_rtf=%.3f",
                    batch_start, batch_start + len(batch), len(items),
                    len(outs), batch_elapsed, batch_audio_s,
                    batch_elapsed / max(batch_audio_s, 1e-6),
                )
            except Exception as e:
                # Failure of a whole batch — count all as failed, log loudly,
                # and continue with the next batch (don't crash the run).
                stats["n_failed"] += len(batch)
                logger.error(
                    "[batch %d-%d/%d] FAILED: %s",
                    batch_start, batch_start + len(batch), len(items), e,
                    exc_info=False,
                )

    return stats


# ── CLI ──────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", required=True,
                   choices=["voxcpm2", "voxcpm2_vllm", "piper", "parler", "f5", "cosyvoice2"])
    p.add_argument("--vllm-endpoint", default=None,
                   help="Base URL of the running vllm-omni server. Required for "
                        "--backend=voxcpm2_vllm. e.g. http://nid001234:8000")
    p.add_argument("--input", required=True, type=Path, help="JSON manifest (id, text)")
    p.add_argument("--output-shar", required=True, type=Path, help="Output Lhotse Shar directory")
    p.add_argument("--checkpoint", default=None, help="Backend-specific checkpoint override")
    p.add_argument("--reference-audio", default=None, help="Path to reference WAV for voice cloning")
    p.add_argument("--reference-text", default=None, help="Reference transcript (for OmniVoice/F5)")
    p.add_argument("--shard-size", type=int, default=1000, help="Cuts per shard in the output Shar")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Utterances per backend call. Values >1 use backend.generate_batch() "
                        "instead of per-utt generate() — the path that unlocks vllm-omni's "
                        "continuous batching via /v1/audio/speech/batch. For Direct-Python "
                        "backends generate_batch() falls back to a sequential loop (no win).")
    p.add_argument("--target-sr", type=int, default=24000,
                   help="Resample to this rate before writing Shar. Default 24000 matches "
                        "WavTokenizer's expected input; the interleave config's 16000 floor "
                        "is a filter, not a target.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--shard-idx", type=int, default=0,
                   help="0-indexed shard this task processes. Combined with --num-shards lets "
                        "a SLURM array distribute the input round-robin across tasks.")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Total number of shards. Each task processes items[shard_idx::num_shards] "
                        "(round-robin, balances variable utterance lengths across tasks). "
                        "When >1, output is written under <output-shar>/shard_{idx:04d}/.")
    args = p.parse_args()

    if not args.input.exists():
        logger.error("Input not found: %s", args.input)
        sys.exit(2)
    if args.num_shards < 1 or not (0 <= args.shard_idx < args.num_shards):
        logger.error("Invalid sharding: shard-idx=%d num-shards=%d", args.shard_idx, args.num_shards)
        sys.exit(2)

    logger.info("Reading input: %s", args.input)
    items = load_input(args.input)
    total = len(items)
    if args.num_shards > 1:
        items = items[args.shard_idx::args.num_shards]
        # Each shard writes to its own subdir so SLURM array tasks don't race
        # on the same SharWriter. Combine post-hoc by enumerating shard_*/.
        shard_out = args.output_shar / f"shard_{args.shard_idx:04d}"
        logger.info("  shard %d/%d: %d of %d utterances", args.shard_idx, args.num_shards, len(items), total)
    else:
        shard_out = args.output_shar
        logger.info("  %d utterances to synthesize", total)

    logger.info("Loading backend: %s", args.backend)
    backend = load_backend(args.backend, args.device, args.checkpoint,
                           vllm_endpoint=args.vllm_endpoint)

    logger.info("Loading reference audio: %s", args.reference_audio)
    ref_audio, ref_sr = load_reference(args.reference_audio)

    t_start = time.time()
    stats = synthesize_and_write_shar(
        backend=backend,
        items=items,
        output_shar=shard_out,
        reference_audio=ref_audio,
        reference_audio_sr=ref_sr,
        reference_text=args.reference_text,
        shard_size=args.shard_size,
        target_sr=args.target_sr,
        batch_size=args.batch_size,
    )
    stats["wall_seconds"] = round(time.time() - t_start, 2)
    stats["shard_idx"] = args.shard_idx
    stats["num_shards"] = args.num_shards
    if stats["total_audio_seconds"] > 0:
        stats["aggregate_rtf"] = round(
            stats["total_generation_seconds"] / stats["total_audio_seconds"], 4
        )

    (shard_out / "synthesis_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    print(json.dumps(stats, indent=2))
    if stats["n_failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
