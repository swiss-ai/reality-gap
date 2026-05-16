"""nano-vllm-voxcpm concurrency smoke: does it actually batch?

The question this answers:
  vLLM-omni on VoxCPM2 didn't parallelize (per prior testing — concurrent
  requests serialized on the GPU). nano-vllm-voxcpm exposes the same
  VoxCPM2 model through a different engine (VoxCPM2Engine, max_num_seqs=512,
  max_num_batched_tokens=16384) that's specifically built for continuous
  batching. Does it actually fuse concurrent in-flight generations into
  the same forward pass for this model?

What it measures (on data/tts_bench/pl_50.json — 50 Polish utts):
  For each concurrency level in --batch-sizes, dispatches the full set via
  asyncio.gather() against AsyncVoxCPM2ServerPool. Reports aggregate
  RTF = total_gen_seconds / total_audio_seconds. Compare to the
  Direct-Python VoxCPM2 baseline RTF 0.185.

  - Aggregate RTF that DROPS as concurrency rises → engine is fusing, nano-vllm
    pays for itself.
  - Aggregate RTF FLAT across concurrency → same failure mode as vLLM-omni;
    skip nano-vllm and ship Direct-Python.

Also saves --keep-samples WAVs from the first batch-size run for native-ear
QA against the Direct-Python slow-ref baseline.

Run inside the nano-vllm container (workdir /opt, /users mounted via
nanovllm-voxcpm2-colocated.toml). Container ships nano-vllm-voxcpm in
/opt/venv — no separate venv build needed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("smoke_nanovllm")


async def gen_one(pool, text: str, prompt_id: str) -> tuple[np.ndarray | None, float]:
    """Run one generation; return (waveform, gen_seconds)."""
    t0 = time.perf_counter()
    chunks: list[np.ndarray] = []
    try:
        async for chunk in pool.generate(text, prompt_id=prompt_id):
            chunks.append(chunk)
    except Exception:
        logger.exception("generate() failed for text=%r", text[:60])
        return None, time.perf_counter() - t0
    wall = time.perf_counter() - t0
    if not chunks:
        return None, wall
    return np.concatenate(chunks).astype(np.float32, copy=False), wall


async def run_sweep(args: argparse.Namespace) -> dict:
    from nanovllm_voxcpm import VoxCPM

    # Load test set
    items = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if isinstance(items, dict) and "items" in items:
        items = items["items"]
    logger.info("Loaded %d items from %s", len(items), args.input)

    # Init pool. from_pretrained returns AsyncPool when called inside a loop.
    logger.info("Initializing nano-vllm pool: model=%s devices=[0]", args.model_path)
    pool = VoxCPM.from_pretrained(
        args.model_path,
        devices=[0],
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    await pool.wait_for_ready()

    info = await pool.get_model_info()
    sr = info["output_sample_rate"]
    logger.info("Model ready. output_sample_rate=%d  full info=%s", sr, info)

    # Register reference voice once — slow MLS-6892
    ref_wav = Path(args.ref_wav).read_bytes()
    ref_format = Path(args.ref_wav).suffix.lstrip(".") or "wav"
    ref_text = Path(args.ref_txt).read_text(encoding="utf-8").strip()
    logger.info("Registering ref voice: wav=%s (%d bytes, %s)  text=%r",
                args.ref_wav, len(ref_wav), ref_format, ref_text[:80])
    prompt_id = await pool.add_prompt(ref_wav, ref_format, ref_text)
    logger.info("Ref prompt_id=%s", prompt_id)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = out_dir / "samples"
    sample_dir.mkdir(exist_ok=True)

    results = []
    first_run = True
    for B in args.batch_sizes:
        logger.info("=== Concurrency = %d ===", B)
        t_run = time.perf_counter()
        total_audio_s = 0.0
        total_gen_s = 0.0
        n_ok = 0
        n_fail = 0
        # Process pl_50 in chunks of B; each chunk dispatched via gather.
        for i in range(0, len(items), B):
            chunk = items[i : i + B]
            coros = [gen_one(pool, it["text"], prompt_id) for it in chunk]
            outs = await asyncio.gather(*coros)
            for it, (wav, gen_s) in zip(chunk, outs):
                total_gen_s += gen_s
                if wav is None or wav.size == 0:
                    n_fail += 1
                    continue
                n_ok += 1
                total_audio_s += wav.size / sr
                if first_run and n_ok <= args.keep_samples:
                    out_path = sample_dir / f"{it['id']}.wav"
                    torchaudio.save(
                        str(out_path),
                        torch.from_numpy(wav).unsqueeze(0),
                        sr,
                    )
        wall_s = time.perf_counter() - t_run
        # Aggregate RTF: under perfect serial execution wall == total_gen_s,
        # so aggregate_rtf = total_gen_s / total_audio_s matches the
        # baseline 0.185. Under true batching, gen_s per request stays
        # similar but wall drops below sum(gen_s) — what we want to see is
        # wall_rtf = wall_s / total_audio_s, which IS the throughput RTF.
        per_req_rtf = total_gen_s / total_audio_s if total_audio_s else float("nan")
        wall_rtf = wall_s / total_audio_s if total_audio_s else float("nan")
        speedup_vs_per_req = total_gen_s / wall_s if wall_s else float("nan")
        logger.info(
            "B=%2d  ok=%d  fail=%d  wall=%.1fs  total_gen=%.1fs  audio=%.1fs  "
            "per_req_RTF=%.3f  wall_RTF=%.3f  batch_speedup=%.2fx",
            B, n_ok, n_fail, wall_s, total_gen_s, total_audio_s,
            per_req_rtf, wall_rtf, speedup_vs_per_req,
        )
        results.append({
            "batch_size": B,
            "n_ok": n_ok,
            "n_fail": n_fail,
            "wall_seconds": round(wall_s, 3),
            "total_gen_seconds": round(total_gen_s, 3),
            "total_audio_seconds": round(total_audio_s, 3),
            "per_request_rtf": round(per_req_rtf, 4),
            "wall_rtf": round(wall_rtf, 4),
            "batch_speedup": round(speedup_vs_per_req, 3),
        })
        first_run = False

    await pool.stop()

    stats = {
        "input": str(args.input),
        "model_path": args.model_path,
        "ref_wav": args.ref_wav,
        "output_sample_rate": sr,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "sweep": results,
        "baseline_direct_python_rtf": 0.185,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path",
                   default="/capstor/store/cscs/swissai/infra01/hf_models/models/openbmb/VoxCPM2")
    p.add_argument("--input", default="data/tts_bench/pl_50.json",
                   help="JSON list of {id, text}")
    p.add_argument("--ref-wav", default="outputs/reference_audio_slow/pl_speaker_0.wav")
    p.add_argument("--ref-txt", default="outputs/reference_audio_slow/pl_speaker_0.txt")
    p.add_argument("--output-dir", default="results/nanovllm_smoke")
    p.add_argument("--batch-sizes", type=str, default="1,4,16,32",
                   help="Comma-separated concurrency levels to sweep")
    p.add_argument("--keep-samples", type=int, default=5,
                   help="Save this many WAVs from the FIRST batch-size run for listening QA")
    p.add_argument("--max-num-seqs", type=int, default=512)
    p.add_argument("--max-num-batched-tokens", type=int, default=16384)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = p.parse_args()
    args.batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]

    stats = asyncio.run(run_sweep(args))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
