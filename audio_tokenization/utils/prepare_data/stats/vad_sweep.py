#!/usr/bin/env python3
"""Dry-run VAD chunking stats — sweep merge-gap and min-chunk parameters.

Reads per-shard VAD JSONL files (produced by ``filter_langid_vad.py``) and
computes chunk statistics for a grid of (min_chunk_sec, merge_gap_sec) values
without touching any audio or writing Shar output.

Usage:
    python -m audio_tokenization.utils.prepare_data.stats.vad_sweep \
        --vad-per-shard-dir /path/to/vad_results_per_shard \
        --num-workers 32

    python -m audio_tokenization.utils.prepare_data.stats.vad_sweep \
        --vad-per-shard-dir /path/to/vad_results_per_shard \
        --min-chunk-sweep 1,5,10,20,30 \
        --num-workers 64

    python -m audio_tokenization.utils.prepare_data.stats.vad_sweep \
        --vad-per-shard-dir /iopsstor/scratch/cscs/xyixuan/audio-datasets/unsupervised_peoples_speech_commercial_wds/vad_results_european_per_shard \
        --num-workers 288
"""

import argparse
import logging
import multiprocessing as mp
import time
from pathlib import Path

from audio_tokenization.utils.prepare_data.stats._common import (
    read_jsonl_recordings,
    speech_sec_in_chunks,
)
from audio_tokenization.utils.prepare_data.vad_segmenting import (
    VADChunkingConfig,
    build_chunk_ranges_from_vad,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(processName)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _dry_run_worker(args_tuple):
    """Compute VAD chunk stats for a subset of JSONL files across all (min_chunk, merge_gap) pairs.

    Mirrors the 3-tier logic in ``split_cut_by_vad``:
      tier-1: duration < min_chunk_sec  -> dropped (too short)
      tier-2: min_chunk_sec <= duration < max_chunk_sec -> kept full (no chunking)
      tier-3: duration >= max_chunk_sec  -> VAD-chunked (varies by merge_gap)

    Since min_chunk_sec affects both tier boundaries and chunk filtering,
    the full sweep is 2D: (min_chunk_sec, merge_gap).
    """
    jsonl_paths, merge_gaps, max_chunk_sec, min_chunk_secs, sample_rate, min_sr = args_tuple

    recordings, skipped_min_sr = read_jsonl_recordings(jsonl_paths, min_sr=min_sr)

    # For each min_chunk_sec, classify tiers and sweep merge_gaps.
    results_by_min_chunk = {}
    for min_chunk_sec in min_chunk_secs:
        tier1_too_short = 0
        tier2_kept_full_count = 0
        tier2_kept_full_sec = 0.0
        tier2_speech_sec = 0.0
        tier3_recordings = []

        for timestamps, duration_sec in recordings:
            if duration_sec < min_chunk_sec:
                tier1_too_short += 1
            elif duration_sec < max_chunk_sec:
                tier2_kept_full_count += 1
                tier2_kept_full_sec += duration_sec
                tier2_speech_sec += speech_sec_in_chunks(
                    timestamps, duration_sec, sample_rate,
                    [(0.0, duration_sec)],
                )
            else:
                tier3_recordings.append((timestamps, duration_sec))

        gap_results = {}
        for gap in merge_gaps:
            cfg = VADChunkingConfig(
                max_chunk_sec=max_chunk_sec,
                min_chunk_sec=min_chunk_sec,
                sample_rate=sample_rate,
                max_merge_gap_sec=gap,
            )
            chunked_sec = 0.0
            speech_sec = 0.0
            num_chunks = 0
            empty_vad = 0
            chunks_below_min = 0
            chunks_over_max = 0

            for timestamps, duration_sec in tier3_recordings:
                if not timestamps:
                    empty_vad += 1
                    continue
                chunks = build_chunk_ranges_from_vad(
                    timestamps=timestamps,
                    audio_duration_sec=duration_sec,
                    cfg=cfg,
                )
                if not chunks:
                    chunks_below_min += 1
                speech_sec += speech_sec_in_chunks(
                    timestamps, duration_sec, sample_rate, chunks,
                )
                for _offset, chunk_dur in chunks:
                    chunked_sec += chunk_dur
                    num_chunks += 1
                    if chunk_dur > max_chunk_sec:
                        chunks_over_max += 1

            gap_results[gap] = {
                "chunked_sec": chunked_sec,
                "speech_sec": speech_sec,
                "num_chunks": num_chunks,
                "empty_vad": empty_vad,
                "chunks_below_min": chunks_below_min,
                "chunks_over_max": chunks_over_max,
            }

        results_by_min_chunk[min_chunk_sec] = {
            "too_short": tier1_too_short,
            "kept_full_count": tier2_kept_full_count,
            "kept_full_sec": tier2_kept_full_sec,
            "kept_full_speech_sec": tier2_speech_sec,
            "tier3_count": len(tier3_recordings),
            "gap_results": gap_results,
        }

    total_duration_sec = sum(dur for _, dur in recordings)
    return {
        "num_recordings": len(recordings),
        "skipped_min_sr": skipped_min_sr,
        "total_duration_sec": total_duration_sec,
        "results_by_min_chunk": results_by_min_chunk,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Dry-run: sweep VAD chunking parameters and print stats",
    )
    parser.add_argument("--vad-per-shard-dir", type=Path, required=True,
                        help="Directory of per-shard VAD JSONL files")
    parser.add_argument("--vad-max-chunk-sec", type=float, default=200.0,
                        help="Target max duration while packing VAD segments")
    parser.add_argument("--vad-min-chunk-sec", type=float, default=10.0,
                        help="Drop chunks shorter than this duration")
    parser.add_argument("--vad-sample-rate", type=int, default=16000,
                        help="Sample rate used to decode VAD timestamp units")
    parser.add_argument("--vad-max-merge-gap-sec", type=float, default=0.5,
                        help="Merge adjacent VAD spans when silence gap <= this threshold")
    parser.add_argument("--min-sr", type=int, default=16000,
                        help="Drop audio below this sample rate (default: 16000)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    parser.add_argument("--min-chunk-sweep", type=str, default="1,5,10,20,30",
                        help="Comma-separated min_chunk_sec values to sweep "
                             "(default: '1,5,10,20,30')")
    parser.add_argument("--max-merge-gap-sweep", type=str, default="0,0.25,0.5,1,2,5",
                        help="Comma-separated max_merge_gap_sec values to sweep "
                             "(default: '0,0.25,0.5,1,2,5')")

    args = parser.parse_args(argv)

    if not args.vad_per_shard_dir.is_dir():
        raise NotADirectoryError(f"VAD per-shard directory not found: {args.vad_per_shard_dir}")

    sweep_gaps = sorted(set(float(v.strip()) for v in args.max_merge_gap_sweep.split(",")))
    sweep_min_chunks = sorted(set(float(v.strip()) for v in args.min_chunk_sweep.split(",")))
    logger.info(f"Dry-run: sweeping merge_gap_sec = {sweep_gaps}")
    logger.info(f"Dry-run: sweeping min_chunk_sec = {sweep_min_chunks}")

    # Glob JSONL files and distribute round-robin across workers.
    jsonl_files = sorted(args.vad_per_shard_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files in {args.vad_per_shard_dir}")

    num_workers = min(args.num_workers, len(jsonl_files))
    worker_jsonls = [[] for _ in range(num_workers)]
    for i, jf in enumerate(jsonl_files):
        worker_jsonls[i % num_workers].append(jf)

    dry_run_args = [
        (jfiles, sweep_gaps, args.vad_max_chunk_sec, sweep_min_chunks, args.vad_sample_rate, args.min_sr)
        for jfiles in worker_jsonls
        if jfiles
    ]

    logger.info(f"Dry-run: processing {len(jsonl_files)} JSONL files across {len(dry_run_args)} workers")
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(dry_run_args)) as pool:
        partial_results = pool.map(_dry_run_worker, dry_run_args)
    elapsed = time.time() - t0
    logger.info(f"Dry-run: parallel computation done in {elapsed:.1f}s")

    # SR-filter stats (invariant across min_chunk).
    total_in_jsonl = sum(p["num_recordings"] + p["skipped_min_sr"] for p in partial_results)
    total_skipped_min_sr = sum(p["skipped_min_sr"] for p in partial_results)
    total_recordings = sum(p["num_recordings"] for p in partial_results)
    total_duration_hrs = sum(p["total_duration_sec"] for p in partial_results) / 3600.0

    max_s = args.vad_max_chunk_sec
    print()
    print(f"Recordings in JSONL:    {total_in_jsonl:>10d}")
    print(f"  skipped_min_sr (<{args.min_sr}Hz): {total_skipped_min_sr:>10d}   (dropped)")
    print(f"After SR filter:        {total_recordings:>10d}   ({total_duration_hrs:.1f} hours total)")

    # Print one table per min_chunk_sec.
    for min_s in sweep_min_chunks:
        total_too_short = sum(p["results_by_min_chunk"][min_s]["too_short"] for p in partial_results)
        total_kept_full_count = sum(p["results_by_min_chunk"][min_s]["kept_full_count"] for p in partial_results)
        total_kept_full_sec = sum(p["results_by_min_chunk"][min_s]["kept_full_sec"] for p in partial_results)
        total_kept_full_speech_sec = sum(p["results_by_min_chunk"][min_s]["kept_full_speech_sec"] for p in partial_results)
        total_tier3 = sum(p["results_by_min_chunk"][min_s]["tier3_count"] for p in partial_results)
        no_vad_hrs = total_kept_full_sec / 3600.0
        no_vad_speech_hrs = total_kept_full_speech_sec / 3600.0

        gap_stats = {}
        for gap in sweep_gaps:
            chunked_sec = sum(p["results_by_min_chunk"][min_s]["gap_results"][gap]["chunked_sec"] for p in partial_results)
            speech_sec = sum(p["results_by_min_chunk"][min_s]["gap_results"][gap]["speech_sec"] for p in partial_results)
            num_chunks = sum(p["results_by_min_chunk"][min_s]["gap_results"][gap]["num_chunks"] for p in partial_results)
            empty_vad = sum(p["results_by_min_chunk"][min_s]["gap_results"][gap]["empty_vad"] for p in partial_results)
            chunks_below_min = sum(p["results_by_min_chunk"][min_s]["gap_results"][gap]["chunks_below_min"] for p in partial_results)
            chunks_over_max = sum(p["results_by_min_chunk"][min_s]["gap_results"][gap]["chunks_over_max"] for p in partial_results)
            gap_stats[gap] = {
                "chunked_sec": chunked_sec,
                "speech_sec": speech_sec,
                "num_chunks": num_chunks,
                "empty_vad": empty_vad,
                "chunks_below_min": chunks_below_min,
                "chunks_over_max": chunks_over_max,
            }

        is_current = (min_s == args.vad_min_chunk_sec)
        label = f"min_chunk_sec = {min_s}"
        if is_current:
            label += " (current)"
        print()
        print(f"--- {label} ---")
        print(f"  too_short (<{min_s}s):  {total_too_short:>10d}   (dropped)")
        print(f"  kept_full ({min_s}-{max_s}s): {total_kept_full_count:>10d}   -> {no_vad_hrs:.1f} hours")
        print(f"  long (>={max_s}s):     {total_tier3:>10d}   (VAD-chunked, varies by merge_gap)")

        header = (
            f" {'merge_gap':>9s} | {'total_hrs':>10s} | {'speech_hrs':>10s} | {'%kept':>6s} | {'no_vad_hrs':>10s} | {'chunk_hrs':>10s} | "
            f"{'chunks':>8s} | {'avg_sec':>8s} | {'empty':>6s} | {'<min':>6s} | {'>max':>6s}"
        )
        sep = "-" * len(header)
        print(sep)
        print(header)
        print(sep)
        for gap in sweep_gaps:
            gs = gap_stats[gap]
            chunk_hrs = gs["chunked_sec"] / 3600.0
            speech_hrs = gs["speech_sec"] / 3600.0 + no_vad_speech_hrs
            total_hrs = no_vad_hrs + chunk_hrs
            pct_kept = (speech_hrs / total_duration_hrs * 100.0) if total_duration_hrs > 0 else 0.0
            n_chunks = gs["num_chunks"]
            avg = (gs["chunked_sec"] / n_chunks) if n_chunks else 0.0
            marker = " <-- current" if (gap == args.vad_max_merge_gap_sec and is_current) else ""
            print(
                f" {gap:>9.2f} | {total_hrs:>10.1f} | {speech_hrs:>10.1f} | {pct_kept:>5.1f}% | {no_vad_hrs:>10.1f} | {chunk_hrs:>10.1f} | "
                f"{n_chunks:>8d} | {avg:>8.1f} | {gs['empty_vad']:>6d} | "
                f"{gs['chunks_below_min']:>6d} | {gs['chunks_over_max']:>6d}{marker}"
            )
        print(sep)

    logger.info("Dry-run complete.")


if __name__ == "__main__":
    main()
