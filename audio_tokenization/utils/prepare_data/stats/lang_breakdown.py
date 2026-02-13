#!/usr/bin/env python3
"""Per-language breakdown of samples and post-VAD speech hours from VAD JSONL files.

Reads per-shard VAD JSONL files (produced by ``filter_langid_vad.py``) and
prints a table of sample counts, raw hours, and post-VAD speech hours per
language.  Applies the same 3-tier logic as ``vad_sweep.py``:

  tier-1: duration < min_chunk_sec  -> dropped (too short)
  tier-2: min_chunk_sec <= duration < max_chunk_sec -> kept full
  tier-3: duration >= max_chunk_sec  -> VAD-chunked

Usage:
    python -m audio_tokenization.utils.prepare_data.stats.lang_breakdown \
        --vad-per-shard-dir /path/to/vad_results_per_shard \
        --num-workers 64

    python -m audio_tokenization.utils.prepare_data.stats.lang_breakdown \
        --vad-per-shard-dir /path/to/vad_results_per_shard \
        --vad-min-chunk-sec 5 --vad-max-merge-gap-sec 1.0 \
        --num-workers 64
"""

import argparse
import logging
import multiprocessing as mp
import time
from collections import Counter
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


def _lang_worker(args_tuple):
    """Compute per-language raw + post-VAD stats for a subset of JSONL files."""
    jsonl_paths, min_sr, min_chunk_sec, max_chunk_sec, max_merge_gap_sec, sample_rate = args_tuple

    recordings, skipped_min_sr = read_jsonl_recordings(
        jsonl_paths, min_sr=min_sr, with_lang=True,
    )

    cfg = VADChunkingConfig(
        max_chunk_sec=max_chunk_sec,
        min_chunk_sec=min_chunk_sec,
        sample_rate=sample_rate,
        max_merge_gap_sec=max_merge_gap_sec,
    )

    lang_counts = Counter()          # total recordings per lang (after SR filter)
    lang_raw_sec = Counter()         # raw duration per lang
    lang_kept_count = Counter()      # kept recordings (tier-2 + tier-3 with chunks)
    lang_kept_sec = Counter()        # kept chunk duration incl. gaps (tokenized)
    lang_speech_sec = Counter()      # post-VAD speech seconds per lang
    lang_too_short = Counter()       # tier-1 dropped per lang

    for timestamps, duration_sec, lang in recordings:
        lang_counts[lang] += 1
        lang_raw_sec[lang] += duration_sec

        if duration_sec < min_chunk_sec:
            # tier-1: too short, dropped
            lang_too_short[lang] += 1
        elif duration_sec < max_chunk_sec:
            # tier-2: kept full
            lang_kept_count[lang] += 1
            lang_kept_sec[lang] += duration_sec
            lang_speech_sec[lang] += speech_sec_in_chunks(
                timestamps, duration_sec, sample_rate,
                [(0.0, duration_sec)],  # whole recording as one chunk
            )
        else:
            # tier-3: VAD-chunked
            if not timestamps:
                continue
            chunks = build_chunk_ranges_from_vad(
                timestamps=timestamps,
                audio_duration_sec=duration_sec,
                cfg=cfg,
            )
            if chunks:
                lang_kept_count[lang] += 1
                lang_kept_sec[lang] += sum(d for _, d in chunks)
                lang_speech_sec[lang] += speech_sec_in_chunks(
                    timestamps, duration_sec, sample_rate, chunks,
                )

    return {
        "num_recordings": len(recordings),
        "skipped_min_sr": skipped_min_sr,
        "lang_counts": dict(lang_counts),
        "lang_raw_sec": dict(lang_raw_sec),
        "lang_kept_count": dict(lang_kept_count),
        "lang_kept_sec": dict(lang_kept_sec),
        "lang_speech_sec": dict(lang_speech_sec),
        "lang_too_short": dict(lang_too_short),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Per-language breakdown of samples and post-VAD speech hours",
    )
    parser.add_argument("--vad-per-shard-dir", type=Path, required=True,
                        help="Directory of per-shard VAD JSONL files")
    parser.add_argument("--min-sr", type=int, default=16000,
                        help="Drop audio below this sample rate (default: 16000)")
    parser.add_argument("--vad-min-chunk-sec", type=float, default=5.0,
                        help="Drop chunks shorter than this (default: 5.0)")
    parser.add_argument("--vad-max-chunk-sec", type=float, default=200.0,
                        help="Target max duration for VAD chunking (default: 200.0)")
    parser.add_argument("--vad-max-merge-gap-sec", type=float, default=1.0,
                        help="Merge adjacent VAD spans when gap <= this (default: 1.0)")
    parser.add_argument("--vad-sample-rate", type=int, default=16000,
                        help="Sample rate for VAD timestamp units (default: 16000)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")

    args = parser.parse_args(argv)

    if not args.vad_per_shard_dir.is_dir():
        raise NotADirectoryError(f"VAD per-shard directory not found: {args.vad_per_shard_dir}")

    jsonl_files = sorted(args.vad_per_shard_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files in {args.vad_per_shard_dir}")

    num_workers = min(args.num_workers, len(jsonl_files))
    worker_jsonls = [[] for _ in range(num_workers)]
    for i, jf in enumerate(jsonl_files):
        worker_jsonls[i % num_workers].append(jf)

    worker_args = [
        (jfiles, args.min_sr, args.vad_min_chunk_sec, args.vad_max_chunk_sec,
         args.vad_max_merge_gap_sec, args.vad_sample_rate)
        for jfiles in worker_jsonls
        if jfiles
    ]

    logger.info(f"Processing {len(jsonl_files)} JSONL files across {len(worker_args)} workers")
    logger.info(f"VAD config: min_chunk={args.vad_min_chunk_sec}s, "
                f"max_chunk={args.vad_max_chunk_sec}s, "
                f"merge_gap={args.vad_max_merge_gap_sec}s")
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(worker_args)) as pool:
        partial_results = pool.map(_lang_worker, worker_args)
    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.1f}s")

    # Aggregate.
    total_recordings = sum(p["num_recordings"] for p in partial_results)
    total_skipped = sum(p["skipped_min_sr"] for p in partial_results)
    agg_counts = Counter()
    agg_raw_sec = Counter()
    agg_kept_count = Counter()
    agg_kept_sec = Counter()
    agg_speech_sec = Counter()
    agg_too_short = Counter()
    for p in partial_results:
        agg_counts.update(p["lang_counts"])
        agg_raw_sec.update(p["lang_raw_sec"])
        agg_kept_count.update(p["lang_kept_count"])
        agg_kept_sec.update(p["lang_kept_sec"])
        agg_speech_sec.update(p["lang_speech_sec"])
        agg_too_short.update(p["lang_too_short"])

    total_raw_hrs = sum(agg_raw_sec.values()) / 3600.0
    total_kept_hrs = sum(agg_kept_sec.values()) / 3600.0
    total_speech_hrs = sum(agg_speech_sec.values()) / 3600.0
    total_kept = sum(agg_kept_count.values())

    # Print.
    print()
    print(f"Recordings after SR filter: {total_recordings:,}  (skipped {total_skipped:,} < {args.min_sr}Hz)")
    print(f"Total raw duration: {total_raw_hrs:,.1f} hours")
    print(f"Kept (tokenized):   {total_kept_hrs:,.1f} hours  "
          f"({total_kept_hrs / total_raw_hrs * 100:.1f}% of raw)" if total_raw_hrs > 0 else "")
    print(f"Post-VAD speech:    {total_speech_hrs:,.1f} hours  "
          f"({total_speech_hrs / total_raw_hrs * 100:.1f}% of raw)" if total_raw_hrs > 0 else "")
    print(f"VAD config: min_chunk={args.vad_min_chunk_sec}s, "
          f"max_chunk={args.vad_max_chunk_sec}s, merge_gap={args.vad_max_merge_gap_sec}s")
    print()

    header = (
        f" {'lang':<8s} | {'samples':>10s} | {'kept':>8s} | {'dropped':>8s} | "
        f"{'raw_hrs':>10s} | {'kept_hrs':>10s} | {'speech_hrs':>10s} | "
        f"{'%samples':>8s} | {'%speech':>8s}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for lang, count in agg_counts.most_common():
        raw_hrs = agg_raw_sec[lang] / 3600.0
        kept_hrs = agg_kept_sec[lang] / 3600.0
        speech_hrs = agg_speech_sec[lang] / 3600.0
        kept = agg_kept_count[lang]
        dropped = agg_too_short[lang]
        pct_samples = count / total_recordings * 100.0 if total_recordings else 0.0
        pct_speech = speech_hrs / total_speech_hrs * 100.0 if total_speech_hrs else 0.0
        print(
            f" {lang:<8s} | {count:>10,d} | {kept:>8,d} | {dropped:>8,d} | "
            f"{raw_hrs:>10.1f} | {kept_hrs:>10.1f} | {speech_hrs:>10.1f} | "
            f"{pct_samples:>7.2f}% | {pct_speech:>7.2f}%"
        )
    print(sep)
    print(
        f" {'TOTAL':<8s} | {total_recordings:>10,d} | {total_kept:>8,d} | "
        f"{sum(agg_too_short.values()):>8,d} | "
        f"{total_raw_hrs:>10.1f} | {total_kept_hrs:>10.1f} | {total_speech_hrs:>10.1f} | "
        f" 100.00% |  100.00%"
    )
    print()


if __name__ == "__main__":
    main()
