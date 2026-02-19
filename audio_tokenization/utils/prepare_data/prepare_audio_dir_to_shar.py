#!/usr/bin/env python3
"""Convert individual audio files + VAD JSONL to Lhotse Shar format.

Reads audio files from a directory tree and VAD timestamps from JSONL files
(one per language+year or arbitrary grouping), applies VAD-aware chunking
via ``merge_and_pack_vad``, resamples to target SR, and writes to Shar.

Each worker processes a subset of JSONL files and writes to its own
``worker_XX/`` sub-directory.  After all workers finish, a merged
``shar_index.json`` is written so that the tokenization pipeline can load
the output directly.

Designed for datasets like VoxPopuli where audio is stored as individual
files (e.g. ``.ogg``) rather than WebDataset tar shards.

Usage:
    python -m audio_tokenization.utils.prepare_data.prepare_audio_dir_to_shar \
        --audio-root /capstor/.../voxpopuli/raw_audios \
        --jsonl-files /capstor/.../per_lang_year/*.jsonl \
        --shar-dir /iopsstor/.../voxpopuli_shar \
        --target-sr 24000 \
        --num-workers 272 \
        --shard-size 2000 \
        --shar-format flac
"""

import argparse
from collections import Counter
import json
import logging
import multiprocessing as mp
import time
from pathlib import Path

from audio_tokenization.utils.prepare_data.common import (
    PREPARE_SUMMARY_FILE,
    SUCCESS_MARKER_FILE,
    WORKER_ASSIGNMENT_FILE,
    WORKER_STATS_FILE,
    build_audio_index,
    build_shar_index,
    distribute_round_robin,
    load_worker_assignment,
    mark_partition_success,
    run_aggregate,
    setup_partition_dir,
    to_mono,
    write_worker_assignment,
)
from audio_tokenization.utils.prepare_data.chunking import (
    _parse_vad_jsonl_line,
    merge_and_pack_vad,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(processName)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

WORKER_SUCCESS_FILE = SUCCESS_MARKER_FILE

_ITEMS_KEY = "resolved_jsonls"


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _convert_worker(args_tuple):
    """Convert a subset of VAD JSONL entries to Shar.

    Each worker writes to its own ``worker_XX/`` directory to avoid contention.
    Resume is considered complete only when ``worker_XX/_SUCCESS`` exists.
    """
    (
        worker_id,
        jsonl_paths,
        audio_index,
        shar_dir,
        target_sr,
        shard_size,
        shar_format,
        min_sr,
        mono_downmix,
        vad_max_chunk_sec,
        vad_min_chunk_sec,
        vad_sample_rate,
        vad_max_merge_gap_sec,
        vad_max_duration_sec,
        audio_ext,
    ) = args_tuple

    worker_dir = Path(shar_dir) / f"worker_{worker_id:02d}"
    worker_stats_path = worker_dir / WORKER_STATS_FILE
    if setup_partition_dir(
        worker_dir,
        success_marker_name=WORKER_SUCCESS_FILE,
        reuse_log=f"Worker {worker_id}: reusing completed Shar in {worker_dir}",
        reset_log=f"Worker {worker_id}: removing partial output in {worker_dir}",
        logger=logger,
    ):
        reused_worker_stats = {}
        if worker_stats_path.is_file():
            try:
                reused_worker_stats = json.loads(worker_stats_path.read_text())
            except Exception:
                reused_worker_stats = {}
        return {
            "worker_id": worker_id,
            "written": -1,
            "skipped": 0,
            "errors": 0,
            "elapsed": 0,
            "total_duration_sec": reused_worker_stats.get("total_duration_sec", 0.0),
            "reused": True,
            "reason_counts": {},
            "worker_stats": reused_worker_stats,
        }

    # SoX resampling backend (per-process state)
    try:
        from lhotse.audio.resampling_backend import set_current_resampling_backend
        set_current_resampling_backend("sox")
    except Exception:
        pass

    from lhotse import Recording
    from lhotse.shar import SharWriter

    reason_counts = Counter()
    runtime_counts = Counter()

    t0 = time.time()
    written = skipped = errors = 0
    total_duration_sec = 0.0

    with SharWriter(
        output_dir=str(worker_dir),
        fields={"recording": shar_format},
        shard_size=shard_size,
    ) as writer:
        for jsonl_path in jsonl_paths:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parsed = _parse_vad_jsonl_line(
                        line,
                        with_duration=True,
                        with_sample_rate=True,
                        with_lang=True,
                    )
                    if parsed is None:
                        runtime_counts["parse_failed"] += 1
                        continue

                    key, timestamps, duration_sec, sr, lang = parsed

                    # Resolve audio path
                    audio_path = audio_index.get(key)
                    if audio_path is None:
                        runtime_counts["missing_audio"] += 1
                        skipped += 1
                        continue

                    # Build Lhotse recording from file
                    try:
                        recording = Recording.from_file(audio_path)
                        cut = recording.to_cut()
                    except Exception as e:
                        errors += 1
                        runtime_counts["failed_build_cut"] += 1
                        if errors <= 5:
                            logger.warning(
                                f"Worker {worker_id} error loading {key}: {e}"
                            )
                        continue

                    # Min sample rate check
                    if min_sr and cut.sampling_rate < min_sr:
                        skipped += 1
                        runtime_counts["skipped_min_sr"] += 1
                        continue

                    # Resample if needed
                    if target_sr and cut.sampling_rate != target_sr:
                        cut = cut.resample(target_sr)
                        runtime_counts["resampled"] += 1

                    # VAD chunking
                    if not timestamps:
                        reason_counts["empty_vad"] += 1
                        skipped += 1
                        continue

                    ranges = merge_and_pack_vad(
                        timestamps=timestamps,
                        audio_duration_sec=float(cut.duration),
                        sample_rate=vad_sample_rate,
                        max_merge_gap_sec=vad_max_merge_gap_sec,
                        max_chunk_sec=vad_max_chunk_sec,
                        min_chunk_sec=vad_min_chunk_sec,
                        max_duration_sec=vad_max_duration_sec,
                    )

                    if not ranges:
                        reason_counts["chunks_below_min_duration"] += 1
                        skipped += 1
                        continue

                    reason_counts["chunked"] += 1

                    for offset, chunk_duration in ranges:
                        try:
                            try:
                                subcut = cut.truncate(
                                    offset=offset,
                                    duration=chunk_duration,
                                    preserve_id=False,
                                )
                            except TypeError:
                                subcut = cut.truncate(
                                    offset=offset, duration=chunk_duration
                                )

                            subcut = to_mono(
                                subcut,
                                mono_downmix=mono_downmix,
                                stats=runtime_counts,
                            )
                            subcut.custom = subcut.custom or {}
                            subcut.custom["global_offset_sec"] = offset
                            subcut.custom["lang"] = lang

                            writer.write(subcut)
                            written += 1
                            total_duration_sec += subcut.duration
                            runtime_counts["cuts_written"] += 1
                        except Exception as e:
                            errors += 1
                            runtime_counts["processing_errors"] += 1
                            if errors <= 5:
                                logger.warning(
                                    f"Worker {worker_id} error on chunk "
                                    f"{key}@{offset:.1f}: {e}"
                                )

                    if written % 1000 == 0 and written > 0:
                        elapsed = time.time() - t0
                        logger.info(
                            f"Worker {worker_id}: {written} written, "
                            f"{skipped} skipped, {errors} errors "
                            f"({written / elapsed:.1f} samples/s)"
                        )

    elapsed = time.time() - t0
    logger.info(
        f"Worker {worker_id} done in {elapsed:.1f}s: "
        f"{written} written, {skipped} skipped, {errors} errors"
    )
    if reason_counts:
        logger.info(f"Worker {worker_id} VAD reasons: {dict(reason_counts)}")

    worker_stats = {
        "worker_id": worker_id,
        "elapsed_sec": elapsed,
        "written": written,
        "skipped": skipped,
        "errors": errors,
        "total_duration_sec": total_duration_sec,
        "reused": False,
        "runtime_counts": dict(runtime_counts),
        "reason_counts": dict(reason_counts),
    }
    worker_stats_path.write_text(json.dumps(worker_stats, indent=2) + "\n")

    mark_partition_success(worker_dir, success_marker_name=WORKER_SUCCESS_FILE)
    return {
        "worker_id": worker_id,
        "written": written,
        "skipped": skipped,
        "errors": errors,
        "elapsed": elapsed,
        "total_duration_sec": total_duration_sec,
        "reused": False,
        "reason_counts": dict(reason_counts),
        "worker_stats": worker_stats,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert audio dir + VAD JSONL -> Lhotse Shar (parallel)",
    )

    # Input
    parser.add_argument("--audio-root", type=Path, required=None,
                        help="Root directory of audio files (searched recursively)")
    parser.add_argument("--audio-ext", type=str, default=".ogg",
                        help="Audio file extension (default: .ogg)")
    parser.add_argument("--jsonl-files", nargs="+", required=None,
                        help="VAD JSONL file paths (shell glob expanded by caller)")

    # Shar output
    parser.add_argument("--shar-dir", type=Path, default=None,
                        help="Output directory for Shar format")
    parser.add_argument("--shard-size", type=int, default=2000,
                        help="Samples per Shar shard (default: 2000)")
    parser.add_argument("--shar-format", type=str, default="flac",
                        choices=["flac", "wav", "mp3", "opus"],
                        help="Audio format in Shar (default: flac)")

    # Audio processing
    parser.add_argument("--target-sr", type=int, default=24000,
                        help="Target sample rate (default: 24000)")
    parser.add_argument("--min-sr", type=int, default=16000,
                        help="Drop audio below this sample rate (default: 16000)")
    parser.add_argument("--no-mono-downmix", action="store_true",
                        help="Select channel 0 instead of averaging stereo channels")

    # VAD chunking
    parser.add_argument("--vad-max-chunk-sec", type=float, default=200.0,
                        help="Target max duration while packing VAD segments")
    parser.add_argument("--vad-min-chunk-sec", type=float, default=5.0,
                        help="Drop chunks shorter than this duration")
    parser.add_argument("--vad-sample-rate", type=int, default=16000,
                        help="Sample rate used to decode VAD timestamp units")
    parser.add_argument("--vad-max-merge-gap-sec", type=float, default=1.0,
                        help="Merge adjacent VAD spans when silence gap <= this threshold")
    parser.add_argument("--vad-max-duration-sec", type=float, default=None,
                        help="Drop atomic speech segments longer than this "
                             "(default: same as --vad-max-chunk-sec)")

    # Parallelism
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of parallel workers (default: one per JSONL file)")

    parser.add_argument("--aggregate", type=Path, default=None, metavar="SHAR_ROOT",
                        help="Aggregate stats from completed multi-node runs and exit.")

    args = parser.parse_args(argv)

    # ---- Aggregate mode ----
    if args.aggregate is not None:
        run_aggregate(args.aggregate)
        return

    if args.audio_root is None:
        parser.error("--audio-root is required (unless using --aggregate)")
    if not args.jsonl_files:
        parser.error("--jsonl-files is required (unless using --aggregate)")
    if args.shar_dir is None:
        parser.error("--shar-dir is required (unless using --aggregate)")

    if not args.audio_root.is_dir():
        raise NotADirectoryError(f"Audio root not found: {args.audio_root}")

    resolved_jsonls = sorted(args.jsonl_files)
    if not resolved_jsonls:
        raise FileNotFoundError("No JSONL files provided via --jsonl-files")

    args.shar_dir.mkdir(parents=True, exist_ok=True)

    # Resume safety: check existing worker assignment
    assignment = load_worker_assignment(args.shar_dir, items_key=_ITEMS_KEY)
    if assignment is not None:
        if assignment[_ITEMS_KEY] != resolved_jsonls:
            raise RuntimeError(
                "Existing worker assignment JSONL list does not match current resolved files. "
                f"Delete {args.shar_dir / WORKER_ASSIGNMENT_FILE} and worker_* directories to start fresh."
            )
        if args.num_workers is not None and int(args.num_workers) != assignment["num_workers"]:
            raise RuntimeError(
                f"Existing worker assignment requires num_workers={assignment['num_workers']}, "
                f"but got {args.num_workers}. Keep num_workers stable when resuming."
            )
        num_workers = assignment["num_workers"]
        logger.info(f"Reusing worker assignment from {assignment['path']} (num_workers={num_workers})")
    else:
        num_workers = min(args.num_workers or len(resolved_jsonls), len(resolved_jsonls))
        assignment_path = write_worker_assignment(
            args.shar_dir, num_workers, resolved_jsonls, items_key=_ITEMS_KEY,
        )
        logger.info(f"Wrote worker assignment to {assignment_path}")

    logger.info(f"Found {len(resolved_jsonls)} JSONL files, using {num_workers} workers")
    logger.info(f"Output: {args.shar_dir}")

    # Build audio index (stem -> full path)
    logger.info(f"Building audio index from {args.audio_root} (*{args.audio_ext}) ...")
    t_idx = time.time()
    audio_index = build_audio_index(args.audio_root, f"**/*{args.audio_ext}")
    logger.info(
        f"Indexed {len(audio_index):,} audio files in {time.time() - t_idx:.1f}s"
    )
    if not audio_index:
        raise FileNotFoundError(
            f"No *{args.audio_ext} files found under {args.audio_root}"
        )

    # Distribute JSONL files across workers (round-robin)
    worker_jsonls = distribute_round_robin(resolved_jsonls, num_workers)

    # Validate VAD params
    if args.vad_max_chunk_sec <= 0:
        raise ValueError("--vad-max-chunk-sec must be > 0")
    if args.vad_min_chunk_sec < 0:
        raise ValueError("--vad-min-chunk-sec must be >= 0")
    if args.vad_min_chunk_sec > args.vad_max_chunk_sec:
        raise ValueError("--vad-min-chunk-sec must be <= --vad-max-chunk-sec")
    if args.vad_sample_rate <= 0:
        raise ValueError("--vad-sample-rate must be > 0")
    if args.vad_max_merge_gap_sec < 0:
        raise ValueError("--vad-max-merge-gap-sec must be >= 0")

    worker_args = [
        (
            wid,
            jsonls,
            audio_index,
            str(args.shar_dir),
            args.target_sr,
            args.shard_size,
            args.shar_format,
            args.min_sr,
            not args.no_mono_downmix,
            args.vad_max_chunk_sec,
            args.vad_min_chunk_sec,
            args.vad_sample_rate,
            args.vad_max_merge_gap_sec,
            args.vad_max_duration_sec,
            args.audio_ext,
        )
        for wid, jsonls in enumerate(worker_jsonls)
        if jsonls
    ]

    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=len(worker_args)) as pool:
        results = pool.map(_convert_worker, worker_args)

    elapsed = time.time() - t0
    total_written = sum(r["written"] for r in results if r["written"] >= 0)
    total_skipped = sum(r["skipped"] for r in results)
    total_errors = sum(r["errors"] for r in results)
    total_reused = sum(1 for r in results if r.get("reused"))
    total_duration_sec = sum(r.get("total_duration_sec", 0.0) for r in results)
    total_reason_counts = Counter()
    total_runtime_counts = Counter()
    for r in results:
        total_reason_counts.update(r.get("reason_counts", {}))
        total_runtime_counts.update(
            (r.get("worker_stats") or {}).get("runtime_counts", {})
        )

    logger.info(
        f"All workers done in {elapsed:.1f}s -- "
        f"{total_written} samples, {total_skipped} skipped, {total_errors} errors, "
        f"{total_duration_sec / 3600.0:.1f} hours written"
    )
    if total_reason_counts:
        logger.info(f"VAD reasons (global): {dict(total_reason_counts)}")
    if total_runtime_counts:
        logger.info(f"Runtime counters (global): {dict(total_runtime_counts)}")

    summary = {
        "version": 1,
        "num_workers": num_workers,
        "workers_reused": total_reused,
        "elapsed_sec": elapsed,
        "total_written": total_written,
        "total_skipped": total_skipped,
        "total_errors": total_errors,
        "total_duration_sec": total_duration_sec,
        "runtime_counts": dict(total_runtime_counts),
        "reason_counts": dict(total_reason_counts),
        "results": results,
    }
    summary_path = args.shar_dir / PREPARE_SUMMARY_FILE
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info(f"Wrote prepare summary: {summary_path}")

    # Build merged index
    build_shar_index(args.shar_dir, num_workers=num_workers)
    mark_partition_success(args.shar_dir, success_marker_name=WORKER_SUCCESS_FILE)
    logger.info("All done!")


if __name__ == "__main__":
    main()
