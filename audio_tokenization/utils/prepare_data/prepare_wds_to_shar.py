#!/usr/bin/env python3
"""Convert standard WebDataset shards (.wav/.flac/.mp3) to Lhotse Shar format.

Reads standard WDS tars via the webdataset library, creates Lhotse Cuts from
raw audio bytes (Recording.from_bytes), resamples to target SR, and writes to
Shar. Each worker processes a subset of tar shards and writes to its own
``worker_XX/`` sub-directory. After all workers finish, a merged
``shar_index.json`` is written so that the tokenization pipeline can load the
output directly with ``source_type: shar, stage: tokenize`` — no prepare
stage needed.

Note: Lhotse's ``CutSet.from_webdataset()`` expects Lhotse's own pickle
format, NOT standard WDS tars. This script bridges the gap.

Usage:
    python -m audio_tokenization.utils.prepare_data.prepare_wds_to_shar \
        --wds-shards '/path/to/shards/*.tar' \
        --shar-dir /output/path/shar \
        --target-sr 24000 \
        --num-workers 288 \
        --shard-size 2000 \
        --shar-format flac \
        --min-sr 16000
"""

import argparse
from collections import Counter
import glob
import json
import logging
import multiprocessing as mp
import time
from pathlib import Path
from typing import Optional

from audio_tokenization.utils.prepare_data.common import (
    SUCCESS_MARKER_FILE,
    build_shar_index_from_parts,
    mark_partition_success,
    setup_partition_dir,
)
from audio_tokenization.utils.prepare_data.vad_segmenting import (
    VADChunkingConfig,
    canonical_sample_key,
    load_vad_from_per_shard_dir,
    shard_name_from_tar_path,
    split_cut_by_vad,
    vad_per_shard_file,
)

logging.basicConfig(
    
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(processName)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

WORKER_ASSIGNMENT_FILE = "_worker_assignment.json"
WORKER_SUCCESS_FILE = SUCCESS_MARKER_FILE
WORKER_STATS_FILE = "worker_stats.json"
PREPARE_SUMMARY_FILE = "prepare_summary.json"


# ---------------------------------------------------------------------------
# WDS → Lhotse cuts iterator
# ---------------------------------------------------------------------------

AUDIO_SUFFIXES = (".wav", ".flac", ".mp3")
TEXT_SUFFIX = ".txt"


def _to_mono(cut, mono_downmix=False, stats=None):
    if cut.num_channels <= 1:
        return cut
    if mono_downmix:
        try:
            result = cut.to_mono(mono_downmix=True)
            # Force-load to catch AudioLoadingError from broken headers now,
            # rather than letting it bubble up later in SharWriter.
            result.load_audio()
            return result
        except Exception:
            if stats is not None:
                stats["downmix_fallback_ch0"] += 1
            # Broken header — fall back to channel 0.
    result = cut.to_mono(mono_downmix=False)
    if isinstance(result, list):
        return result[0]  # Take first channel (channel 0)
    return result


def iter_tar_cuts(tar_paths, text_ext=TEXT_SUFFIX, stats: Optional[Counter] = None):
    """Iterate over WDS tar shards and yield Lhotse cuts with supervisions.

    Uses ``tarfile`` directly instead of the ``webdataset`` library to avoid
    filename parsing issues — WDS splits on ``.`` which breaks dirty filenames
    with spaces or dots (e.g. ``"Ft. jim jones).mp3"``).

    If ``text_ext`` is set (default ``.txt``), text files sharing the same key
    as an audio file are attached as a ``SupervisionSegment`` on the cut.
    """
    import tarfile
    from lhotse import Recording, SupervisionSegment

    for tar_path in tar_paths:
        with tarfile.open(tar_path) as tf:
            # Phase 1: collect text and audio members in a single scan.
            texts = {}
            audio_members = []
            for member in tf:
                if not member.isfile():
                    continue
                if text_ext and member.name.endswith(text_ext):
                    key = member.name.rsplit(".", 1)[0]
                    try:
                        texts[key] = tf.extractfile(member).read().decode("utf-8").strip()
                    except Exception:
                        if stats is not None:
                            stats["text_decode_failed"] += 1
                        pass
                elif member.name.endswith(AUDIO_SUFFIXES):
                    audio_members.append(member)

            # Phase 2: decode audio and attach text as supervision.
            for member in audio_members:
                key = member.name.rsplit(".", 1)[0]
                try:
                    extracted = tf.extractfile(member)
                    if extracted is None:
                        if stats is not None:
                            stats["missing_payload"] += 1
                        raise ValueError("tar member has no readable payload")
                    audio_bytes = extracted.read()
                    recording = Recording.from_bytes(data=audio_bytes, recording_id=key)
                    # Lazy: only reads the audio header, no decoding yet.
                    cut = recording.to_cut()
                except Exception as e:
                    if stats is not None:
                        stats["failed_build_cut"] += 1
                    logger.warning(f"Skipping {key}: failed to build cut ({e})")
                    continue

                text = texts.get(key)
                if text:
                    cut.supervisions = [SupervisionSegment(
                        id=cut.id,
                        recording_id=cut.recording_id,
                        start=0.0,
                        duration=cut.duration,
                        text=text,
                    )]

                if stats is not None:
                    stats["cuts_yielded"] += 1
                yield cut


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _convert_worker(args_tuple):
    """Convert a subset of WDS tar shards to Shar.

    Each worker writes to its own ``worker_XX/`` directory to avoid contention.
    Resume is considered complete only when ``worker_XX/_SUCCESS`` exists.
    Partial output (cuts manifests without marker) is deleted and recomputed.
    """
    (
        worker_id,
        tar_paths,
        shar_dir,
        target_sr,
        shard_size,
        shar_format,
        min_sr,
        text_ext,
        mono_downmix,
        vad_per_shard_dir,
        vad_max_chunk_sec,
        vad_min_chunk_sec,
        vad_sample_rate,
        vad_max_merge_gap_sec,
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

    from lhotse.shar import SharWriter

    use_vad_segmenting = bool(vad_per_shard_dir)
    reason_counts = Counter()

    # Load VAD entries from per-shard files for this worker's tar shards.
    if use_vad_segmenting:
        vad_cfg = VADChunkingConfig(
            max_chunk_sec=float(vad_max_chunk_sec),
            min_chunk_sec=float(vad_min_chunk_sec),
            sample_rate=int(vad_sample_rate),
            max_merge_gap_sec=float(vad_max_merge_gap_sec),
        )
        vad_lookup, lang_lookup = load_vad_from_per_shard_dir(
            Path(vad_per_shard_dir), tar_paths, with_lang=True, logger=logger,
        )
    else:
        vad_cfg = None
        vad_lookup = {}
        lang_lookup = {}

    t0 = time.time()
    written = skipped = errors = 0
    total_duration_sec = 0.0
    runtime_counts = Counter()

    with SharWriter(
        output_dir=str(worker_dir),
        fields={"recording": shar_format},
        shard_size=shard_size,
    ) as writer:
        for cut in iter_tar_cuts(tar_paths, text_ext=text_ext, stats=runtime_counts):
            try:
                if min_sr and cut.sampling_rate < min_sr:
                    skipped += 1
                    runtime_counts["skipped_min_sr"] += 1
                    continue

                if target_sr and cut.sampling_rate != target_sr:
                    cut = cut.resample(target_sr)
                    runtime_counts["resampled"] += 1

                # No intermediate WAV dump: decode -> optional resample -> split -> write.
                if use_vad_segmenting:
                    out_cuts, reason = split_cut_by_vad(
                        cut=cut,
                        sample_key=cut.recording_id,
                        vad_lookup=vad_lookup,
                        cfg=vad_cfg,
                    )
                    reason_counts[reason] += 1
                else:
                    out_cuts = [cut]

                if not out_cuts:
                    skipped += 1
                    runtime_counts["skipped_empty_output"] += 1
                    continue

                sample_lang = lang_lookup.get(
                    canonical_sample_key(cut.recording_id)
                ) if lang_lookup else None

                for out_cut in out_cuts:
                    out_cut = _to_mono(out_cut, mono_downmix=mono_downmix, stats=runtime_counts)
                    if sample_lang is not None:
                        out_cut.custom = out_cut.custom or {}
                        out_cut.custom["lang"] = sample_lang
                    writer.write(out_cut)
                    written += 1
                    total_duration_sec += out_cut.duration
                    runtime_counts["cuts_written"] += 1

                if written % 1000 == 0:
                    elapsed = time.time() - t0
                    logger.info(
                        f"Worker {worker_id}: {written} written, {skipped} skipped, "
                        f"{errors} errors ({written / elapsed:.1f} samples/s)"
                    )

            except Exception as e:
                errors += 1
                runtime_counts["processing_errors"] += 1
                if errors <= 5:
                    logger.warning(f"Worker {worker_id} error on {cut.id}: {e}")

    elapsed = time.time() - t0
    logger.info(
        f"Worker {worker_id} done in {elapsed:.1f}s: "
        f"{written} written, {skipped} skipped, {errors} errors"
    )
    if use_vad_segmenting and reason_counts:
        logger.info(f"Worker {worker_id} VAD reasons: {dict(reason_counts)}")

    worker_stats = {
        "worker_id": worker_id,
        "elapsed_sec": elapsed,
        "written": written,
        "skipped": skipped,
        "errors": errors,
        "total_duration_sec": total_duration_sec,
        "reused": False,
        "vad_enabled": use_vad_segmenting,
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
# Index builder
# ---------------------------------------------------------------------------

def build_shar_index(shar_root: Path, num_workers: int, index_filename: str = "shar_index.json"):
    """Build a merged ``shar_index.json`` from all ``worker_*`` directories.

    The index maps field names (``cuts``, ``recording``, ...) to sorted lists
    of absolute file paths, so that ``CutSet.from_shar(fields=...)`` can load
    all worker outputs as a single logical CutSet.
    """
    worker_dirs = [shar_root / f"worker_{wid:02d}" for wid in range(num_workers)]
    index_path, cuts_count = build_shar_index_from_parts(
        shar_root=shar_root,
        part_dirs=worker_dirs,
        index_filename=index_filename,
        success_marker_name=WORKER_SUCCESS_FILE,
    )
    logger.info(f"Wrote merged index: {index_path} ({cuts_count} cut shards)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _distribute_round_robin(resolved_shards, num_workers):
    worker_shards = [[] for _ in range(num_workers)]
    for i, tar_path in enumerate(resolved_shards):
        worker_shards[i % num_workers].append(tar_path)
    return worker_shards


def _assignment_path(shar_dir: Path) -> Path:
    return shar_dir / WORKER_ASSIGNMENT_FILE


def _load_worker_assignment(shar_dir: Path):
    path = _assignment_path(shar_dir)
    if not path.is_file():
        return None

    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid assignment file format: {path}")

    try:
        num_workers = int(payload["num_workers"])
        resolved = payload["resolved_shards"]
    except KeyError as e:
        raise RuntimeError(f"Invalid assignment file (missing key {e.args[0]}): {path}") from e

    if num_workers < 1:
        raise RuntimeError(f"Invalid num_workers in assignment file: {path}")
    if not isinstance(resolved, list):
        raise RuntimeError(f"Invalid resolved_shards in assignment file: {path}")

    return {
        "path": path,
        "num_workers": num_workers,
        "resolved_shards": [str(p) for p in resolved],
    }


def _write_worker_assignment(shar_dir: Path, num_workers: int, resolved_shards):
    path = _assignment_path(shar_dir)
    payload = {
        "version": 1,
        "num_workers": int(num_workers),
        "resolved_shards": list(resolved_shards),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def _run_aggregate(shar_root: Path):
    """Read prepare_summary.json from all node_*/ dirs, sum totals, and print."""
    node_dirs = sorted(shar_root.glob("node_*"))
    if not node_dirs:
        # Fallback: check if shar_root itself has a summary (single-node run).
        single = shar_root / PREPARE_SUMMARY_FILE
        if single.is_file():
            node_dirs = [shar_root]
        else:
            raise FileNotFoundError(
                f"No node_*/ dirs (or {PREPARE_SUMMARY_FILE}) found under {shar_root}"
            )

    summaries = []
    for nd in node_dirs:
        sp = nd / PREPARE_SUMMARY_FILE
        if not sp.is_file():
            logger.warning(f"Missing {sp}, skipping")
            continue
        summaries.append(json.loads(sp.read_text()))

    if not summaries:
        raise FileNotFoundError(f"No {PREPARE_SUMMARY_FILE} found in any node dir")

    total_written = 0
    total_skipped = 0
    total_errors = 0
    total_duration_sec = 0.0
    total_elapsed_sec = 0.0
    agg_reason = Counter()
    agg_runtime = Counter()

    for s in summaries:
        total_written += s.get("total_written", 0)
        total_skipped += s.get("total_skipped", 0)
        total_errors += s.get("total_errors", 0)
        total_duration_sec += s.get("total_duration_sec", 0.0)
        total_elapsed_sec = max(total_elapsed_sec, s.get("elapsed_sec", 0.0))
        agg_reason.update(s.get("reason_counts", {}))
        agg_runtime.update(s.get("runtime_counts", {}))

    total_hours = total_duration_sec / 3600.0

    print()
    print(f"=== Aggregate stats from {len(summaries)} node(s) under {shar_root} ===")
    print(f"  Samples written:  {total_written:>12d}")
    print(f"  Samples skipped:  {total_skipped:>12d}")
    print(f"  Errors:           {total_errors:>12d}")
    print(f"  Total hours:      {total_hours:>12.1f}")
    print(f"  Max wall-time:    {total_elapsed_sec:>12.1f}s")
    if agg_reason:
        print(f"  VAD reasons:      {dict(agg_reason)}")
    if agg_runtime:
        print(f"  Runtime counters: {dict(agg_runtime)}")
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert standard WDS → Lhotse Shar (parallel)",
    )

    # Input
    parser.add_argument("--wds-shards", type=str, nargs="+", default=None,
                        help="Glob patterns or file paths for WDS tar shards")

    # Shar output
    parser.add_argument("--shar-dir", type=Path, default=None,
                        help="Output directory for Shar format")
    parser.add_argument("--shard-size", type=int, default=5000,
                        help="Samples per Shar shard (default: 5000)")
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

    # Text
    parser.add_argument("--text-ext", type=str, default=".txt",
                        help="Extension for text files in WDS tars (set to '' to skip)")

    # Parallelism
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of parallel workers (default: one per WDS shard)")
    parser.add_argument("--vad-segmentation", action="store_true",
                        help="Split long recordings into speech-aware segments during prepare")
    parser.add_argument("--vad-per-shard-dir", type=Path, default=None,
                        help="Directory of per-shard VAD JSONL files (required with --vad-segmentation)")
    parser.add_argument("--vad-max-chunk-sec", type=float, default=200.0,
                        help="Target max duration while packing VAD segments")
    parser.add_argument("--vad-min-chunk-sec", type=float, default=10.0,
                        help="Drop chunks shorter than this duration")
    parser.add_argument("--vad-sample-rate", type=int, default=16000,
                        help="Sample rate used to decode VAD timestamp units")
    parser.add_argument("--vad-max-merge-gap-sec", type=float, default=0.5,
                        help="Merge adjacent VAD spans when silence gap <= this threshold")

    parser.add_argument("--aggregate", type=Path, default=None, metavar="SHAR_ROOT",
                        help="Aggregate stats from completed multi-node runs and exit. "
                             "Reads prepare_summary.json from all node_*/ dirs under SHAR_ROOT.")

    args = parser.parse_args(argv)

    # ---- Aggregate mode: read summaries and exit ----
    if args.aggregate is not None:
        _run_aggregate(args.aggregate)
        return

    if not args.wds_shards:
        parser.error("--wds-shards is required (unless using --aggregate)")
    if args.shar_dir is None:
        parser.error("--shar-dir is required (unless using --aggregate)")

    resolved = sorted(set(p for pattern in args.wds_shards for p in glob.glob(pattern)))
    if not resolved:
        raise FileNotFoundError(f"No files match patterns: {args.wds_shards}")

    # Pre-filter shards that have no VAD file (avoids empty workers).
    if args.vad_segmentation and args.vad_per_shard_dir:
        before = len(resolved)
        resolved = [
            p for p in resolved
            if vad_per_shard_file(args.vad_per_shard_dir, shard_name_from_tar_path(p)).is_file()
        ]
        skipped_shards = before - len(resolved)
        if skipped_shards:
            logger.info(f"Skipped {skipped_shards} shards with no VAD file ({len(resolved)} remaining)")
        if not resolved:
            logger.info(
                "All shards were skipped (no matching VAD files) — nothing to do."
            )
            return

    args.shar_dir.mkdir(parents=True, exist_ok=True)

    assignment = _load_worker_assignment(args.shar_dir)
    if assignment is not None:
        if assignment["resolved_shards"] != resolved:
            raise RuntimeError(
                "Existing worker assignment shard list does not match current resolved shards. "
                f"Delete {_assignment_path(args.shar_dir)} and worker_* directories to start fresh."
            )
        if args.num_workers is not None and int(args.num_workers) != assignment["num_workers"]:
            raise RuntimeError(
                f"Existing worker assignment requires num_workers={assignment['num_workers']}, "
                f"but got {args.num_workers}. Keep num_workers stable when resuming."
            )
        num_workers = assignment["num_workers"]
        logger.info(f"Reusing worker assignment from {assignment['path']} (num_workers={num_workers})")
    else:
        num_workers = min(args.num_workers or len(resolved), len(resolved))
        assignment_path = _write_worker_assignment(args.shar_dir, num_workers, resolved)
        logger.info(f"Wrote worker assignment to {assignment_path}")

    logger.info(f"Found {len(resolved)} WDS shards, using {num_workers} workers")
    logger.info(f"Output: {args.shar_dir}")

    # Distribute tar shards across workers (round-robin)
    worker_shards = _distribute_round_robin(resolved, num_workers)

    if args.vad_segmentation:
        if args.vad_per_shard_dir is None:
            raise ValueError("--vad-per-shard-dir is required with --vad-segmentation")
        if not args.vad_per_shard_dir.is_dir():
            raise NotADirectoryError(f"VAD per-shard directory not found: {args.vad_per_shard_dir}")
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
        logger.info(f"VAD segmenting enabled: per_shard_dir={args.vad_per_shard_dir}")
    else:
        logger.info("VAD segmenting disabled; writing full recordings")

    worker_args = [
        (
            wid,
            shards,
            str(args.shar_dir),
            args.target_sr,
            args.shard_size,
            args.shar_format,
            args.min_sr,
            args.text_ext,
            not args.no_mono_downmix,
            str(args.vad_per_shard_dir) if args.vad_segmentation else None,
            args.vad_max_chunk_sec,
            args.vad_min_chunk_sec,
            args.vad_sample_rate,
            args.vad_max_merge_gap_sec,
        )
        for wid, shards in enumerate(worker_shards)
        if shards
    ]

    t0 = time.time()
    ctx = mp.get_context("spawn")
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
        total_runtime_counts.update((r.get("worker_stats") or {}).get("runtime_counts", {}))

    logger.info(
        f"All workers done in {elapsed:.1f}s — "
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

    # Build merged index for pipeline compatibility
    build_shar_index(args.shar_dir, num_workers=num_workers)
    mark_partition_success(args.shar_dir, success_marker_name=WORKER_SUCCESS_FILE)
    logger.info("All done!")


if __name__ == "__main__":
    main()
