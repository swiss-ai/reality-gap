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
import glob
import json
import logging
import multiprocessing as mp
import shutil
import time
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(processName)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

WORKER_ASSIGNMENT_FILE = "_worker_assignment.json"
WORKER_SUCCESS_FILE = "_SUCCESS"


# ---------------------------------------------------------------------------
# WDS → Lhotse cuts iterator
# ---------------------------------------------------------------------------

AUDIO_SUFFIXES = (".wav", ".flac", ".mp3")
TEXT_SUFFIX = ".txt"


def _to_mono(cut):
    return cut.to_mono(mono_downmix=True) if cut.num_channels > 1 else cut


def iter_tar_cuts(tar_paths, text_ext=TEXT_SUFFIX):
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
                        pass
                elif member.name.endswith(AUDIO_SUFFIXES):
                    audio_members.append(member)

            # Phase 2: decode audio and attach text as supervision.
            for member in audio_members:
                key = member.name.rsplit(".", 1)[0]
                try:
                    audio_bytes = tf.extractfile(member).read()
                    recording = Recording.from_bytes(data=audio_bytes, recording_id=key)
                except Exception as e:
                    logger.warning(f"Skipping {key}: failed to decode audio ({e})")
                    continue

                cut = _to_mono(recording.to_cut())

                text = texts.get(key)
                if text:
                    cut.supervisions = [SupervisionSegment(
                        id=cut.id,
                        recording_id=cut.recording_id,
                        start=0.0,
                        duration=cut.duration,
                        text=text,
                    )]

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
    worker_id, tar_paths, shar_dir, target_sr, shard_size, shar_format, min_sr, text_ext = args_tuple

    worker_dir = Path(shar_dir) / f"worker_{worker_id:02d}"
    success_marker = worker_dir / WORKER_SUCCESS_FILE

    # Resume: reuse only if worker completed successfully.
    if success_marker.is_file():
        logger.info(f"Worker {worker_id}: reusing completed Shar in {worker_dir}")
        return {"worker_id": worker_id, "written": -1, "skipped": 0, "errors": 0, "elapsed": 0}

    # Partial output from interrupted runs is unsafe to reuse.
    if worker_dir.is_dir():
        logger.warning(f"Worker {worker_id}: removing partial output in {worker_dir}")
        shutil.rmtree(worker_dir)

    worker_dir.mkdir(parents=True, exist_ok=True)

    # SoX resampling backend (per-process state)
    try:
        from lhotse.audio.resampling_backend import set_current_resampling_backend
        set_current_resampling_backend("sox")
    except Exception:
        pass

    from lhotse.shar import SharWriter

    t0 = time.time()
    written = skipped = errors = 0

    with SharWriter(
        output_dir=str(worker_dir),
        fields={"recording": shar_format},
        shard_size=shard_size,
    ) as writer:
        for cut in iter_tar_cuts(tar_paths, text_ext=text_ext):
            try:
                if min_sr and cut.sampling_rate < min_sr:
                    skipped += 1
                    continue

                if target_sr and cut.sampling_rate != target_sr:
                    cut = cut.resample(target_sr)

                writer.write(cut)
                written += 1

                if written % 1000 == 0:
                    elapsed = time.time() - t0
                    logger.info(
                        f"Worker {worker_id}: {written} written, {skipped} skipped, "
                        f"{errors} errors ({written / elapsed:.1f} samples/s)"
                    )

            except Exception as e:
                errors += 1
                if errors <= 5:
                    logger.warning(f"Worker {worker_id} error on {cut.id}: {e}")

    elapsed = time.time() - t0
    logger.info(
        f"Worker {worker_id} done in {elapsed:.1f}s: "
        f"{written} written, {skipped} skipped, {errors} errors"
    )
    success_marker.write_text("ok\n")
    return {"worker_id": worker_id, "written": written, "skipped": skipped, "errors": errors, "elapsed": elapsed}


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_shar_index(shar_root: Path, index_filename: str = "shar_index.json"):
    """Build a merged ``shar_index.json`` from all ``worker_*`` directories.

    The index maps field names (``cuts``, ``recording``, ...) to sorted lists
    of absolute file paths, so that ``CutSet.from_shar(fields=...)`` can load
    all worker outputs as a single logical CutSet.
    """
    fields = defaultdict(list)

    for worker_dir in sorted(shar_root.glob("worker_*")):
        if not worker_dir.is_dir():
            continue

        success_marker = worker_dir / WORKER_SUCCESS_FILE
        if not success_marker.is_file():
            if any(worker_dir.glob("cuts*.jsonl.gz")):
                logger.warning(f"Skipping partial worker dir (missing _SUCCESS): {worker_dir}")
            continue

        if not any(worker_dir.glob("cuts*.jsonl.gz")):
            logger.warning(f"Skipping worker dir with _SUCCESS but no cuts manifests: {worker_dir}")
            continue

        for p in sorted(worker_dir.iterdir()):
            if not p.is_file():
                continue
            field = p.name.split(".")[0]
            if field == "cuts" and p.suffix == ".gz":
                fields["cuts"].append(str(p))
            elif p.suffix in (".tar", ".gz"):
                fields[field].append(str(p))

    if not fields.get("cuts"):
        raise FileNotFoundError(f"No Shar cuts found under {shar_root}")

    payload = {
        "version": 1,
        "fields": {k: sorted(v) for k, v in fields.items()},
    }
    index_path = shar_root / index_filename
    index_path.write_text(json.dumps(payload, indent=2))
    logger.info(f"Wrote merged index: {index_path} ({len(fields['cuts'])} cut shards)")


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


def main():
    parser = argparse.ArgumentParser(
        description="Convert standard WDS → Lhotse Shar (parallel)",
    )

    # Input
    parser.add_argument("--wds-shards", type=str, required=True,
                        help="Glob pattern for WDS tar shards")

    # Shar output
    parser.add_argument("--shar-dir", type=Path, required=True,
                        help="Output directory for Shar format")
    parser.add_argument("--shard-size", type=int, default=1000,
                        help="Samples per Shar shard (default: 1000)")
    parser.add_argument("--shar-format", type=str, default="flac",
                        choices=["flac", "wav", "mp3", "opus"],
                        help="Audio format in Shar (default: flac)")

    # Audio processing
    parser.add_argument("--target-sr", type=int, default=24000,
                        help="Target sample rate (default: 24000)")
    parser.add_argument("--min-sr", type=int, default=16000,
                        help="Drop audio below this sample rate (default: 16000)")

    # Text
    parser.add_argument("--text-ext", type=str, default=".txt",
                        help="Extension for text files in WDS tars (set to '' to skip)")

    # Parallelism
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of parallel workers (default: one per WDS shard)")

    args = parser.parse_args()

    resolved = sorted(glob.glob(args.wds_shards))
    if not resolved:
        raise FileNotFoundError(f"No files match pattern: {args.wds_shards}")

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

    worker_args = [
        (wid, shards, str(args.shar_dir), args.target_sr, args.shard_size, args.shar_format, args.min_sr, args.text_ext)
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

    logger.info(
        f"All workers done in {elapsed:.1f}s — "
        f"{total_written} samples, {total_skipped} skipped, {total_errors} errors"
    )

    # Build merged index for pipeline compatibility
    build_shar_index(args.shar_dir)
    (args.shar_dir / "_SUCCESS").write_text("ok\n")
    logger.info("All done!")


if __name__ == "__main__":
    main()
