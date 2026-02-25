#!/usr/bin/env python3
"""Convert HF-style parquet shards (with audio bytes) to Lhotse Shar format.

Designed for the SPC-R (Speech Parliament Corpus) dataset which ships as ~130
HuggingFace parquet shards. Each row has:
- ``id``:       ``row{NNNNN}_seg{NNN}`` (source + segment)
- ``duration``: float (some rows have <=0 duration with 0 audio bytes)
- ``audio``:    struct ``{bytes: Binary, sampling_rate: Int64}`` (FLAC at 16kHz)
- ``text``:     str (transcription)

Workers are assigned *whole parquet files* (not rows). With 130 files / 20
workers, each worker handles ~6-7 files (~1001 rows each), giving good balance.

Usage (SPC-R):
    python -m audio_tokenization.utils.prepare_data.prepare_parquet_to_shar \
        --parquet-dir /capstor/store/cscs/swissai/infra01/audio-datasets/raw/spc-r-segmented/train \
        --shar-dir /capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/spc_r_segmented_shar \
        --target-sr 24000 \
        --shard-size 2000 \
        --shar-format flac \
        --text-tokenizer /capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json \
        --num-workers 20
"""

import argparse
from collections import Counter
import logging
import time
from pathlib import Path

from audio_tokenization.utils.prepare_data.common import (
    PREPARE_STATE_FILE,
    check_worker_reuse,
    distribute_round_robin,
    ensure_worker_assignment,
    init_worker_process,
    load_text_tokenizer,
    make_text_tokenize_fn,
    normalize_optional_path,
    run_pool_and_finalize,
    to_mono,
    validate_or_write_prepare_state,
    write_worker_result,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(processName)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

_ITEMS_KEY = "resolved_parquets"


def _convert_worker(args_tuple):
    """Convert rows from assigned parquet files to Shar.

    Each worker writes to its own ``worker_XX/`` directory. Resume is complete
    only when ``worker_XX/_SUCCESS`` exists.
    """
    (
        worker_id,
        parquet_paths,
        shar_dir,
        target_sr,
        shard_size,
        shar_format,
        id_column,
        audio_column,
        text_column,
        duration_column,
        text_tokenizer,
        resampling_backend,
    ) = args_tuple

    reused = check_worker_reuse(worker_id, shar_dir)
    if reused is not None:
        return reused
    init_worker_process(resampling_backend)

    import polars as pl
    from lhotse import Recording, SupervisionSegment
    from lhotse.shar import SharWriter

    worker_dir = Path(shar_dir) / f"worker_{worker_id:02d}"
    t0 = time.time()
    written = skipped = errors = 0
    total_duration_sec = 0.0
    runtime_counts = Counter()
    _tokenize_text = make_text_tokenize_fn(text_tokenizer) if text_tokenizer is not None else None

    with SharWriter(
        output_dir=str(worker_dir),
        fields={"recording": shar_format},
        shard_size=shard_size,
    ) as writer:
        for pq_path in parquet_paths:
            pq_name = Path(pq_path).name
            logger.info(f"Worker {worker_id}: reading {pq_name}")
            df = pl.read_parquet(pq_path)

            for row in df.iter_rows(named=True):
                row_id = row[id_column]
                try:
                    # Filter bad rows
                    duration = row.get(duration_column)
                    if duration is not None and duration <= 0:
                        skipped += 1
                        runtime_counts["skipped_non_positive_duration"] += 1
                        continue

                    audio_struct = row[audio_column]
                    audio_bytes = audio_struct["bytes"] if isinstance(audio_struct, dict) else None
                    if not audio_bytes:
                        skipped += 1
                        runtime_counts["skipped_empty_audio"] += 1
                        continue

                    recording = Recording.from_bytes(
                        data=audio_bytes, recording_id=row_id,
                    )
                    cut = recording.to_cut()

                    # Attach supervision with text
                    text = row.get(text_column)
                    if text:
                        cut.supervisions = [SupervisionSegment(
                            id=cut.id,
                            recording_id=cut.recording_id,
                            start=0.0,
                            duration=cut.duration,
                            text=text,
                        )]

                    # Resample if needed
                    if target_sr and cut.sampling_rate != target_sr:
                        cut = cut.resample(target_sr)
                        runtime_counts["resampled"] += 1

                    # Mono
                    cut = to_mono(cut, mono_downmix=True, stats=runtime_counts)

                    # Pre-tokenize text
                    if _tokenize_text is not None:
                        cut = _tokenize_text(cut)

                    writer.write(cut)
                    written += 1
                    total_duration_sec += cut.duration

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
                        logger.warning(f"Worker {worker_id} error on {row_id}: {e}")

    return write_worker_result(
        worker_id=worker_id, worker_dir=worker_dir,
        written=written, skipped=skipped, errors=errors,
        total_duration_sec=total_duration_sec,
        runtime_counts=runtime_counts, t0=t0,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _validate_or_write_prepare_state(args) -> None:
    state_path = args.shar_dir / PREPARE_STATE_FILE
    expected = {
        "parquet_dir": str(Path(args.parquet_dir).resolve()),
        "text_tokenizer": normalize_optional_path(args.text_tokenizer),
    }
    wrote = validate_or_write_prepare_state(
        state_path,
        expected=expected,
        invariant_keys=("parquet_dir", "text_tokenizer"),
        guidance=(
            "Use the same --parquet-dir and --text-tokenizer to resume this "
            f"output directory, or remove {args.shar_dir} and restart from scratch."
        ),
    )
    if wrote:
        logger.info(f"Wrote prepare state: {state_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert HF parquet shards → Lhotse Shar (parallel)",
    )

    # Input
    parser.add_argument("--parquet-dir", type=Path, required=True,
                        help="Directory containing parquet files")
    parser.add_argument("--parquet-glob", type=str, default="*.parquet",
                        help="Glob pattern for parquet files (default: '*.parquet')")

    # Shar output
    parser.add_argument("--shar-dir", type=Path, required=True,
                        help="Output directory for Shar format")
    parser.add_argument("--shard-size", type=int, default=2000,
                        help="Samples per Shar shard (default: 2000)")
    parser.add_argument("--shar-format", type=str, default="flac",
                        choices=["flac", "wav", "mp3", "opus"],
                        help="Audio format in Shar (default: flac)")

    # Audio processing
    parser.add_argument("--target-sr", type=int, default=24000,
                        help="Target sample rate (default: 24000)")
    parser.add_argument("--resampling-backend", type=str, default=None,
                        choices=["default", "sox"],
                        help="Lhotse resampling backend override (default: use "
                             "$LHOTSE_RESAMPLING_BACKEND or 'default')")

    # Column names
    parser.add_argument("--id-column", type=str, default="id",
                        help="Column name for row ID (default: 'id')")
    parser.add_argument("--audio-column", type=str, default="audio",
                        help="Column name for audio struct (default: 'audio')")
    parser.add_argument("--text-column", type=str, default="text",
                        help="Column name for transcription text (default: 'text')")
    parser.add_argument("--duration-column", type=str, default="duration",
                        help="Column name for duration (default: 'duration')")

    # Text tokenizer
    parser.add_argument("--text-tokenizer", type=str, default=None,
                        help="Path to tokenizer.json for pre-tokenizing supervision text")

    # Parallelism
    parser.add_argument("--num-workers", type=int, default=20,
                        help="Number of parallel workers (default: 20)")

    args = parser.parse_args(argv)

    # Resolve parquet files
    resolved = sorted(str(p) for p in args.parquet_dir.glob(args.parquet_glob))
    if not resolved:
        raise FileNotFoundError(
            f"No files match {args.parquet_dir / args.parquet_glob}"
        )

    args.shar_dir.mkdir(parents=True, exist_ok=True)
    _validate_or_write_prepare_state(args)

    num_workers = ensure_worker_assignment(
        args.shar_dir, resolved, args.num_workers, _ITEMS_KEY, "parquet files",
    )

    logger.info(f"Found {len(resolved)} parquet files, using {num_workers} workers")
    logger.info(f"Output: {args.shar_dir}")

    # Distribute parquet files across workers (round-robin)
    worker_parquets = distribute_round_robin(resolved, num_workers)

    # Load text tokenizer before forking (shared via COW across workers)
    text_tokenizer = load_text_tokenizer(args.text_tokenizer)

    worker_args = [
        (
            wid,
            parquets,
            str(args.shar_dir),
            args.target_sr,
            args.shard_size,
            args.shar_format,
            args.id_column,
            args.audio_column,
            args.text_column,
            args.duration_column,
            text_tokenizer,
            args.resampling_backend,
        )
        for wid, parquets in enumerate(worker_parquets)
        if parquets
    ]

    run_pool_and_finalize(_convert_worker, worker_args, args.shar_dir, num_workers)


if __name__ == "__main__":
    main()
