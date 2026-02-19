#!/usr/bin/env python3
"""Convert a HuggingFace audio dataset to Lhotse Shar format.

Spawns N parallel workers via multiprocessing. Each worker:
  1. Loads the full HF dataset (arrow-backed, no audio decoding yet)
  2. Shards it by worker index (contiguous=False for even distribution)
  3. Wraps it as a Lhotse CutSet (byte-backed, audio decoded on demand)
  4. Writes its partition to ``part-{rank:05d}/`` via ``CutSet.to_shar()``

After all workers finish, builds a merged ``shar_index.json`` so that
downstream Lhotse code can load all partitions as a single CutSet.

Note: ``to_shar(num_jobs=1)`` is required because HF byte-backed cuts
contain raw ``bytes`` that are not JSON-serializable, and Lhotse's
``split_lazy()`` (used when num_jobs > 1) triggers JSON serialization.

Usage:
    python -m audio_tokenization.utils.prepare_data.prepare_hf_to_shar \
        --num_workers 20

    python -m audio_tokenization.utils.prepare_data.prepare_hf_to_shar \
        --dataset_name agkphysics/AudioSet \
        --dataset_split bal_train \
        --cache_dir /capstor/store/cscs/swissai/infra01/audio-datasets/audioset_cache \
        --num_workers 240
"""

import argparse
import logging
from multiprocessing import Process
from pathlib import Path

from audio_tokenization.utils.prepare_data.common import (
    SUCCESS_MARKER_FILE,
    build_shar_index_from_parts,
    load_text_tokenizer,
    make_text_tokenize_fn,
    mark_partition_success,
    setup_partition_dir,
    to_mono,
    validate_or_write_prepare_state,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PART_SUCCESS_MARKER = SUCCESS_MARKER_FILE
PREPARE_STATE_FILE = "_PREPARE_STATE.json"


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def convert_worker(rank: int, world_size: int, args, text_tokenizer=None):
    """Convert one shard of the HF dataset to Shar format.

    Each worker writes to its own ``part-{rank:05d}/`` directory.
    Reuses output only when ``part-*/_SUCCESS`` exists.
    """
    import datasets
    from lhotse import CutSet

    output_dir = args.shar_dir / f"part-{rank:05d}"
    if setup_partition_dir(
        output_dir,
        success_marker_name=PART_SUCCESS_MARKER,
        reuse_log=f"[worker {rank}] Reusing completed Shar in {output_dir}",
        reset_log=f"[worker {rank}] Removing partial Shar output in {output_dir}",
        logger=logger,
    ):
        return

    # Load HF dataset (arrow-backed, no audio decoding happens here)
    load_kwargs = {"split": args.dataset_split}
    if args.config_name:
        load_kwargs["name"] = args.config_name
    if args.cache_dir:
        load_kwargs["cache_dir"] = str(args.cache_dir)

    logger.info(f"[worker {rank}] Loading HF dataset...")
    ds = datasets.load_dataset(args.dataset_name, **load_kwargs)

    # Shard across workers (interleaved for balanced durations)
    ds = ds.shard(num_shards=world_size, index=rank, contiguous=False)
    logger.info(f"[worker {rank}] Shard has {len(ds)} examples")

    if len(ds) == 0:
        logger.warning(f"[worker {rank}] Empty shard, skipping")
        mark_partition_success(output_dir, success_marker_name=PART_SUCCESS_MARKER)
        return

    # Wrap as Lhotse CutSet (byte-backed: audio decoded lazily during to_shar)
    hf_kwargs = {"audio_key": args.audio_field}
    if args.text_field:
        hf_kwargs["text_key"] = args.text_field
    cuts = CutSet.from_huggingface_dataset(ds, **hf_kwargs)
    cuts = cuts.map(to_mono)

    if text_tokenizer is not None:
        cuts = cuts.map(make_text_tokenize_fn(text_tokenizer))

    if args.target_sample_rate:
        cuts = cuts.resample(args.target_sample_rate)

    cuts.to_shar(
        output_dir=str(output_dir),
        fields={"recording": args.shar_format},
        shard_size=args.shar_shard_size,
        num_jobs=1,
        verbose=(rank == 0),
    )
    mark_partition_success(output_dir, success_marker_name=PART_SUCCESS_MARKER)
    logger.info(f"[worker {rank}] Done → {output_dir}")


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_shar_index(shar_root: Path, index_filename: str, world_size: int):
    """Build a merged ``shar_index.json`` from all ``part-*`` directories.

    The index maps field names (``cuts``, ``recording``, ...) to sorted lists
    of absolute file paths, so that ``CutSet.from_shar(fields=...)`` can load
    all partitions as a single logical CutSet.
    """
    part_dirs = [shar_root / f"part-{rank:05d}" for rank in range(world_size)]
    index_path, cuts_count = build_shar_index_from_parts(
        shar_root=shar_root,
        part_dirs=part_dirs,
        index_filename=index_filename,
        success_marker_name=PART_SUCCESS_MARKER,
    )
    logger.info(f"Wrote merged index: {index_path} ({cuts_count} cut shards)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _validate_or_write_prepare_state(args) -> None:
    """Persist/validate resume-critical run config.

    HF sharding uses ``ds.shard(num_shards=num_workers, ...)``. Resuming with a
    different ``--num_workers`` changes worker-to-sample assignment and can
    produce overlaps/gaps if old partitions are reused.
    """
    state_path = args.shar_dir / PREPARE_STATE_FILE
    expected = {
        "dataset_name": args.dataset_name,
        "config_name": args.config_name,
        "dataset_split": args.dataset_split,
        "text_tokenizer": args.text_tokenizer,
        "num_workers": int(args.num_workers),
    }
    wrote = validate_or_write_prepare_state(
        state_path,
        expected=expected,
        invariant_keys=("dataset_name", "config_name", "dataset_split", "text_tokenizer", "num_workers"),
        guidance=(
            "Use the same dataset, --text_tokenizer, and --num_workers to resume this "
            f"output directory, or remove {args.shar_dir} and restart from scratch."
        ),
    )
    if wrote:
        logger.info(f"Wrote prepare state: {state_path}")


def main():
    parser = argparse.ArgumentParser(
        description="CPU-only parallel HF → Shar conversion",
    )

    # Dataset source
    parser.add_argument("--dataset_name", default="agkphysics/AudioSet")
    parser.add_argument("--config_name", default="full")
    parser.add_argument("--dataset_split", default="bal_train")
    parser.add_argument("--audio_field", default="audio")
    parser.add_argument("--text_field", default=None,
                        help="HF column for transcription text (e.g. 'sentence', 'text')")
    parser.add_argument("--cache_dir", type=Path,
                        default=Path("/capstor/store/cscs/swissai/infra01/audio-datasets/audioset_cache"))

    # Shar output
    parser.add_argument("--shar_base_dir", type=Path,
                        default=Path("/iopsstor/scratch/cscs/xyixuan/audio-datasets"))
    parser.add_argument("--shar_shard_size", type=int, default=1000)
    parser.add_argument("--shar_format", default="flac")
    parser.add_argument("--shar_index_filename", default="shar_index.json")

    # Audio processing
    parser.add_argument("--target_sample_rate", type=int, default=None)

    # Text tokenization
    parser.add_argument("--text_tokenizer", type=str, default=None,
                        help="Path to tokenizer.json for pre-tokenizing supervision text")

    # Parallelism
    parser.add_argument("--num_workers", type=int, default=20)

    args = parser.parse_args()

    # Derive output directory: agkphysics/AudioSet + unbal_train → audioset_unbal_train_shar
    name = args.dataset_name.rsplit("/", 1)[-1].lower()
    args.shar_dir = args.shar_base_dir / f"{name}_{args.dataset_split}_shar"
    args.shar_dir.mkdir(parents=True, exist_ok=True)
    _validate_or_write_prepare_state(args)

    logger.info(f"Converting {args.dataset_name} ({args.dataset_split}) → {args.shar_dir}")
    logger.info(f"Using {args.num_workers} parallel workers")

    # Load text tokenizer before forking (shared via COW across workers)
    text_tokenizer = load_text_tokenizer(args.text_tokenizer)

    # Spawn workers
    procs = [
        Process(target=convert_worker, args=(i, args.num_workers, args, text_tokenizer))
        for i in range(args.num_workers)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    failed = [i for i, p in enumerate(procs) if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"Workers {failed} failed")

    # Merge all part-* into a single index
    build_shar_index(args.shar_dir, args.shar_index_filename, args.num_workers)
    mark_partition_success(args.shar_dir, success_marker_name=PART_SUCCESS_MARKER)
    logger.info("All done!")


if __name__ == "__main__":
    main()
