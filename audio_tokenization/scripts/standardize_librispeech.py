#!/usr/bin/env python3
"""
Normalize LibriSpeech clean splits into the shared audio schema.
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import datasets
import librosa
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="LibriSpeech → unified schema")
    parser.add_argument(
        "--output-root",
        default="data/standardized/librispeech_clean",
        help="Output directory for the unified dataset.",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Paths to HF `save_to_disk` directories (absolute or relative).",
    )
    parser.add_argument(
        "--dataset-label",
        default="librispeech",
        help="Value to store in the `dataset` field.",
    )
    parser.add_argument(
        "--resample",
        action="store_true",
        help="If set, audio is resampled to --target-sr.",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=16_000,
        help="Target sample rate when resampling.",
    )
    parser.add_argument(
        "--write-parquet",
        action="store_true",
        help="Also write a Parquet file per split.",
    )
    parser.add_argument(
        "--parquet-name",
        default="dataset.parquet",
        help="Parquet file name relative to each split directory.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="Number of worker processes for datasets.map (defaults to sequential).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Process only the first N examples of each split (handy for dry-runs).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip splits whose output directory already exists.",
    )
    parser.add_argument(
        "--stage-dir",
        type=str,
        default=None,
        help=(
            "Optional staging directory on fast local storage. Results are written "
            "there first and moved into --output-root once complete."
        ),
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Disable datasets' on-disk caching to reduce extra I/O on shared filesystems.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for datasets.map. Values >1 enable batched processing to reduce I/O.",
    )
    return parser.parse_args()


def build_schema(target_sr: Optional[int]):
    return datasets.Features(
        {
            "example_id": datasets.Value("string"),
            "dataset": datasets.Value("string"),
            "audio": datasets.Audio(sampling_rate=target_sr),
            "audio_path": datasets.Value("string"),
            "sampling_rate": datasets.Value("int32"),
            "duration": datasets.Value("float32"),
            "language": datasets.Value("string"),
            "speaker_id": datasets.Value("string"),
            "gender": datasets.Value("string"),
            "accent": datasets.Value("string"),
            "transcript": datasets.Value("string"),
            "transcript_type": datasets.Value("string"),
            "source_url": datasets.Value("string"),
            "metadata": datasets.Value("string"),
        }
    )


def _normalize_single(
    example, *, split: str, dataset_label: str, resample: bool, target_sr: int
) -> Dict[str, object]:
    audio = example["audio"]
    array = np.asarray(audio["array"], dtype=np.float32)
    sampling_rate = audio["sampling_rate"]

    if resample and sampling_rate != target_sr:
        array = librosa.resample(array, orig_sr=sampling_rate, target_sr=target_sr)
        sampling_rate = target_sr

    duration = np.float32(len(array) / sampling_rate)
    example_id = f"{dataset_label}_{example['id']}"

    metadata = {
        "chapter_id": str(example.get("chapter_id", "")),
        "split": split,
    }

    return {
        "example_id": example_id,
        "dataset": dataset_label,
        "audio": {"array": array, "sampling_rate": sampling_rate},
        "audio_path": str(example.get("file", "")),
        "sampling_rate": sampling_rate,
        "duration": float(duration),
        "language": "en",
        "speaker_id": str(example.get("speaker_id", "")),
        "gender": "unknown",
        "accent": "unknown",
        "transcript": "",
        "transcript_type": "none",
        "source_url": "",
        "metadata": json.dumps(metadata),
    }


def process_example(**kwargs):
    # Retained for compatibility when batch_size == 1.
    example = kwargs.pop("example")
    return _normalize_single(example, **kwargs)


def process_batch(
    batch, *, split: str, dataset_label: str, resample: bool, target_sr: int
) -> Dict[str, Iterable]:
    size = len(batch["audio"])
    normalized = []
    for idx in range(size):
        example = {key: batch[key][idx] for key in batch}
        normalized.append(
            _normalize_single(
                example,
                split=split,
                dataset_label=dataset_label,
                resample=resample,
                target_sr=target_sr,
            )
        )

    keys = normalized[0].keys() if normalized else build_schema(None).keys()
    return {key: [sample[key] for sample in normalized] for key in keys}


def standardize_split(split_path: str, args):
    raw_path = Path(split_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} not found. Please download/prep the split first.")

    split_name = raw_path.name
    output_dir = Path(args.output_root) / split_name
    if args.skip_existing and output_dir.exists():
        print(f"↷ {split_name}: found existing dataset in {output_dir}, skipping.")
        return

    dataset = datasets.load_from_disk(str(raw_path))
    if args.max_examples is not None:
        count = min(args.max_examples, len(dataset))
        dataset = dataset.select(range(count))
        print(f"→ {split_name}: limiting to first {count} examples for this run.")

    schema = build_schema(args.target_sr if args.resample else None)
    map_kwargs = {
        "fn_kwargs": {
            "split": split_name,
            "dataset_label": args.dataset_label,
            "resample": args.resample,
            "target_sr": args.target_sr,
        },
        "remove_columns": dataset.column_names,
        "features": schema,
        "desc": f"Standardizing {split_name}",
    }
    if args.num_proc:
        map_kwargs["num_proc"] = args.num_proc
    if args.batch_size and args.batch_size > 1:
        map_kwargs["batched"] = True
        map_kwargs["batch_size"] = args.batch_size
        map_fn = process_batch
    else:
        map_fn = lambda example, **fn_kwargs: process_example(
            example=example, **fn_kwargs
        )

    processed = dataset.map(map_fn, **map_kwargs)

    final_save_path = output_dir
    staging_dir = None
    if args.stage_dir:
        staging_root = Path(args.stage_dir).expanduser()
        staging_root.mkdir(parents=True, exist_ok=True)
        staging_dir = staging_root / f"{split_name}-{int(time.time())}-{os.getpid()}"
        final_save_path = staging_dir
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    try:
        processed.save_to_disk(str(final_save_path))
        if staging_dir is not None:
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            if output_dir.exists():
                shutil.rmtree(output_dir)
            shutil.move(str(staging_dir), str(output_dir))
    finally:
        if staging_dir is not None and staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

    if args.write_parquet:
        parquet_path = output_dir / args.parquet_name
        processed.to_parquet(str(parquet_path))

    print(f"✓ {split_name}: stored {len(processed)} examples in {output_dir}")


def main():
    args = parse_args()
    if args.disable_cache:
        datasets.disable_caching()

    for split_path in args.inputs:
        standardize_split(split_path, args)


if __name__ == "__main__":
    main()

