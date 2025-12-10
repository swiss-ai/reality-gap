#!/usr/bin/env python3
"""
Tokenize standardized LibriSpeech data with the AudioTokenizer + WavTokenizer backend.
Produces Megatron-ready IndexedDatasets (bin/idx/meta) plus optional HF token dumps.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import datasets
import numpy as np
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legacy_tokenizers import AudioMegatronWriter, AudioTokenizer, AudioTokenizerConfig


def parse_args():
    parser = argparse.ArgumentParser(description="LibriSpeech audio → discrete tokens")
    parser.add_argument(
        "--standardized-root",
        default="data/standardized/librispeech_clean",
        help="Directory containing the standardized HF datasets.",
    )
    parser.add_argument(
        "--tokenized-root",
        default="data/tokenized/librispeech_clean",
        help="Output path for Megatron-compatible IndexedDatasets.",
    )
    parser.add_argument(
        "--hf-tokenized-root",
        default="data/hf_tokenized/librispeech_clean",
        help="Output path for optional HF token datasets.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train.100", "validation", "test"],
        help="Split directory names to process under --standardized-root.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for WavTokenizer (e.g., cuda, cuda:1, cpu).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Process only the first N examples per split (useful for smoke tests).",
    )
    parser.add_argument(
        "--write-tokenized",
        action="store_true",
        help="Also persist HF token datasets (defaults to Megatron-only).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Optional override for log directory; defaults to <tokenized_root>/<split>/logs.",
    )
    parser.add_argument(
        "--stage-dir",
        type=str,
        default=None,
        help="Optional staging directory (e.g., $SCRATCH) before moving results into --tokenized-root.",
    )
    parser.add_argument(
        "--skip-megatron",
        action="store_true",
        help="Skip writing Megatron outputs (useful when only HF dumps are required).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Partition each split into N shards (dataset.shard).",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Shard index to process (0-based, < num-shards).",
    )
    parser.add_argument(
        "--log-name",
        default="tokenization_stats.json",
        help="Per-split log filename.",
    )
    return parser.parse_args()


def tokenize_example(tokenizer: AudioTokenizer, example: Dict[str, Any]) -> Dict[str, Any]:
    audio = example["audio"]
    audio_array = np.asarray(audio["array"], dtype=np.float32)
    sampling_rate = example["sampling_rate"]
    token_payload = tokenizer.tokenize_audio(audio_array, sampling_rate)
    return {
        "token_ids": token_payload["tokens"],
        "sequence_length": len(token_payload["tokens"]),
        "audio_token_length": token_payload["info"]["num_audio_tokens"],
        "tokenizer_info": token_payload["info"],
    }


def ensure_dir(path: Optional[Path]):
    if path is None:
        return
    path.mkdir(parents=True, exist_ok=True)


def process_split(split: str, args, tokenizer: AudioTokenizer):
    split_root = Path(args.standardized_root) / split
    if not split_root.exists():
        raise FileNotFoundError(f"{split_root} not found. Please standardize first.")

    dataset = datasets.load_from_disk(str(split_root))
    if args.max_examples is not None:
        take = min(args.max_examples, len(dataset))
        dataset = dataset.select(range(take))
        print(f"→ {split}: limiting to first {take} examples.")
    if args.num_shards > 1:
        if not 0 <= args.shard_id < args.num_shards:
            raise ValueError("--shard-id must be in [0, num_shards)")
        dataset = dataset.shard(num_shards=args.num_shards, index=args.shard_id)
        print(
            f"→ {split}: shard {args.shard_id}/{args.num_shards} contains {len(dataset)} examples."
        )

    base_output = Path(args.tokenized_root) / split
    tokens_dir = base_output / "tokens"
    logs_dir = (
        Path(args.log_dir) / split
        if args.log_dir
        else base_output / "logs"
    )

    stage_dir = None
    if args.stage_dir:
        stage_root = Path(args.stage_dir).expanduser()
        stage_dir = stage_root / f"{split}-{int(time.time())}"
        tokens_stage = stage_dir / "tokens" if not args.skip_megatron else None
        logs_stage = stage_dir / "logs"
    else:
        tokens_stage = tokens_dir if not args.skip_megatron else None
        logs_stage = logs_dir

    ensure_dir(tokens_stage)
    ensure_dir(logs_stage)

    writer: Optional[AudioMegatronWriter] = None
    if not args.skip_megatron and tokens_stage is not None:
        writer_prefix = str(tokens_stage / "tokens")
        writer = AudioMegatronWriter(writer_prefix, tokenizer.vocab_size)
    tokenized_records: List[Dict[str, Any]] = []

    lengths: List[int] = []
    audio_lengths: List[int] = []

    iterator = tqdm(dataset, desc=f"Tokenizing {split}", unit="ex")
    for example in iterator:
        payload = tokenize_example(tokenizer, example)
        if writer is not None:
            writer.add_sequence(payload["token_ids"])
        lengths.append(payload["sequence_length"])
        audio_lengths.append(payload["audio_token_length"])

        if args.write_tokenized:
            tokenized_records.append(
                {
                    "example_id": example.get("example_id"),
                    "token_ids": payload["token_ids"],
                    "sequence_length": payload["sequence_length"],
                    "audio_token_length": payload["audio_token_length"],
                    "tokenizer_info": json.dumps(payload["tokenizer_info"]),
                }
            )

    stats = {
        "split": split,
        "num_examples": len(lengths),
        "token_length": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "avg": (sum(lengths) / len(lengths)) if lengths else 0,
        },
        "audio_token_length": {
            "min": min(audio_lengths) if audio_lengths else 0,
            "max": max(audio_lengths) if audio_lengths else 0,
            "avg": (sum(audio_lengths) / len(audio_lengths)) if audio_lengths else 0,
        },
        "vocab_size": tokenizer.vocab_size,
        "timestamp": time.time(),
    }

    if writer is not None:
        writer.finalize(metadata=stats)

    log_path = logs_stage / args.log_name
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    if args.write_tokenized:
        tokenized_dataset = datasets.Dataset.from_list(tokenized_records)
        tokenized_dir = Path(args.hf_tokenized_root) / split
        tokenized_dir.mkdir(parents=True, exist_ok=True)
        tokenized_dataset.save_to_disk(str(tokenized_dir))

    if stage_dir:
        base_output.mkdir(parents=True, exist_ok=True)
        if not args.skip_megatron and tokens_stage is not None:
            if tokens_dir.exists():
                shutil.rmtree(tokens_dir)
            shutil.move(str(tokens_stage), str(tokens_dir))
        if logs_dir.exists():
            shutil.rmtree(logs_dir)
        shutil.move(str(logs_stage), str(logs_dir))
        shutil.rmtree(stage_dir, ignore_errors=True)

    print(
        f"✓ Split {split}: tokenized {len(lengths)} examples. Outputs in {base_output}"
    )


def main():
    args = parse_args()
    if args.skip_megatron and not args.write_tokenized:
        raise ValueError("At least one output target must be enabled (Megatron or HF).")
    config = AudioTokenizerConfig(device=args.device)
    tokenizer = AudioTokenizer(config)

    for split in args.splits:
        process_split(split, args, tokenizer)


if __name__ == "__main__":
    main()

