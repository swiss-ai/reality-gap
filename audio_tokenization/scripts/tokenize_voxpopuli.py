#!/usr/bin/env python3
"""
Tokenize standardized VoxPopuli data (parquet format) with the AudioTokenizer + WavTokenizer backend.
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

import numpy as np
import pyarrow.parquet as pq
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tokenizers import AudioMegatronWriter, AudioTokenizer, AudioTokenizerConfig


def parse_args():
    parser = argparse.ArgumentParser(description="VoxPopuli audio → discrete tokens")
    parser.add_argument(
        "--standardized-root",
        default="data/standardized/voxpopuli",
        help="Directory containing the standardized parquet files.",
    )
    parser.add_argument(
        "--tokenized-root",
        default="data/tokenized/voxpopuli",
        help="Output path for Megatron-compatible IndexedDatasets.",
    )
    parser.add_argument(
        "--hf-tokenized-root",
        default="data/hf_tokenized/voxpopuli",
        help="Output path for optional HF token datasets.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Language codes to process (e.g., cs de en). If not specified, processes all available.",
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
        help="Process only the first N examples per language (useful for smoke tests).",
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
        help="Optional override for log directory; defaults to <tokenized_root>/<lang>/logs.",
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
        help="Partition each language into N shards.",
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
        help="Per-language log filename.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of rows to read from parquet at a time (memory efficiency).",
    )
    return parser.parse_args()


def tokenize_row(tokenizer: AudioTokenizer, row: Dict[str, Any]) -> Dict[str, Any]:
    """Tokenize a single row from the parquet data."""
    # Audio is stored as a list of floats in the parquet
    audio_array = np.asarray(row["audio"], dtype=np.float32)
    sampling_rate = row["sampling_rate"]
    
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


def get_available_languages(standardized_root: Path) -> List[str]:
    """Detect available languages from parquet filenames."""
    parquet_files = list(standardized_root.glob("voxpopuli_*.parquet"))
    languages = []
    for f in parquet_files:
        # Extract language code from voxpopuli_<lang>.parquet
        lang = f.stem.replace("voxpopuli_", "")
        languages.append(lang)
    return sorted(languages)


def process_language(lang: str, args, tokenizer: AudioTokenizer):
    """Process a single language's parquet file."""
    parquet_path = Path(args.standardized_root) / f"voxpopuli_{lang}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"{parquet_path} not found.")

    # Open parquet file for efficient reading
    parquet_file = pq.ParquetFile(parquet_path)
    total_rows = parquet_file.metadata.num_rows
    
    # Handle sharding
    if args.num_shards > 1:
        if not 0 <= args.shard_id < args.num_shards:
            raise ValueError("--shard-id must be in [0, num_shards)")
        shard_size = total_rows // args.num_shards
        start_idx = args.shard_id * shard_size
        end_idx = start_idx + shard_size if args.shard_id < args.num_shards - 1 else total_rows
        print(f"→ {lang}: shard {args.shard_id}/{args.num_shards}, rows {start_idx}-{end_idx} ({end_idx - start_idx} examples)")
    else:
        start_idx = 0
        end_idx = total_rows

    # Apply max_examples limit
    if args.max_examples is not None:
        end_idx = min(start_idx + args.max_examples, end_idx)
        print(f"→ {lang}: limiting to first {end_idx - start_idx} examples.")

    num_to_process = end_idx - start_idx

    # Setup output directories
    base_output = Path(args.tokenized_root) / lang
    tokens_dir = base_output / "tokens"
    logs_dir = Path(args.log_dir) / lang if args.log_dir else base_output / "logs"

    stage_dir = None
    if args.stage_dir:
        stage_root = Path(args.stage_dir).expanduser()
        stage_dir = stage_root / f"{lang}-{int(time.time())}"
        tokens_stage = stage_dir / "tokens" if not args.skip_megatron else None
        logs_stage = stage_dir / "logs"
    else:
        tokens_stage = tokens_dir if not args.skip_megatron else None
        logs_stage = logs_dir

    ensure_dir(tokens_stage)
    ensure_dir(logs_stage)

    # Initialize writer
    writer: Optional[AudioMegatronWriter] = None
    if not args.skip_megatron and tokens_stage is not None:
        writer_prefix = str(tokens_stage / "tokens")
        writer = AudioMegatronWriter(writer_prefix, tokenizer.vocab_size)
    
    tokenized_records: List[Dict[str, Any]] = []
    lengths: List[int] = []
    audio_lengths: List[int] = []

    # Process in batches for memory efficiency
    processed = 0
    current_idx = 0
    
    with tqdm(total=num_to_process, desc=f"Tokenizing {lang}", unit="ex") as pbar:
        for batch in parquet_file.iter_batches(batch_size=args.batch_size):
            batch_df = batch.to_pandas()
            batch_size_actual = len(batch_df)
            
            # Skip rows before start_idx
            if current_idx + batch_size_actual <= start_idx:
                current_idx += batch_size_actual
                continue
            
            # Process rows in this batch
            for i, row in batch_df.iterrows():
                row_idx = current_idx + (i if isinstance(i, int) else batch_df.index.get_loc(i))
                
                # Skip if before start
                if row_idx < start_idx:
                    continue
                # Stop if past end
                if row_idx >= end_idx:
                    break
                
                try:
                    payload = tokenize_row(tokenizer, row.to_dict())
                    
                    if writer is not None:
                        writer.add_sequence(payload["token_ids"])
                    
                    lengths.append(payload["sequence_length"])
                    audio_lengths.append(payload["audio_token_length"])

                    if args.write_tokenized:
                        tokenized_records.append({
                            "example_id": row.get("example_id"),
                            "token_ids": payload["token_ids"],
                            "sequence_length": payload["sequence_length"],
                            "audio_token_length": payload["audio_token_length"],
                            "tokenizer_info": json.dumps(payload["tokenizer_info"]),
                            "language": lang,
                        })
                    
                    processed += 1
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"[ERROR] Failed on {row.get('example_id', 'unknown')}: {e}")
                    continue
            
            current_idx += batch_size_actual
            
            # Early exit if we've processed enough
            if processed >= num_to_process:
                break

    # Compile statistics
    stats = {
        "language": lang,
        "num_examples": len(lengths),
        "token_length": {
            "min": int(min(lengths)) if lengths else 0,
            "max": int(max(lengths)) if lengths else 0,
            "avg": float(sum(lengths) / len(lengths)) if lengths else 0,
        },
        "audio_token_length": {
            "min": int(min(audio_lengths)) if audio_lengths else 0,
            "max": int(max(audio_lengths)) if audio_lengths else 0,
            "avg": float(sum(audio_lengths) / len(audio_lengths)) if audio_lengths else 0,
        },
        "vocab_size": tokenizer.vocab_size,
        "timestamp": time.time(),
    }

    if writer is not None:
        writer.finalize(metadata=stats)

    # Write log
    log_path = logs_stage / args.log_name
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Write HF tokenized dataset if requested
    if args.write_tokenized:
        import datasets
        tokenized_dataset = datasets.Dataset.from_list(tokenized_records)
        tokenized_dir = Path(args.hf_tokenized_root) / lang
        tokenized_dir.mkdir(parents=True, exist_ok=True)
        tokenized_dataset.save_to_disk(str(tokenized_dir))

    # Move from staging if used
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

    print(f"✓ Language {lang}: tokenized {len(lengths)} examples. Outputs in {base_output}")


def main():
    args = parse_args()
    
    if args.skip_megatron and not args.write_tokenized:
        raise ValueError("At least one output target must be enabled (Megatron or HF).")
    
    # Initialize tokenizer
    config = AudioTokenizerConfig(device=args.device)
    tokenizer = AudioTokenizer(config)

    # Determine languages to process
    standardized_root = Path(args.standardized_root)
    available_languages = get_available_languages(standardized_root)
    
    if args.languages:
        languages = [l for l in args.languages if l in available_languages]
        missing = set(args.languages) - set(languages)
        if missing:
            print(f"[WARN] Languages not found: {missing}")
    else:
        languages = available_languages
    
    print(f"[INFO] Processing languages: {languages}")

    for lang in languages:
        process_language(lang, args, tokenizer)


if __name__ == "__main__":
    main()