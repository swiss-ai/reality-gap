#!/usr/bin/env python3
"""Filter silence/noise and merge indexed datasets into a single Megatron-compatible output.

Single-pass streaming: reads each chunk via mmap, applies a per-sequence unique-token
filter on the audio portion, and writes passing sequences directly to the output.
No Megatron dependency required — uses our own IndexedDatasetBuilder for writing and
a lightweight numpy mmap reader for reading.

Usage
-----
    python -m audio_tokenization.utils.indexed_dataset.filter_and_merge \
        --input-dirs /path/to/tokenized/dataset1 /path/to/tokenized/dataset2 \
        --output-prefix /path/to/output/merged_filtered \
        --min-unique-tokens 5 \
        --recursive
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from audio_tokenization.utils.indexed_dataset.indexed_dataset_megatron import (
    DType,
    IndexedDatasetBuilder,
    _INDEX_HEADER,
    get_bin_path,
    get_idx_path,
)
from audio_tokenization.utils.indexed_dataset.merge_indexed_dataset import (
    discover_indexed_prefixes,
)

# ---------------------------------------------------------------------------
# Lightweight mmap reader (no Megatron dependency)
# ---------------------------------------------------------------------------

class IndexedDatasetReader:
    """Read-only mmap access to a Megatron-format indexed dataset (.bin + .idx)."""

    def __init__(self, prefix: str) -> None:
        idx_path = prefix + ".idx"
        bin_path = prefix + ".bin"

        with open(idx_path, "rb") as f:
            header = f.read(9)
            assert header == _INDEX_HEADER, f"Bad header in {idx_path}"
            (version,) = struct.unpack("<Q", f.read(8))
            assert version == 1, f"Unsupported version {version}"
            (dtype_code,) = struct.unpack("<B", f.read(1))
            self.dtype = DType.dtype_from_code(dtype_code)
            self.itemsize = np.dtype(self.dtype).itemsize
            (self.num_sequences,) = struct.unpack("<Q", f.read(8))
            (self.num_documents,) = struct.unpack("<Q", f.read(8))

            # lengths: int32[num_sequences]
            self.lengths = np.frombuffer(
                f.read(self.num_sequences * 4), dtype=np.int32
            )
            # pointers: int64[num_sequences] — byte offsets into .bin
            self.pointers = np.frombuffer(
                f.read(self.num_sequences * 8), dtype=np.int64
            )

        self.bin_mmap = np.memmap(bin_path, dtype=self.dtype, mode="r")

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> np.ndarray:
        offset = self.pointers[idx] // self.itemsize
        length = self.lengths[idx]
        return self.bin_mmap[offset : offset + length]


# ---------------------------------------------------------------------------
# Core filter + merge
# ---------------------------------------------------------------------------

def _load_audio_token_offset(tokenizer_path: str) -> int:
    """Read audio_token_offset from audio_token_mapping.json."""
    mapping_path = os.path.join(tokenizer_path, "audio_token_mapping.json")
    if not os.path.isfile(mapping_path):
        raise FileNotFoundError(
            f"audio_token_mapping.json not found in {tokenizer_path}. "
            "Provide --tokenizer-path pointing to the omni-tokenizer directory."
        )
    with open(mapping_path, "r") as f:
        data = json.load(f)
    return data["audio_token_offset"]


def filter_and_merge(
    input_dirs: List[str],
    output_prefix: str,
    min_unique_tokens: int = 5,
    recursive: bool = False,
    force: bool = False,
    audio_token_offset: Optional[int] = None,
    stats_json: Optional[str] = None,
) -> dict:
    """Stream-filter and merge indexed datasets.

    Parameters
    ----------
    input_dirs : list of str
        Directories containing .bin/.idx pairs.
    output_prefix : str
        Output prefix (will create ``<prefix>.bin`` and ``<prefix>.idx``).
    min_unique_tokens : int
        Minimum number of unique codebook values in the audio content tokens
        to keep a sequence. Default 5.
    recursive : bool
        Scan input directories recursively.
    force : bool
        Overwrite existing output files.
    audio_token_offset : int
        First audio content token ID. Tokens with ID >= this value are
        audio content; everything below (BOS, EOS, audio_start,
        audio_end, RESERVED_OMNI, etc.) is excluded from unique count.
    stats_json : str, optional
        Path to write stats JSON.

    Returns
    -------
    dict
        Summary statistics.
    """
    if audio_token_offset is None:
        raise ValueError(
            "audio_token_offset is required. "
            "Use --tokenizer-path or --audio-token-offset."
        )
    output_prefix_path = Path(output_prefix).expanduser().resolve()
    if output_prefix_path.suffix in {".bin", ".idx"}:
        output_prefix_path = output_prefix_path.with_suffix("")
    output_prefix_path.parent.mkdir(parents=True, exist_ok=True)
    output_prefix_str = str(output_prefix_path)

    out_bin = Path(get_bin_path(output_prefix_str))
    out_idx = Path(get_idx_path(output_prefix_str))

    if not force and (out_bin.exists() or out_idx.exists()):
        raise FileExistsError(
            f"Output already exists: {out_bin} or {out_idx}. Use --force to overwrite."
        )
    if force:
        for p in (out_bin, out_idx):
            if p.exists():
                p.unlink()

    # Discover input prefixes
    prefixes, _ = discover_indexed_prefixes(input_dirs, recursive=recursive)
    # Exclude the output itself (in case output dir overlaps with input dir)
    prefixes = [p for p in prefixes if p != output_prefix_str]

    if not prefixes:
        raise RuntimeError("No valid input prefixes found (.bin + .idx).")

    # Map each prefix to its source directory for per-source stats.
    # Source = the top-level subdirectory under each input_dir, or the
    # input_dir name itself if the prefix sits directly inside it.
    resolved_input_dirs = [
        str(Path(d).expanduser().resolve()) for d in input_dirs
    ]

    def _source_for_prefix(prefix: str) -> str:
        """Identify the source dataset name for a prefix."""
        for d in resolved_input_dirs:
            if prefix.startswith(d + "/"):
                rel = prefix[len(d) + 1 :]
                # Use the first path component as the source name
                # e.g. "dataset_A/rank_0_shard_0_1" -> "dataset_A"
                parts = rel.split("/")
                if len(parts) > 1:
                    return parts[0]
                return Path(d).name
        return Path(prefix).parent.name

    # Per-source statistics
    source_stats: Dict[str, Dict[str, int]] = collections.OrderedDict()

    def _get_source(name: str) -> Dict[str, int]:
        if name not in source_stats:
            source_stats[name] = {
                "chunks": 0,
                "input_seqs": 0,
                "output_seqs": 0,
                "filtered_seqs": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "bytes_read": 0,
            }
        return source_stats[name]

    # Detect dtype from first chunk
    first_reader = IndexedDatasetReader(prefixes[0])
    dtype = first_reader.dtype
    del first_reader

    builder = IndexedDatasetBuilder(get_bin_path(output_prefix_str), dtype=dtype)

    # Global statistics
    total_input_seqs = 0
    total_output_seqs = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_filtered = 0
    total_bytes_read = 0
    t0 = time.time()

    print(f"Filter+Merge: {len(prefixes)} chunks, min_unique={min_unique_tokens}")
    print(f"Audio content tokens: ID >= {audio_token_offset}")
    print(f"Output: {output_prefix_str}")
    print(f"Dtype: {dtype.__name__}")
    print()

    for chunk_idx, prefix in enumerate(prefixes):
        reader = IndexedDatasetReader(prefix)
        src = _get_source(_source_for_prefix(prefix))
        src["chunks"] += 1
        chunk_kept = 0
        chunk_filtered = 0

        for i in range(len(reader)):
            seq = reader[i]
            seq_len = len(seq)
            total_input_seqs += 1
            total_input_tokens += seq_len
            src["input_seqs"] += 1
            src["input_tokens"] += seq_len

            # Extract audio content tokens by value range (>= offset),
            # excluding all special tokens (BOS, EOS, audio_start,
            # audio_end, RESERVED_OMNI, etc.)
            audio_tokens = seq[seq >= audio_token_offset]

            if len(audio_tokens) > 0 and len(np.unique(audio_tokens)) >= min_unique_tokens:
                builder.add_item(seq)
                builder.end_document()
                total_output_seqs += 1
                total_output_tokens += seq_len
                src["output_seqs"] += 1
                src["output_tokens"] += seq_len
                chunk_kept += 1
            else:
                total_filtered += 1
                src["filtered_seqs"] += 1
                chunk_filtered += 1

        # Track bytes read (bin file size)
        bin_size = os.path.getsize(prefix + ".bin")
        total_bytes_read += bin_size
        src["bytes_read"] += bin_size

        elapsed = time.time() - t0
        throughput_gbs = (total_bytes_read / 1e9) / max(elapsed, 1e-9)

        if (chunk_idx + 1) % 50 == 0 or (chunk_idx + 1) == len(prefixes):
            print(
                f"  [{chunk_idx + 1}/{len(prefixes)}] "
                f"kept={chunk_kept} filtered={chunk_filtered} "
                f"| total kept={total_output_seqs:,} filtered={total_filtered:,} "
                f"| {total_bytes_read / 1e9:.1f} GB read @ {throughput_gbs:.2f} GB/s"
            )

        # Release mmap
        del reader

    builder.finalize(get_idx_path(output_prefix_str))

    elapsed = time.time() - t0
    throughput_gbs = (total_bytes_read / 1e9) / max(elapsed, 1e-9)
    filter_pct = (total_filtered / max(total_input_seqs, 1)) * 100

    # Estimate audio hours: 1 audio token ≈ 40ms (25 Hz frame rate typical)
    # Adjust if your tokenizer has a different frame rate
    TOKEN_DURATION_S = 0.04
    input_audio_hours = (total_input_tokens * TOKEN_DURATION_S) / 3600
    output_audio_hours = (total_output_tokens * TOKEN_DURATION_S) / 3600

    out_bin_size = os.path.getsize(get_bin_path(output_prefix_str))
    out_idx_size = os.path.getsize(get_idx_path(output_prefix_str))

    stats = {
        "input_chunks": len(prefixes),
        "input_sequences": total_input_seqs,
        "input_tokens": total_input_tokens,
        "input_audio_hours_est": round(input_audio_hours, 1),
        "output_sequences": total_output_seqs,
        "output_tokens": total_output_tokens,
        "output_audio_hours_est": round(output_audio_hours, 1),
        "filtered_sequences": total_filtered,
        "filtered_pct": round(filter_pct, 2),
        "min_unique_tokens": min_unique_tokens,
        "audio_token_offset": audio_token_offset,
        "elapsed_s": round(elapsed, 1),
        "throughput_gbs": round(throughput_gbs, 3),
        "output_bin_bytes": out_bin_size,
        "output_idx_bytes": out_idx_size,
        "per_source": dict(source_stats),
    }

    print()
    print("=" * 60)
    print("Filter + Merge Complete")
    print("=" * 60)
    print(f"  Input:    {total_input_seqs:>12,} sequences, {total_input_tokens:>14,} tokens")
    print(f"  Output:   {total_output_seqs:>12,} sequences, {total_output_tokens:>14,} tokens")
    print(f"  Filtered: {total_filtered:>12,} sequences ({filter_pct:.2f}%)")
    print(f"  Audio:    ~{input_audio_hours:,.1f}h input -> ~{output_audio_hours:,.1f}h output")
    print(f"  Time:     {elapsed:.1f}s ({throughput_gbs:.2f} GB/s read)")
    print(f"  Output:   {out_bin_size / 1e9:.2f} GB .bin, {out_idx_size / 1e6:.1f} MB .idx")
    print(f"  Files:    {output_prefix_str}.{{bin,idx}}")

    # Per-source breakdown
    print()
    print("Per-source breakdown:")
    print(f"  {'Source':<50s} {'Chunks':>6s} {'Input seqs':>12s} {'Kept':>12s} {'Filtered':>10s} {'Filt%':>6s} {'Input tokens':>14s} {'Kept tokens':>14s}")
    print(f"  {'-'*50} {'-'*6} {'-'*12} {'-'*12} {'-'*10} {'-'*6} {'-'*14} {'-'*14}")
    for name, s in source_stats.items():
        src_filt_pct = (s["filtered_seqs"] / max(s["input_seqs"], 1)) * 100
        print(
            f"  {name:<50s} {s['chunks']:>6,} {s['input_seqs']:>12,} {s['output_seqs']:>12,} "
            f"{s['filtered_seqs']:>10,} {src_filt_pct:>5.1f}% {s['input_tokens']:>14,} {s['output_tokens']:>14,}"
        )

    if stats_json:
        stats_path = Path(stats_json).expanduser().resolve()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\n  Stats written to: {stats_path}")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Filter silence/noise and merge indexed datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        required=True,
        help="Directories containing .bin/.idx pairs.",
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Output prefix (creates <prefix>.bin and <prefix>.idx).",
    )
    parser.add_argument(
        "--min-unique-tokens",
        type=int,
        default=5,
        help="Minimum unique audio codebook values to keep a sequence (default: 5).",
    )

    # Audio token offset — either from tokenizer path or explicit value
    token_group = parser.add_mutually_exclusive_group(required=True)
    token_group.add_argument(
        "--tokenizer-path",
        help="Path to omni-tokenizer directory (reads audio_token_mapping.json).",
    )
    token_group.add_argument(
        "--audio-token-offset",
        type=int,
        help="First audio content token ID. Tokens >= this are audio content.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan input directories recursively for .bin/.idx pairs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--stats-json",
        help="Path to write stats JSON file.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Resolve audio token offset
    if args.tokenizer_path:
        audio_token_offset = _load_audio_token_offset(args.tokenizer_path)
    else:
        audio_token_offset = args.audio_token_offset

    stats = filter_and_merge(
        input_dirs=args.input_dirs,
        output_prefix=args.output_prefix,
        min_unique_tokens=args.min_unique_tokens,
        recursive=args.recursive,
        force=args.force,
        audio_token_offset=audio_token_offset,
        stats_json=args.stats_json,
    )
    return stats


if __name__ == "__main__":
    main()
