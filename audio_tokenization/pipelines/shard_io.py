"""Shared shard writer helpers for Megatron indexed dataset output."""

import os
from pathlib import Path
from typing import Tuple

from audio_tokenization.utils.indexed_dataset import DType, IndexedDatasetBuilder


def open_shard_writer(
    output_dir: str,
    rank: int,
    shard_id: int,
    total_shards: int,
    vocab_size: int,
) -> Tuple[IndexedDatasetBuilder, str, str, str, str]:
    """Create shard writer and temporary/final output paths."""
    prefix = Path(output_dir) / f"rank_{rank}_shard_{shard_id}_{total_shards}"
    bin_path = str(prefix) + ".bin"
    idx_path = str(prefix) + ".idx"
    tmp_bin = bin_path + ".tmp"
    tmp_idx = idx_path + ".tmp"
    dtype = DType.optimal_dtype(vocab_size)
    builder = IndexedDatasetBuilder(tmp_bin, dtype=dtype)
    return builder, tmp_bin, tmp_idx, bin_path, idx_path


def finalize_shard_writer(
    builder: IndexedDatasetBuilder,
    tmp_bin: str,
    tmp_idx: str,
    bin_path: str,
    idx_path: str,
) -> None:
    """Finalize index and atomically move temporary shard files in place."""
    builder.finalize(tmp_idx)
    os.replace(tmp_bin, bin_path)
    os.replace(tmp_idx, idx_path)
