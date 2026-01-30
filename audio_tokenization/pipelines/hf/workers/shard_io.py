"""Shard writer helpers for Megatron indexed datasets."""

import os
from pathlib import Path
from typing import Tuple

from audio_tokenization.utils.indexed_dataset_megatron import IndexedDatasetBuilder, DType


def open_shard_writer(
    output_dir: str,
    worker_id: int,
    shard_id: int,
    total_shards: int,
    vocab_size: int,
) -> Tuple[IndexedDatasetBuilder, str, str, str, str]:
    output_prefix = Path(output_dir) / f"rank_{worker_id}_shard_{shard_id}_{total_shards}"
    bin_path = str(output_prefix) + ".bin"
    idx_path = str(output_prefix) + ".idx"
    tmp_bin_path = bin_path + ".tmp"
    tmp_idx_path = idx_path + ".tmp"
    dtype = DType.optimal_dtype(vocab_size)
    builder = IndexedDatasetBuilder(tmp_bin_path, dtype=dtype)
    return builder, tmp_bin_path, tmp_idx_path, bin_path, idx_path


def finalize_shard_writer(
    builder: IndexedDatasetBuilder,
    tmp_bin_path: str,
    tmp_idx_path: str,
    bin_path: str,
    idx_path: str,
) -> None:
    builder.finalize(tmp_idx_path)
    try:
        os.replace(tmp_bin_path, bin_path)
        os.replace(tmp_idx_path, idx_path)
    finally:
        for path in (tmp_bin_path, tmp_idx_path):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
