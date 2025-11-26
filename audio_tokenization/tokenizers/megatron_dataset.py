from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

MEGATRON_SRC = (
    REPO_ROOT.parent
    / "src"
    / "repos"
    / "benchmark-image-tokenzier"
    / "vision_tokenization"
    / "pipelines"
)

if str(MEGATRON_SRC) not in sys.path:
    sys.path.insert(0, str(MEGATRON_SRC))

from indexed_dataset_megatron import (  # type: ignore
    DType,
    IndexedDatasetBuilder,
    get_bin_path,
    get_idx_path,
)


class AudioMegatronWriter:
    """Helper to write Megatron-compatible audio token datasets."""

    def __init__(self, output_prefix: str, vocab_size: int):
        self.output_prefix = output_prefix
        self.vocab_size = vocab_size
        self.dtype = DType.optimal_dtype(vocab_size)
        bin_path = get_bin_path(output_prefix)
        Path(bin_path).parent.mkdir(parents=True, exist_ok=True)
        self.builder = IndexedDatasetBuilder(bin_path, dtype=self.dtype)
        self.num_sequences = 0
        self.total_tokens = 0
        self.meta_path = Path(bin_path).with_suffix(".meta.json")

    def add_sequence(self, token_ids: Sequence[int]):
        tensor = torch.as_tensor(np.array(token_ids, dtype=np.int64), dtype=torch.int64)
        self.builder.add_item(tensor)
        self.num_sequences += 1
        self.total_tokens += len(token_ids)

    def finalize(self, metadata: Optional[Dict] = None):
        self.builder.finalize(get_idx_path(self.output_prefix))
        payload = metadata.copy() if metadata else {}
        payload.setdefault("num_sequences", self.num_sequences)
        payload.setdefault("total_tokens", self.total_tokens)
        payload.setdefault("vocab_size", self.vocab_size)
        payload.setdefault("dtype", np.dtype(self.dtype).name)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(
            f"✓ Wrote Megatron dataset: {self.output_prefix} "
            f"({self.num_sequences} sequences, {self.total_tokens} tokens)"
        )

