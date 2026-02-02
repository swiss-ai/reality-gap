#!/usr/bin/env python3
"""
Scan tokenized shards for low unique-token counts and optionally decode them.

Example:
  python scripts/check_silence_tokens.py \
    --dataset-dir /capstor/store/cscs/swissai/infra01/audio-datasets/tokenized/AudioSet_full_bal_train_audio_only_bucket_240000 \
    --unique-threshold 5 \
    --max-decode 10
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch

from audio_tokenization.utils.indexed_dataset_megatron import DType


def _load_idx(idx_path: Path):
    with idx_path.open("rb") as f:
        header = f.read(9)
        version = struct.unpack("<Q", f.read(8))[0]
        dtype_code = struct.unpack("<B", f.read(1))[0]
        sequence_count = struct.unpack("<Q", f.read(8))[0]
        document_count = struct.unpack("<Q", f.read(8))[0]
        seq_lengths = np.frombuffer(f.read(sequence_count * 4), dtype=np.int32)
        seq_ptrs = np.frombuffer(f.read(sequence_count * 8), dtype=np.int64)
        _ = np.frombuffer(f.read(document_count * 8), dtype=np.int64)
    return header, version, dtype_code, seq_lengths, seq_ptrs


def _iter_shards(dataset_dir: Path) -> Iterable[tuple[Path, Path]]:
    for idx_path in sorted(dataset_dir.glob("rank_*_shard_*_*.idx")):
        bin_path = idx_path.with_suffix(".bin")
        if bin_path.exists():
            yield idx_path, bin_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Find low-unique-token sequences and decode a few.")
    parser.add_argument(
        "--dataset-dir",
        default="/capstor/store/cscs/swissai/infra01/audio-datasets/tokenized/AudioSet_full_bal_train_audio_only_bucket_240000",
        help="Directory with rank_*_shard_*.bin/.idx files.",
    )
    parser.add_argument(
        "--omni-tokenizer-path",
        default=None,
        help="Path to omni tokenizer (auto-read from dataset_info.json if omitted).",
    )
    parser.add_argument("--unique-threshold", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit total sequences scanned.")
    parser.add_argument("--max-decode", type=int, default=10, help="Decode at most this many low-unique samples.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="outputs/silence_check")
    parser.add_argument("--csv", default="outputs/silence_check.csv")
    parser.add_argument("--torch-compile", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    import sys
    sys.path.insert(0, str(repo_root / "src"))

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset dir not found: {dataset_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from audio_tokenizers.implementations.wavtokenizer import WavTokenizer40
    from audio_tokenization.vokenizers.wavtokenizer.audio_only import WavTokenizerAudioOnly

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    wt = WavTokenizer40(device=device, torch_compile=args.torch_compile)
    if args.omni_tokenizer_path is None:
        info_path = dataset_dir / "dataset_info.json"
        if not info_path.exists():
            raise FileNotFoundError("dataset_info.json not found; pass --omni-tokenizer-path")
        info = json.loads(info_path.read_text())
        args.omni_tokenizer_path = info["tokenizer"]["path"]

    audio_only = WavTokenizerAudioOnly(
        omni_tokenizer_path=args.omni_tokenizer_path,
        device=device,
        torch_compile=args.torch_compile,
    )

    rows = []
    total_scanned = 0
    total_low_unique = 0
    decoded = 0

    for idx_path, bin_path in _iter_shards(dataset_dir):
        header, version, dtype_code, seq_lengths, seq_ptrs = _load_idx(idx_path)
        dtype = DType.dtype_from_code(dtype_code)
        itemsize = dtype().itemsize

        data = np.memmap(bin_path, dtype=dtype, mode="r")
        for i in range(len(seq_lengths)):
            if args.max_samples is not None and total_scanned >= args.max_samples:
                break
            total_scanned += 1

            length = int(seq_lengths[i])
            start = int(seq_ptrs[i] // itemsize)
            tokens = np.array(data[start:start + length], copy=False).astype(np.int64)
            if tokens.size < 4:
                continue
            # Strip BOS/audio_start/audio_end/EOS and de-offset audio tokens.
            audio_tokens = tokens[2:-2] - audio_only.audio_token_offset
            uniq = int(np.unique(audio_tokens).size)

            is_low = uniq <= args.unique_threshold
            if is_low:
                total_low_unique += 1

            rms = None
            out_path = ""
            if is_low and decoded < args.max_decode:
                decoded += 1
                tokens_t = torch.from_numpy(audio_tokens).unsqueeze(0)
                with torch.inference_mode():
                    recon, _ = wt.decode(tokens_t)
                recon = recon.squeeze(0).cpu().numpy()
                rms = float(np.sqrt(np.mean(recon ** 2)))
                out_path = output_dir / f"lowuniq_{idx_path.stem}_{i:06d}_u{uniq}_n{length}.wav"
                sf.write(out_path, recon, wt.output_sample_rate)
                out_path = str(out_path)

            rows.append({
                "idx_file": idx_path.name,
                "seq_index": i,
                "token_len": length,
                "unique_tokens": uniq,
                "is_low_unique": int(is_low),
                "decoded": int(bool(out_path)),
                "rms": rms,
                "output_path": out_path,
            })

        if args.max_samples is not None and total_scanned >= args.max_samples:
            break

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"scanned: {total_scanned}")
    print(f"low_unique (<= {args.unique_threshold}): {total_low_unique}")
    print(f"decoded: {decoded}")
    print(f"wrote: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
