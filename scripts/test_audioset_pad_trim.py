#!/usr/bin/env python3
"""
Sample non-10s AudioSet clips, pad to 10s, remove padding, trim tail tokens,
and write reconstructions for inspection.

Example:
  python scripts/test_audioset_pad_trim.py --split bal_train --num-samples 10
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset


@dataclass
class BucketEntry:
    youtube_id: str
    global_id: int
    bucket_len: int
    actual_len: int


def _load_bucket_entries(path: Path) -> list[BucketEntry]:
    entries: list[BucketEntry] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            entries.append(
                BucketEntry(
                    youtube_id=parts[0],
                    global_id=int(parts[1]),
                    bucket_len=int(parts[2]),
                    actual_len=int(parts[3]),
                )
            )
    return entries


def _sanitize_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Test pad+trim behavior on non-10s AudioSet clips.")
    parser.add_argument("--dataset-name", default="agkphysics/AudioSet")
    parser.add_argument("--config-name", default="full")
    parser.add_argument("--split", default="bal_train")
    parser.add_argument("--audio-field", default="audio")
    parser.add_argument(
        "--cache-dir",
        default="/capstor/store/cscs/swissai/infra01/audio-datasets/audioset_cache",
    )
    parser.add_argument(
        "--metadata-dir",
        default="/iopsstor/scratch/cscs/xyixuan/studys/multimodal-data/01-dataset-download/audio_set/length_buckets",
    )
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--target-sr", type=int, default=24000)
    parser.add_argument("--target-seconds", type=float, default=10.0)
    parser.add_argument("--trim-tokens", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--csv", default="outputs/audioset_non10_pad_trim5.csv")
    parser.add_argument("--offline", action="store_true", help="Force HF datasets offline mode.")

    args = parser.parse_args()

    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from audio_tokenizers.implementations.wavtokenizer import WavTokenizer40

    metadata_path = Path(args.metadata_dir) / f"audioset_{args.split}_buckets.tsv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"bucket metadata not found: {metadata_path}")

    target_len = int(round(args.target_seconds * args.target_sr))
    entries = _load_bucket_entries(metadata_path)
    entries = [
        e for e in entries
        if e.bucket_len != target_len and e.bucket_len < target_len
    ]
    if not entries:
        raise RuntimeError("no non-10s entries found in bucket metadata")

    rng = np.random.default_rng(args.seed)
    rng.shuffle(entries)
    selected = entries[: args.num_samples]

    indices = [e.global_id for e in selected]
    dataset = load_dataset(
        args.dataset_name,
        name=args.config_name,
        split=args.split,
        cache_dir=args.cache_dir,
    )
    dataset = dataset.select(indices)
    dataset = dataset.cast_column(args.audio_field, Audio(sampling_rate=args.target_sr))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    wt = WavTokenizer40(device=device, torch_compile=args.torch_compile)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, entry in enumerate(selected):
        sample = dataset[i]
        audio = sample[args.audio_field]["array"]
        sr = sample[args.audio_field]["sampling_rate"]
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        orig_len = int(audio.shape[0])
        if orig_len >= target_len:
            # Skip clips that are already 10s or longer.
            continue

        pad_len = target_len - orig_len
        padded = np.pad(audio, (0, pad_len), mode="constant")

        wave = torch.from_numpy(audio).unsqueeze(0)
        padded_wave = torch.from_numpy(padded).unsqueeze(0)

        with torch.inference_mode():
            tokens_orig, _ = wt.encode(wave, sr=sr)
            tokens_pad, _ = wt.encode(padded_wave, sr=sr)

        tokens_orig = tokens_orig.cpu()
        tokens_pad = tokens_pad.cpu()

        orig_tokens_len = int(tokens_orig.shape[-1])
        pad_tokens_len = int(tokens_pad.shape[-1])

        pad_trim = tokens_pad[..., :orig_tokens_len]
        diff_tokens = int((pad_trim != tokens_orig).sum().item())
        diff_pct = diff_tokens / orig_tokens_len * 100.0

        if orig_tokens_len <= args.trim_tokens:
            continue

        pad_trim5 = pad_trim[..., :-args.trim_tokens]
        tokens_orig_trim5 = tokens_orig[..., :-args.trim_tokens]
        diff_trim5_tokens = int((pad_trim5 != tokens_orig_trim5).sum().item())
        diff_trim5_pct = diff_trim5_tokens / tokens_orig_trim5.shape[-1] * 100.0

        with torch.inference_mode():
            recon_orig, _ = wt.decode(tokens_orig)
            recon_trim5, _ = wt.decode(pad_trim5)

        recon_orig = recon_orig.squeeze(0).cpu().numpy()
        recon_trim5 = recon_trim5.squeeze(0).cpu().numpy()

        tag = _sanitize_tag(entry.youtube_id)
        prefix = f"audioset_non10_{i:02d}_{tag}"
        orig_path = output_dir / f"{prefix}_orig.wav"
        recon_path = output_dir / f"{prefix}_recon.wav"
        trim5_path = output_dir / f"{prefix}_pad_recon_trim5.wav"

        sf.write(orig_path, audio, sr)
        sf.write(recon_path, recon_orig, sr)
        sf.write(trim5_path, recon_trim5, sr)

        rows.append({
            "index": i,
            "youtube_id": entry.youtube_id,
            "global_id": entry.global_id,
            "bucket_len": entry.bucket_len,
            "actual_len": entry.actual_len,
            "orig_samples": orig_len,
            "orig_seconds": orig_len / sr,
            "pad_samples": pad_len,
            "tokens_orig": orig_tokens_len,
            "tokens_pad": pad_tokens_len,
            "diff_tokens": diff_tokens,
            "diff_pct": diff_pct,
            "trim_tokens": args.trim_tokens,
            "diff_trim5_tokens": diff_trim5_tokens,
            "diff_trim5_pct": diff_trim5_pct,
            "orig_path": str(orig_path),
            "recon_path": str(recon_path),
            "pad_trim5_path": str(trim5_path),
        })

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote: {csv_path}")
    print(f"samples: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
