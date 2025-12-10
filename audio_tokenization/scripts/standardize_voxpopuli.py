#!/usr/bin/env python3
"""
Standardize VoxPopuli raw audio into the unified audio schema and save per-language HF datasets.

Input layout (expected):
    <raw_root>/
        voxpopuli/
            raw_audios/
                <lang>/**/{*.flac,*.wav,*.ogg}

Output layout:
    <output_root>/<lang>/  (HF dataset via save_to_disk)
    optional: <output_root>/<lang>/dataset.parquet

Schema (matches README):
    example_id: str
    dataset: str
    audio: {"array": float32 ndarray, "sampling_rate": int}
    audio_path: str
    sampling_rate: int
    duration: float
    language: str
    speaker_id: str
    gender: str
    accent: str
    transcript: str
    transcript_type: str
    source_url: str
    metadata: str (JSON string, can be "{}")
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import datasets
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="VoxPopuli raw audio -> unified schema (per language)")
    parser.add_argument(
        "--raw-root",
        type=str,
        required=True,
        help="Root containing voxpopuli/raw_audios/<lang>/...",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/standardized/voxpopuli",
        help="Destination root; each language saved as HF dataset directory.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Languages to process (default: autodetect under raw_audios).",
    )
    parser.add_argument(
        "--dataset-label",
        type=str,
        default="voxpopuli",
        help="Value for the `dataset` field.",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=16_000,
        help="Resample target sampling rate.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit number of examples per language (for smoke tests).",
    )
    parser.add_argument(
        "--write-parquet",
        action="store_true",
        help="Also write dataset.parquet per language.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip language if output dir exists.",
    )
    return parser.parse_args()


def build_features(target_sr: int) -> datasets.Features:
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


def _load_and_resample(path: Path, target_sr: int) -> np.ndarray:
    audio_arr, sr = sf.read(path)
    if audio_arr.ndim > 1:
        audio_arr = audio_arr.mean(axis=1)  # mono
    if sr != target_sr:
        audio_arr = librosa.resample(audio_arr, orig_sr=sr, target_sr=target_sr)
    return np.asarray(audio_arr, dtype=np.float32), target_sr


def iter_audio_files(lang_dir: Path) -> Iterator[Path]:
    exts = (".flac", ".wav", ".ogg")
    for ext in exts:
        yield from lang_dir.rglob(f"*{ext}")


def example_generator(
    lang: str,
    lang_dir: Path,
    dataset_label: str,
    target_sr: int,
    limit: Optional[int] = None,
) -> Iterator[Dict]:
    files = list(iter_audio_files(lang_dir))
    if limit is not None:
        files = list(islice(files, limit))

    for path in tqdm(files, desc=f"[{lang}] standardizing", unit="audio"):
        try:
            audio_arr, sr = _load_and_resample(path, target_sr)
            duration = float(len(audio_arr) / sr)
            example_id = f"{dataset_label}_{path.stem}"

            yield {
                "example_id": example_id,
                "dataset": dataset_label,
                "audio": {"array": audio_arr, "sampling_rate": sr},
                "audio_path": str(path),
                "sampling_rate": sr,
                "duration": duration,
                "language": lang,
                "speaker_id": "",
                "gender": "unknown",
                "accent": "unknown",
                "transcript": "",
                "transcript_type": "none",
                "source_url": "",
                "metadata": json.dumps({}),
            }
        except Exception as e:
            print(f"[WARN] Failed on {path}: {e}")
            continue


def detect_languages(raw_root: Path) -> List[str]:
    vox_root = raw_root / "voxpopuli" / "raw_audios"
    langs = [p.name for p in vox_root.iterdir() if p.is_dir()]
    return sorted(langs)


def process_language(lang: str, args):
    raw_root = Path(args.raw_root)
    vox_raw = raw_root / "voxpopuli" / "raw_audios" / lang
    if not vox_raw.exists():
        raise FileNotFoundError(f"{vox_raw} not found")

    out_dir = Path(args.output_root) / lang
    if args.skip_existing and out_dir.exists():
        print(f"↷ {lang}: exists, skipping.")
        return

    features = build_features(args.target_sr)
    gen = lambda: example_generator(
        lang=lang,
        lang_dir=vox_raw,
        dataset_label=args.dataset_label,
        target_sr=args.target_sr,
        limit=args.max_examples,
    )
    ds = datasets.Dataset.from_generator(gen, features=features)

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))
    if args.write_parquet:
        ds.to_parquet(str(out_dir / "dataset.parquet"))

    print(f"✓ {lang}: stored {len(ds)} examples in {out_dir}")


def main():
    args = parse_args()
    if args.max_examples is not None and args.max_examples <= 0:
        args.max_examples = None

    langs = args.languages or detect_languages(Path(args.raw_root))
    if not langs:
        raise ValueError("No languages found under voxpopuli/raw_audios")

    print(f"[INFO] Languages: {langs}")
    for lang in langs:
        process_language(lang, args)


if __name__ == "__main__":
    main()

