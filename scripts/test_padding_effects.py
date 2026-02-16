#!/usr/bin/env python3
"""
Test how different padding values affect wavtokenizer tokens.

Example:
  python scripts/test_padding_effects.py \
    --input outputs/voxpopuli_60s.wav \
    --pad-seconds 0,0.5,1,2,5,10 \
    --modes zero,last,noise \
    --noise-std 1e-4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def _parse_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _load_audio(path: Path, target_sr: int, duration_sec: float | None, force_ffmpeg: bool) -> tuple[np.ndarray, int]:
    suffix = path.suffix.lower()
    use_ffmpeg = force_ffmpeg or suffix not in {".wav", ".flac"}

    if use_ffmpeg:
        ffmpeg = Path(os.environ.get("FFMPEG_PATH", "/users/xyixuan/.local/ffmpeg/bin/ffmpeg"))
        if not ffmpeg.exists():
            raise FileNotFoundError(f"ffmpeg not found at {ffmpeg}")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            cmd = [
                str(ffmpeg),
                "-y",
                "-v",
                "error",
                "-i",
                str(path),
                "-ac",
                "1",
                "-ar",
                str(target_sr),
            ]
            if duration_sec is not None:
                cmd += ["-t", str(duration_sec)]
            cmd.append(tmp_path)
            subprocess.check_call(cmd)
            audio, sr = sf.read(tmp_path, dtype="float32")
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    else:
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != target_sr:
            # Use ffmpeg for resample to avoid extra deps.
            return _load_audio(path, target_sr, duration_sec, force_ffmpeg=True)
        if duration_sec is not None:
            max_len = int(round(duration_sec * sr))
            audio = audio[:max_len]

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio, target_sr


def _make_pad(audio: np.ndarray, pad_len: int, mode: str, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    if pad_len <= 0:
        return np.zeros((0,), dtype=np.float32)
    if mode == "zero":
        return np.zeros(pad_len, dtype=np.float32)
    if mode == "last":
        last = audio[-1] if audio.size else 0.0
        return np.full(pad_len, last, dtype=np.float32)
    if mode == "noise":
        return rng.standard_normal(pad_len, dtype=np.float32) * noise_std
    if mode == "reflect":
        if audio.size == 0:
            return np.zeros(pad_len, dtype=np.float32)
        if pad_len <= audio.size:
            return audio[-pad_len:][::-1].copy()
        reps = int(np.ceil(pad_len / audio.size))
        tiled = np.tile(audio[::-1], reps)
        return tiled[:pad_len].copy()
    if mode == "repeat":
        if audio.size == 0:
            return np.zeros(pad_len, dtype=np.float32)
        reps = int(np.ceil(pad_len / audio.size))
        tiled = np.tile(audio, reps)
        return tiled[:pad_len].copy()
    raise ValueError(f"unknown pad mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test padding effects on wavtokenizer tokens.")
    parser.add_argument("--input", required=True, help="Input audio file path.")
    parser.add_argument("--target-sr", type=int, default=24000, help="Target sample rate.")
    parser.add_argument("--duration", type=float, default=None, help="Trim to N seconds before padding.")
    parser.add_argument("--pad-seconds", default="0,1,2,5,10", help="Comma-separated pad seconds.")
    parser.add_argument("--modes", default="zero,last,noise", help="Comma-separated pad modes.")
    parser.add_argument("--noise-std", type=float, default=1e-4, help="Std for noise pad.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for noise pad.")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto).")
    parser.add_argument("--torch-compile", action="store_true", help="Enable torch.compile in tokenizer.")
    parser.add_argument("--force-ffmpeg", action="store_true", help="Force ffmpeg decode.")
    parser.add_argument("--csv", default=None, help="Optional CSV output path.")

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from audio_tokenizers.implementations.wavtokenizer import WavTokenizer40

    audio_path = Path(args.input)
    audio, sr = _load_audio(audio_path, args.target_sr, args.duration, args.force_ffmpeg)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    wave = torch.from_numpy(audio).unsqueeze(0)
    wt = WavTokenizer40(device=device, torch_compile=args.torch_compile)

    with torch.inference_mode():
        base_tokens, _ = wt.encode(wave, sr=sr)

    base_len = base_tokens.numel()

    pad_seconds = [float(v) for v in _parse_list(args.pad_seconds)]
    modes = _parse_list(args.modes)

    rows = []
    for mode in modes:
        for pad_sec in pad_seconds:
            pad_len = int(round(pad_sec * sr))
            pad = _make_pad(audio, pad_len, mode, args.noise_std, rng)
            padded = np.concatenate([audio, pad], axis=0)
            padded_wave = torch.from_numpy(padded).unsqueeze(0)
            with torch.inference_mode():
                padded_tokens, _ = wt.encode(padded_wave, sr=sr)
            if padded_tokens.numel() < base_len:
                diff = None
                diff_pct = None
            else:
                slice_tokens = padded_tokens[..., :base_len]
                diff = int((slice_tokens != base_tokens).sum().item())
                diff_pct = diff / base_len * 100.0
            rows.append({
                "mode": mode,
                "pad_sec": pad_sec,
                "pad_len": pad_len,
                "base_len": base_len,
                "padded_len": padded_tokens.numel(),
                "diff": diff,
                "diff_pct": diff_pct,
            })

    # Print table
    print(f"device: {device}")
    print(f"base_tokens_len: {base_len}")
    print("| mode | pad_sec | pad_len | padded_len | diff | diff_pct |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        diff = "" if r["diff"] is None else str(r["diff"])
        diff_pct = "" if r["diff_pct"] is None else f"{r['diff_pct']:.2f}"
        print(f"| {r['mode']} | {r['pad_sec']:.2f} | {r['pad_len']} | {r['padded_len']} | {diff} | {diff_pct} |")

    if args.csv:
        import csv
        out_path = Path(args.csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
