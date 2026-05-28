#!/usr/bin/env python3
"""F0 diversity comparison: K=50 vs K=1 synth on matched ids.

Loads N matched-id pairs (same source utterance synthesized at K=1 and K=50),
extracts F0 via librosa.pyin (voicing-aware), pools voiced F0 across samples,
plots overlapping density histograms.

Publication-ready output: no title, 13pt fonts, vector PDF, K=1 vs K=50
labelled with σ only. fmin=80 Hz suppresses the pyin voicing-detection
floor artifact (anything <80 Hz is below human-speech F0 range — silence
or noise picked up as F0 candidates).
"""
import argparse, glob, io, os, random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import librosa
import matplotlib.pyplot as plt


def load_n_audio_by_id(parquet_dir: str, n: int, ids_filter: set | None = None):
    files = sorted(glob.glob(f"{parquet_dir}/*.parquet"))
    out = []
    for f in files:
        pf = pq.ParquetFile(f)
        table = pf.read(columns=["id", "audio"]).to_pylist()
        for r in table:
            if ids_filter is not None and r["id"] not in ids_filter:
                continue
            out.append((r["id"], r["audio"]["bytes"]))
            if len(out) >= n:
                return out
    return out


def f0_pool(samples, sr_target=22050, fmin=80, fmax=400) -> np.ndarray:
    """Pool voiced F0 values across samples. fmin=80 Hz suppresses the
    pyin voicing-detection floor (below human F0 range)."""
    all_f0 = []
    for sid, abytes in samples:
        try:
            audio, sr = sf.read(io.BytesIO(abytes))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != sr_target:
                audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=sr_target)
                sr = sr_target
            f0, voiced_flag, _ = librosa.pyin(
                audio.astype(np.float32), fmin=fmin, fmax=fmax, sr=sr,
                frame_length=2048, hop_length=512,
            )
            f0v = f0[voiced_flag]
            f0v = f0v[~np.isnan(f0v)]
            all_f0.append(f0v)
        except Exception as e:
            print(f"  WARN {sid}: {e}")
    return np.concatenate(all_f0) if all_f0 else np.array([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k50-dir", required=True)
    ap.add_argument("--k1-dir", required=True)
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--lang-label", default="pl")
    ap.add_argument("--out", required=True,
                    help="Output path; .pdf for vector (recommended for publication)")
    args = ap.parse_args()

    print(f"Sampling {args.n_samples} ids common to both sets...")
    k50_first = sorted(glob.glob(f"{args.k50_dir}/*.parquet"))[0]
    pf = pq.ParquetFile(k50_first)
    pool = pf.read(columns=["id"]).to_pylist()
    random.seed(42)
    random.shuffle(pool)
    candidate_ids = {r["id"] for r in pool[:args.n_samples * 3]}

    print(f"\nReading K=50 samples from {args.k50_dir}")
    k50 = load_n_audio_by_id(args.k50_dir, args.n_samples, candidate_ids)
    matched_ids = {sid for sid, _ in k50}
    print(f"  got {len(k50)} K=50 samples")

    print(f"\nReading K=1 samples from {args.k1_dir} (matching ids)")
    k1 = load_n_audio_by_id(args.k1_dir, args.n_samples, matched_ids)
    print(f"  got {len(k1)} K=1 samples")

    common = {sid for sid, _ in k50} & {sid for sid, _ in k1}
    k50 = [(s, a) for s, a in k50 if s in common]
    k1 = [(s, a) for s, a in k1 if s in common]
    print(f"\nMatched pairs: {len(common)}")

    print(f"\nExtracting F0 (K=50)...")
    f0_k50 = f0_pool(k50)
    print(f"  {len(f0_k50)} voiced frames; mean={np.mean(f0_k50):.1f} Hz, std={np.std(f0_k50):.1f}")

    print(f"\nExtracting F0 (K=1)...")
    f0_k1 = f0_pool(k1)
    print(f"  {len(f0_k1)} voiced frames; mean={np.mean(f0_k1):.1f} Hz, std={np.std(f0_k1):.1f}")

    # Publication style — larger fonts, no title, vector-friendly
    plt.rcParams.update({
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.dpi": 150,
    })

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    bins = np.linspace(80, 400, 60)
    ax.hist(f0_k1, bins=bins, alpha=0.55,
            label=f"K=1  (σ={np.std(f0_k1):.0f} Hz)",
            color="#d62728", density=True)
    ax.hist(f0_k50, bins=bins, alpha=0.55,
            label=f"K=50 (σ={np.std(f0_k50):.0f} Hz)",
            color="#1f77b4", density=True)
    ax.set_xlabel("Fundamental frequency F0 (Hz)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
