#!/usr/bin/env python3
"""Pull slow-paced reference clips from MLS Polish (audiobook reads).

Filters by characters-per-second to find natural-pace narrators. Saves
candidate WAVs + transcripts to an output dir for manual ear-test.

Polish speech rate reference:
    ~11-13 cps = slow/deliberate (audiobook narrator, good for TTS reference)
    ~14-16 cps = natural conversation
    ~17-20 cps = fast news anchor
    ~22+ cps  = very fast (likely current VoxCPM2 default behavior)
"""

import argparse
import random
from pathlib import Path

import soundfile as sf
from lhotse import CutSet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shar-in", required=True,
                   help="Lhotse Shar dir (e.g. MLS Polish train)")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Where to dump candidate WAVs + .txt transcripts")
    p.add_argument("--n-candidates", type=int, default=10)
    p.add_argument("--min-cps", type=float, default=11.0,
                   help="Minimum chars/sec (lower = slower)")
    p.add_argument("--max-cps", type=float, default=14.0,
                   help="Maximum chars/sec")
    p.add_argument("--min-duration", type=float, default=5.0)
    p.add_argument("--max-duration", type=float, default=12.0)
    p.add_argument("--min-text-chars", type=int, default=30,
                   help="Skip very short transcripts")
    p.add_argument("--scan-limit", type=int, default=2000,
                   help="Stop scanning after this many matches "
                        "(saves time on huge Shar dirs)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {args.shar_in} ...")
    cs = CutSet.from_shar(in_dir=args.shar_in)

    candidates = []
    scanned = 0
    for cut in cs:
        scanned += 1
        if not (args.min_duration <= cut.duration <= args.max_duration):
            continue
        sup = cut.supervisions[0] if cut.supervisions else None
        if not sup or not (sup.text or "").strip():
            continue
        text = sup.text.strip()
        if len(text) < args.min_text_chars:
            continue
        cps = len(text) / cut.duration
        if args.min_cps <= cps <= args.max_cps:
            candidates.append((cps, cut, text))
        if len(candidates) >= args.scan_limit:
            break

    print(f"Scanned {scanned} cuts, found {len(candidates)} matches "
          f"in [{args.min_cps}, {args.max_cps}] cps range")

    if not candidates:
        raise SystemExit("No candidates matched filters. Try widening the cps range.")

    random.seed(args.seed)
    random.shuffle(candidates)
    picked = candidates[:args.n_candidates]

    print(f"\nWriting {len(picked)} candidates to {args.output_dir}/")
    for i, (cps, cut, text) in enumerate(picked):
        wav = cut.load_audio()
        if wav.ndim > 1:
            wav = wav.mean(axis=0)
        sr = cut.sampling_rate
        name = f"mls_{i:02d}_cps{cps:.1f}_dur{cut.duration:.1f}s"
        sf.write(args.output_dir / f"{name}.wav", wav, sr, subtype="PCM_16")
        (args.output_dir / f"{name}.txt").write_text(text + "\n",
                                                    encoding="utf-8")
        print(f"  {name}.wav -- '{text[:70]}{'...' if len(text) > 70 else ''}'")

    print(f"\nDone. Listen and pick the one(s) you like the pace of.")


if __name__ == "__main__":
    main()
