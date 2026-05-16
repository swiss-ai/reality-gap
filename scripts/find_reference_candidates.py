#!/usr/bin/env python3
"""Pull slow-paced reference clips from a Lhotse Shar dir (e.g. MLS Polish).

Walks **/cuts.*.jsonl.gz recursively (handles Spark-prepared part-NNNNN/
layouts), filters cuts by chars/sec speaking rate, then extracts audio for
the picked candidates from the matching recording.*.tar.

Polish speech rate reference:
    ~11-13 cps = slow/deliberate (audiobook narrator, good for TTS reference)
    ~14-16 cps = natural conversation
    ~17-20 cps = fast news anchor
    ~22+ cps  = very fast
"""

import argparse
import gzip
import io
import json
import random
import re
import tarfile
from collections import defaultdict
from pathlib import Path

import soundfile as sf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shar-in", required=True, type=Path,
                   help="Lhotse Shar dir root (recurses into part-NNNNN/ etc.)")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Where to dump candidate WAVs + .txt transcripts")
    p.add_argument("--n-candidates", type=int, default=10)
    p.add_argument("--min-cps", type=float, default=11.0)
    p.add_argument("--max-cps", type=float, default=14.0)
    p.add_argument("--min-duration", type=float, default=5.0)
    p.add_argument("--max-duration", type=float, default=12.0)
    p.add_argument("--min-text-chars", type=int, default=30)
    p.add_argument("--scan-limit", type=int, default=2000,
                   help="Stop after this many matches")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find all cuts files. Each lives next to a matching recording.*.tar.
    cuts_files = sorted(args.shar_in.rglob("cuts.*.jsonl.gz"))
    if not cuts_files:
        raise SystemExit(f"No cuts.*.jsonl.gz under {args.shar_in}")
    print(f"Found {len(cuts_files)} cuts files. Scanning...")

    # candidates: list of (cps, cut_dict, rec_tar_path, text)
    candidates = []
    scanned = 0
    for cuts_path in cuts_files:
        # The matching recording tar is the same stem with "recording." prefix
        # and ".tar" extension. e.g. cuts.000003.jsonl.gz -> recording.000003.tar
        m = re.match(r"cuts\.(\d+)\.jsonl\.gz$", cuts_path.name)
        if not m:
            continue
        shard_id = m.group(1)
        rec_tar = cuts_path.parent / f"recording.{shard_id}.tar"
        if not rec_tar.exists():
            continue

        with gzip.open(cuts_path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                scanned += 1
                try:
                    cut = json.loads(line)
                except json.JSONDecodeError:
                    continue
                dur = float(cut.get("duration", 0.0))
                if not (args.min_duration <= dur <= args.max_duration):
                    continue
                supervisions = cut.get("supervisions") or []
                if not supervisions:
                    continue
                text = (supervisions[0].get("text") or "").strip()
                if len(text) < args.min_text_chars:
                    continue
                cps = len(text) / dur
                if args.min_cps <= cps <= args.max_cps:
                    candidates.append((cps, cut, rec_tar, text))
                if len(candidates) >= args.scan_limit:
                    break
        if len(candidates) >= args.scan_limit:
            break

    print(f"Scanned {scanned} cuts, kept {len(candidates)} in "
          f"[{args.min_cps}, {args.max_cps}] cps range")
    if not candidates:
        raise SystemExit("No matches. Try widening --min-cps/--max-cps.")

    random.seed(args.seed)
    random.shuffle(candidates)
    picked = candidates[:args.n_candidates]

    # Group by tar to avoid re-opening the same tar repeatedly.
    by_tar = defaultdict(list)
    for cps, cut, rec_tar, text in picked:
        by_tar[rec_tar].append((cps, cut, text))

    print(f"\nWriting {len(picked)} candidates to {args.output_dir}/")
    idx = 0
    for rec_tar, items in by_tar.items():
        # Build a name->member map for fast lookup. Audio in Shar is
        # typically <cut_id>.flac or <cut_id>.wav.
        try:
            with tarfile.open(rec_tar, "r") as tf:
                # Shar tars often contain both <id>.flac and <id>.json for
                # each cut. Only consider audio files when matching by stem.
                AUDIO_EXTS = {".flac", ".wav", ".opus", ".ogg", ".mp3"}
                name_map = {Path(m.name).stem: m for m in tf.getmembers()
                            if Path(m.name).suffix.lower() in AUDIO_EXTS}
                for cps, cut, text in items:
                    cut_id = cut["id"]
                    member = name_map.get(cut_id)
                    if member is None:
                        print(f"  WARN: {cut_id} not found in {rec_tar.name}")
                        continue
                    audio_bytes = tf.extractfile(member).read()
                    wav, sr = sf.read(io.BytesIO(audio_bytes))
                    if wav.ndim > 1:
                        wav = wav.mean(axis=1)
                    name = f"mls_{idx:02d}_cps{cps:.1f}_dur{cut['duration']:.1f}s"
                    sf.write(args.output_dir / f"{name}.wav", wav, sr,
                             subtype="PCM_16")
                    (args.output_dir / f"{name}.txt").write_text(
                        text + "\n", encoding="utf-8")
                    print(f"  {name}.wav -- "
                          f"'{text[:70]}{'...' if len(text) > 70 else ''}'")
                    idx += 1
        except Exception as e:
            print(f"  ERROR opening {rec_tar}: {e}")

    print(f"\nDone. SCP from {args.output_dir}/ and listen.")


if __name__ == "__main__":
    main()
