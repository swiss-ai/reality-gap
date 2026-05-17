#!/usr/bin/env python3
"""Pick N random synthesized cuts from a Shar output dir for spot-check.

Walks all shard_*/cuts.*.jsonl.gz, samples N cuts, optionally extracts
the matching audio files from their recording tars to --extract-to.
"""
import argparse
import glob
import gzip
import json
import random
import re
import tarfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shar-root", required=True, type=Path,
                   help="OUT_DIR from synthesize_to_shar (contains shard_*/)")
    p.add_argument("-n", "--n-picks", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--extract-to", type=Path, default=None,
                   help="If set, extract picked audio files (.wav/.flac) "
                        "from their recording.NNNNNN.tar into this dir.")
    args = p.parse_args()

    cuts_files = sorted(glob.glob(str(args.shar_root / "shard_*/cuts.*.jsonl.gz")))
    if not cuts_files:
        cuts_files = sorted(glob.glob(str(args.shar_root / "cuts.*.jsonl.gz")))
    if not cuts_files:
        raise SystemExit(f"No cuts.*.jsonl.gz under {args.shar_root}")

    all_cuts = []
    for cuts_path in cuts_files:
        with gzip.open(cuts_path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    c = json.loads(line)
                except json.JSONDecodeError:
                    continue
                all_cuts.append((float(c["duration"]), c["id"], cuts_path))

    random.seed(args.seed)
    random.shuffle(all_cuts)
    picks = sorted(all_cuts[:args.n_picks])

    if args.extract_to is None:
        for d, cid, cuts_path in picks:
            print(f"{d:.1f}\t{cid}\t{cuts_path}")
        return

    args.extract_to.mkdir(parents=True, exist_ok=True)
    AUDIO_EXTS = (".wav", ".flac", ".opus", ".ogg")
    for d, cid, cuts_path in picks:
        # recording.NNNNNN.tar lives next to cuts.NNNNNN.jsonl.gz
        m = re.match(r"cuts\.(\d+)\.jsonl\.gz$", Path(cuts_path).name)
        if not m:
            print(f"  WARN: can't parse shard id from {cuts_path}")
            continue
        rec_tar = Path(cuts_path).parent / f"recording.{m.group(1)}.tar"
        try:
            with tarfile.open(rec_tar, "r") as tf:
                # Match <cid>.<ext> for any audio extension
                found = None
                for mem in tf.getmembers():
                    name = mem.name
                    if name.startswith(f"{cid}.") and name.endswith(AUDIO_EXTS):
                        found = mem
                        break
                if found is None:
                    print(f"  WARN: {cid} not found in {rec_tar.name}")
                    continue
                with tf.extractfile(found) as src:
                    data = src.read()
                out_path = args.extract_to / Path(found.name).name
                out_path.write_bytes(data)
                print(f"  {out_path.name}  ({d:.1f}s)")
        except Exception as e:
            print(f"  ERROR opening {rec_tar}: {e}")


if __name__ == "__main__":
    main()
