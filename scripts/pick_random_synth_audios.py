#!/usr/bin/env python3
"""Pick N random synthesized cuts from a Shar output dir for spot-check.

Walks all shard_*/cuts.*.jsonl.gz, samples N cuts, prints
"duration<TAB>cut_id<TAB>cuts_file_path" — sorted by duration so the
spread of clip lengths is obvious.
"""
import argparse
import glob
import gzip
import json
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shar-root", required=True, type=Path,
                   help="OUT_DIR from synthesize_to_shar (contains shard_*/)")
    p.add_argument("-n", "--n-picks", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
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
    for d, cid, cuts_path in picks:
        print(f"{d:.1f}\t{cid}\t{cuts_path}")


if __name__ == "__main__":
    main()
