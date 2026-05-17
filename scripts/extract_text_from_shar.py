#!/usr/bin/env python3
"""Read a Lhotse Shar dir → emit JSON text manifest for synthesize_to_shar.py.

Walks all `cuts.*.jsonl.gz` files (including multi-worker `worker_NN/`
subdirs) and pulls (id, text) from each cut's first supervision. Output
matches the schema synthesize_to_shar.py expects:
    [{"id": "...", "text": "...", "language": "pl"}, ...]
"""

import argparse
import glob
import gzip
import json
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shar-in", required=True, type=Path,
                   help="Lhotse Shar dir (supports multi-worker worker_NN/ layout)")
    p.add_argument("--output", required=True, type=Path,
                   help="JSON manifest output path")
    p.add_argument("--max-items", type=int, default=None,
                   help="Limit to first N items (after --shuffle if set)")
    p.add_argument("--max-hours", type=float, default=None,
                   help="Cap cumulative source duration at N hours "
                        "(after --shuffle if set). Useful for subsampling a "
                        "huge dataset like VoxPopuli pl down to a fixed "
                        "scale-test target.")
    p.add_argument("--shuffle", action="store_true",
                   help="Shuffle items (seeded) before truncating")
    p.add_argument("--language", default="pl")
    p.add_argument("--min-duration", type=float, default=1.0,
                   help="Skip cuts shorter than this many seconds")
    p.add_argument("--max-duration", type=float, default=30.0,
                   help="Skip cuts longer than this many seconds")
    p.add_argument("--polish-filter", action="store_true", default=True,
                   help="Drop items with no Polish-specific chars when "
                        "--language=pl (default on). --no-polish-filter disables.")
    p.add_argument("--no-polish-filter", dest="polish_filter",
                   action="store_false")
    args = p.parse_args()

    # Polish-specific characters used by --polish-filter (default on for pl).
    PL_CHARS = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")

    # Find cuts files. Multi-worker layout has worker_NN/cuts.*.jsonl.gz.
    cuts_files = sorted(glob.glob(str(args.shar_in / "**/cuts.*.jsonl.gz"),
                                  recursive=True))
    if not cuts_files:
        cuts_files = sorted(glob.glob(str(args.shar_in / "cuts.*.jsonl.gz")))
    if not cuts_files:
        raise SystemExit(f"No cuts.*.jsonl.gz files found under {args.shar_in}")
    print(f"Reading {len(cuts_files)} cuts files...")

    items = []
    skipped = 0
    skipped_non_pl = 0
    total_seconds = 0.0
    for cuts_path in cuts_files:
        with gzip.open(cuts_path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    cut = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                supervisions = cut.get("supervisions") or []
                if not supervisions:
                    skipped += 1
                    continue
                text = (supervisions[0].get("text") or "").strip()
                if not text:
                    skipped += 1
                    continue
                duration = float(cut.get("duration") or 0.0)
                if duration < args.min_duration or duration > args.max_duration:
                    skipped += 1
                    continue
                if args.language == "pl" and args.polish_filter and not any(
                        c in PL_CHARS for c in text):
                    skipped_non_pl += 1
                    continue
                items.append({
                    "id": cut["id"],
                    "text": text,
                    "language": args.language,
                    "source_duration": duration,
                })
                total_seconds += duration

    print(f"Found {len(items)} items "
          f"({total_seconds / 3600:.2f} h of source audio); "
          f"skipped {skipped} (filters), {skipped_non_pl} no Polish chars")

    if args.shuffle:
        random.seed(42)
        random.shuffle(items)

    if args.max_hours is not None:
        cap_seconds = args.max_hours * 3600.0
        kept = []
        running = 0.0
        for it in items:
            if running + it["source_duration"] > cap_seconds:
                continue
            kept.append(it)
            running += it["source_duration"]
            if running >= cap_seconds * 0.999:
                break
        items = kept
        print(f"Capped at {args.max_hours} h: "
              f"{len(items)} items, {running / 3600:.2f} h actual")

    if args.max_items:
        items = items[:args.max_items]
        kept_seconds = sum(it["source_duration"] for it in items)
        print(f"Truncated to {len(items)} items "
              f"({kept_seconds / 3600:.2f} h of source audio)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(items)} items to {args.output}")


if __name__ == "__main__":
    main()
