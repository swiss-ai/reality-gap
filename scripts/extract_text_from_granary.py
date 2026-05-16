#!/usr/bin/env python3
"""Read a Granary JSONL manifest → emit JSON text manifest for synthesize_to_shar.py.

Granary segments are pre-chunked (utt_id format: <session>_pl_<idx>), so no
sentence-splitting needed — each line is one TTS-ready item. Filters by
text-length proxy for duration (Polish ~14 chars/sec).
"""

import argparse
import json
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--granary-jsonl", required=True, type=Path,
                   help="Granary ASR manifest, e.g. .../voxpopuli/pl_asr.jsonl")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--language", default="pl")
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--max-hours", type=float, default=None,
                   help="Cap cumulative duration (estimated from char count)")
    p.add_argument("--min-chars", type=int, default=50,
                   help="≈ 3.5s at 14 chars/sec — skip too-short clips")
    p.add_argument("--max-chars", type=int, default=450,
                   help="≈ 32s — skip too-long clips that may hurt VoxCPM2")
    p.add_argument("--cps", type=float, default=14.0,
                   help="Polish chars/sec used to estimate duration from text")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"Reading {args.granary_jsonl} ...")
    items = []
    skipped_short = 0
    skipped_long = 0
    skipped_other = 0
    total_seconds = 0.0
    with open(args.granary_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped_other += 1
                continue
            text = (rec.get("text") or rec.get("answer") or "").strip()
            if not text:
                skipped_other += 1
                continue
            if len(text) < args.min_chars:
                skipped_short += 1
                continue
            if len(text) > args.max_chars:
                skipped_long += 1
                continue
            est_duration = len(text) / args.cps
            items.append({
                "id": rec.get("utt_id") or rec.get("original_source_id"),
                "text": text,
                "language": args.language,
                "source_duration": est_duration,
            })
            total_seconds += est_duration

    print(f"Kept {len(items)} items (~{total_seconds/3600:.1f} h estimated). "
          f"Skipped: {skipped_short} too short, {skipped_long} too long, "
          f"{skipped_other} no text")

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(items)

    if args.max_hours is not None:
        cap = args.max_hours * 3600.0
        kept = []
        running = 0.0
        for it in items:
            if running + it["source_duration"] > cap:
                continue
            kept.append(it)
            running += it["source_duration"]
            if running >= cap * 0.999:
                break
        items = kept
        print(f"Capped at {args.max_hours} h: {len(items)} items, "
              f"{running/3600:.2f} h")

    if args.max_items:
        items = items[:args.max_items]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(items)} items to {args.output}")


if __name__ == "__main__":
    main()
