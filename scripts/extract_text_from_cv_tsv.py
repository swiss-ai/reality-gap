#!/usr/bin/env python3
"""Common Voice TSV → JSON text manifest for synthesize_to_shar.py.

Reads `validated.tsv` (CV's standard layout) and emits the same manifest
format as our other extractors:
    [{"id": "...", "text": "...", "language": "pl|zh|...", "source_duration": float}]

CV TSV columns (since CV ~v18): client_id, path, sentence_id, sentence,
sentence_domain, up_votes, down_votes, age, gender, accents, variant,
locale, segment.

CV doesn't ship clip durations in validated.tsv; we estimate via
`--cps` (chars/sec): Polish ~14, Mandarin ~5. The `clip_durations.tsv`
sibling file has exact durations if needed — we don't use it by default
since we don't actually need the source audio (we synth from text).
"""

import argparse
import csv
import json
import random
from pathlib import Path


# Per-language char predicates (same as extract_text_from_*).
PL_CHARS = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")
def _has_pl(t): return any(c in PL_CHARS for c in t)
def _has_cjk(t): return any("一" <= c <= "鿿" for c in t)
LANG_PRED = {"pl": _has_pl, "zh": _has_cjk, "none": lambda t: True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tsv", required=True, type=Path,
                   help="Path to CV validated.tsv (or any compatible TSV)")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--language", default="pl",
                   choices=["pl", "zh", "none"])
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--max-hours", type=float, default=None,
                   help="Cap cumulative estimated duration at N hours")
    p.add_argument("--min-chars", type=int, default=20,
                   help="Skip too-short sentences")
    p.add_argument("--max-chars", type=int, default=280,
                   help="Skip too-long sentences (~20s at default cps)")
    p.add_argument("--cps", type=float, default=None,
                   help="Chars/sec for duration estimate. Default: pl=14, zh=5.")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lang-filter", action="store_true", default=True,
                   help="Drop items not matching --language char set")
    p.add_argument("--no-lang-filter", dest="lang_filter",
                   action="store_false")
    p.add_argument("--skip-ids-from", type=Path, default=None,
                   help="JSON manifest of IDs to skip (dedup across extracts)")
    args = p.parse_args()

    if args.cps is None:
        args.cps = {"pl": 14.0, "zh": 5.0}.get(args.language, 14.0)

    pred = LANG_PRED.get(args.language)

    skip_ids = set()
    if args.skip_ids_from is not None:
        prior = json.load(open(args.skip_ids_from, "r", encoding="utf-8"))
        skip_ids = {it["id"] for it in prior}
        print(f"Loaded {len(skip_ids)} IDs to skip from {args.skip_ids_from}")

    print(f"Reading {args.tsv} ...")
    items = []
    skipped_short = skipped_long = skipped_non_lang = skipped_done = 0
    total_seconds = 0.0
    with open(args.tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = (row.get("sentence") or "").strip()
            if not text:
                continue
            if len(text) < args.min_chars:
                skipped_short += 1
                continue
            if len(text) > args.max_chars:
                skipped_long += 1
                continue
            if args.lang_filter and pred is not None and not pred(text):
                skipped_non_lang += 1
                continue
            # CV's `path` is unique per clip; use it as our ID (strip .mp3).
            cid = (row.get("path") or "").replace(".mp3", "")
            if cid in skip_ids:
                skipped_done += 1
                continue
            est_duration = len(text) / args.cps
            items.append({
                "id": cid,
                "text": text,
                "language": args.language,
                "source_duration": est_duration,
            })
            total_seconds += est_duration

    print(f"Kept {len(items)} items (~{total_seconds/3600:.1f} h estimated). "
          f"Skipped: {skipped_short} too short, {skipped_long} too long, "
          f"{skipped_non_lang} wrong language, {skipped_done} already done")

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
