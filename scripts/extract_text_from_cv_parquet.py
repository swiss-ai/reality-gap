#!/usr/bin/env python3
"""Common Voice 25+ processed parquet → JSON text manifest for synth.

CV25's processed parquet layout (cluster path: processed/commonvoice25/<lang>/)
has the same columns as the TSV (path, sentence, ...) but rows in parquet
format with audio bytes inlined. We only need TEXT for synthesis — we
don't decode the audio at all.

Reads any subset of the per-split files (train/validated_extra/dev/test/other),
applies language char filter, text length filters, and dumps the standard
manifest schema other scripts/synthesize_to_shar.py consume:
    [{"id": "...", "text": "...", "language": "...", "source_duration": float}]
"""

import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq


PL_CHARS = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")
def _has_pl(t): return any(c in PL_CHARS for c in t)
def _has_cjk(t): return any("一" <= c <= "鿿" for c in t)
LANG_PRED = {"pl": _has_pl, "zh": _has_cjk, "none": lambda t: True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cv-dir", required=True, type=Path,
                   help="CV25 lang dir (contains train.parquet, validated_extra.parquet, ...)")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--language", default="pl", choices=["pl", "zh", "none"])
    p.add_argument("--splits", nargs="+",
                   default=["train", "validated_extra", "dev", "test", "other"],
                   help="Which parquet splits to ingest (skip 'invalidated' by default).")
    p.add_argument("--text-column", default="sentence",
                   help="Column name with the transcript")
    p.add_argument("--id-column", default="path",
                   help="Column name for unique clip ID")
    p.add_argument("--min-chars", type=int, default=None,
                   help="Default: pl=20, zh=5")
    p.add_argument("--max-chars", type=int, default=280)
    p.add_argument("--cps", type=float, default=None,
                   help="Chars/sec for duration estimate. Default: pl=14, zh=5.")
    p.add_argument("--max-hours", type=float, default=None)
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lang-filter", action="store_true", default=True)
    p.add_argument("--no-lang-filter", dest="lang_filter", action="store_false")
    p.add_argument("--skip-ids-from", type=Path, default=None,
                   help="JSON manifest of IDs to skip (dedup across extracts)")
    args = p.parse_args()

    if args.min_chars is None:
        args.min_chars = {"pl": 20, "zh": 5}.get(args.language, 20)
    if args.cps is None:
        args.cps = {"pl": 14.0, "zh": 5.0}.get(args.language, 14.0)

    pred = LANG_PRED.get(args.language)

    skip_ids = set()
    if args.skip_ids_from is not None:
        prior = json.load(open(args.skip_ids_from, "r", encoding="utf-8"))
        skip_ids = {it["id"] for it in prior}
        print(f"Loaded {len(skip_ids)} IDs to skip from {args.skip_ids_from}")

    items = []
    seen_ids = set()
    skipped_short = skipped_long = skipped_non_lang = skipped_done = skipped_dup = 0
    total_seconds = 0.0

    for split in args.splits:
        pq_path = args.cv_dir / f"{split}.parquet"
        if not pq_path.exists():
            print(f"  Skip missing: {pq_path}")
            continue
        print(f"Reading {pq_path.name} ...")
        pf = pq.ParquetFile(pq_path)
        # Stream row-groups; only project the two text columns we need.
        try:
            tbl = pf.read(columns=[args.id_column, args.text_column])
        except Exception as e:
            print(f"  WARN: failed to read {pq_path.name}: {e}")
            continue
        ids = tbl.column(args.id_column).to_pylist()
        texts = tbl.column(args.text_column).to_pylist()
        for cid, text in zip(ids, texts):
            if not text:
                continue
            text = str(text).strip()
            cid = str(cid or "").replace(".mp3", "")
            if not cid or not text:
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
            if cid in skip_ids:
                skipped_done += 1
                continue
            if cid in seen_ids:
                skipped_dup += 1
                continue
            seen_ids.add(cid)
            est_duration = len(text) / args.cps
            items.append({
                "id": cid,
                "text": text,
                "language": args.language,
                "source_duration": est_duration,
            })
            total_seconds += est_duration

    print(f"Kept {len(items)} items (~{total_seconds/3600:.1f} h estimated).")
    print(f"Skipped: {skipped_short} too short, {skipped_long} too long, "
          f"{skipped_non_lang} wrong language, {skipped_done} already done, "
          f"{skipped_dup} duplicate ids")

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
