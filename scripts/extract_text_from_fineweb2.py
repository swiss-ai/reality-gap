#!/usr/bin/env python3
"""FineWeb-2 parquet shard → JSON text manifest for TTS synth.

FineWeb-2 stores web-scale documents (multi-sentence) keyed by `id` and `text`.
We sentence-segment each document, filter by length + language predicate, and
emit the standard manifest schema other scripts/synthesize_to_shar.py consumes:

    [{"id": "...", "text": "...", "language": "...", "source_duration": float}]

Segment IDs are formatted `{shard_stem}_{rg_idx:03d}_{row_idx:06d}_{seg_idx:04d}`
so the original document position is preserved (useful for any future
re-stitching into long-form sequences — same pattern as our VoxPopuli/YODAS
extracts).

Cluster paths:
    /capstor/store/cscs/swissai/infra01/datasets/HuggingFaceFW/fineweb-2/data/pol_Latn/train/*.parquet
    /capstor/store/cscs/swissai/infra01/datasets/HuggingFaceFW/fineweb-2/data/cmn_Hani/train/*.parquet

Example:
    python scripts/extract_text_from_fineweb2.py \\
        --shard /capstor/.../fineweb-2/data/pol_Latn/train/000_00000.parquet \\
        --language pl \\
        --output data/manifests/pl_fineweb2_shard0.json \\
        --max-hours 200
"""

import argparse
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq


PL_CHARS = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")
def _has_pl(t): return any(c in PL_CHARS for c in t)
def _has_cjk(t): return any("一" <= c <= "鿿" for c in t)
LANG_PRED = {"pl": _has_pl, "zh": _has_cjk, "none": lambda t: True}


_PL_SENT_RE = re.compile(r'(?<=[\.\?\!\…])\s+')

def _split_pl(text):
    return [p.strip() for p in _PL_SENT_RE.split(text) if p.strip()]

_ZH_TERMINAL = set("。！？…")

def _split_zh(text):
    out, buf = [], []
    for ch in text:
        buf.append(ch)
        if ch in _ZH_TERMINAL:
            seg = "".join(buf).strip()
            if seg:
                out.append(seg)
            buf = []
    if buf:
        rest = "".join(buf).strip()
        if rest:
            out.append(rest)
    return out

SPLITTER = {"pl": _split_pl, "zh": _split_zh, "none": lambda t: [t]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard", required=True, type=Path,
                   help="Path to a single FineWeb-2 parquet shard")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--language", default="pl", choices=["pl", "zh", "none"])
    p.add_argument("--text-column", default="text")
    p.add_argument("--id-column", default="id")
    p.add_argument("--min-chars", type=int, default=None,
                   help="Defaults: pl=30, zh=8")
    p.add_argument("--max-chars", type=int, default=None,
                   help="Defaults: pl=350, zh=120 (~2-25s at language CPS)")
    p.add_argument("--cps", type=float, default=None,
                   help="Chars/sec for duration estimate. Defaults: pl=14, zh=5")
    p.add_argument("--max-hours", type=float, default=None,
                   help="Cap output at N hours of estimated synth duration")
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--shuffle", action="store_true",
                   help="Shuffle segments before applying max-hours/items cap")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lang-filter", action="store_true", default=True)
    p.add_argument("--no-lang-filter", dest="lang_filter", action="store_false")
    p.add_argument("--skip-ids-from", type=Path, default=None,
                   help="JSON manifest of IDs to skip (dedup across extracts)")
    args = p.parse_args()

    if args.min_chars is None:
        args.min_chars = {"pl": 30, "zh": 8}.get(args.language, 30)
    if args.max_chars is None:
        args.max_chars = {"pl": 350, "zh": 120}.get(args.language, 350)
    if args.cps is None:
        args.cps = {"pl": 14.0, "zh": 5.0}.get(args.language, 14.0)

    pred = LANG_PRED.get(args.language)
    splitter = SPLITTER.get(args.language, lambda t: [t])

    skip_ids = set()
    if args.skip_ids_from is not None:
        prior = json.load(open(args.skip_ids_from, "r", encoding="utf-8"))
        skip_ids = {it["id"] for it in prior}
        print(f"Loaded {len(skip_ids)} IDs to skip from {args.skip_ids_from}")

    print(f"Reading {args.shard.name} ...")
    pf = pq.ParquetFile(args.shard)
    shard_stem = args.shard.stem

    items = []
    seen_ids = set()
    skipped_short = skipped_long = skipped_non_lang = skipped_done = skipped_dup = 0
    total_chars = 0
    total_seconds = 0.0
    docs_read = 0
    early_cap_seconds = (args.max_hours * 3600.0) if (args.max_hours and not args.shuffle) else None
    early_cap_items = args.max_items if (args.max_items and not args.shuffle) else None

    done = False
    for rg_idx in range(pf.num_row_groups):
        if done:
            break
        rg = pf.read_row_group(rg_idx, columns=[args.id_column, args.text_column])
        ids = rg.column(args.id_column).to_pylist()
        texts = rg.column(args.text_column).to_pylist()
        for row_idx, (doc_id, text) in enumerate(zip(ids, texts)):
            docs_read += 1
            if not text:
                continue
            for seg_idx, seg in enumerate(splitter(text)):
                if len(seg) < args.min_chars:
                    skipped_short += 1
                    continue
                if len(seg) > args.max_chars:
                    skipped_long += 1
                    continue
                if args.lang_filter and pred is not None and not pred(seg):
                    skipped_non_lang += 1
                    continue
                cid = f"{shard_stem}_{rg_idx:03d}_{row_idx:06d}_{seg_idx:04d}"
                if cid in skip_ids:
                    skipped_done += 1
                    continue
                if cid in seen_ids:
                    skipped_dup += 1
                    continue
                seen_ids.add(cid)
                est_duration = len(seg) / args.cps
                items.append({
                    "id": cid,
                    "text": seg,
                    "language": args.language,
                    "source_duration": est_duration,
                })
                total_chars += len(seg)
                total_seconds += est_duration
                if early_cap_seconds is not None and total_seconds >= early_cap_seconds:
                    done = True
                    break
                if early_cap_items is not None and len(items) >= early_cap_items:
                    done = True
                    break
            if done:
                break

    print(f"Read {docs_read} docs; kept {len(items)} segments "
          f"({total_chars} chars, ~{total_seconds/3600:.1f} h synth est.)")
    print(f"Skipped: {skipped_short} too short, {skipped_long} too long, "
          f"{skipped_non_lang} wrong language, {skipped_done} already done, "
          f"{skipped_dup} duplicate ids")

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(items)

    if args.max_hours is not None and args.shuffle:
        cap = args.max_hours * 3600.0
        kept, running = [], 0.0
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

    if args.max_items and args.shuffle:
        items = items[:args.max_items]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(items)} items to {args.output}")


if __name__ == "__main__":
    main()
