#!/usr/bin/env python3
"""Emilia-YODAS tar → JSON text manifest for synthesize_to_shar.py.

Emilia-YODAS layout:
    <lang>/<LANG>-B<NNNNNN>.tar      one tar per ~30 h shard
      └── <LANG>_<videoid>_W<NNNNNN>.json   {text, duration, speaker, language, dnsmos, phone_count, _id}
      └── <LANG>_<videoid>_W<NNNNNN>.flac   (audio, not needed for synth)

Stdlib only — runs on Clariden login node directly. No torch/lhotse needed.

Output schema (compatible with synthesize_to_shar_nanovllm.slurm):
    [{id, text, language, source_duration}, ...]
"""

import argparse
import glob
import json
import random
import tarfile
from pathlib import Path


PL_CHARS = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")
def _has_pl(t): return any(c in PL_CHARS for c in t)
def _has_cjk(t): return any("一" <= c <= "鿿" for c in t)
LANG_PRED = {"pl": _has_pl, "zh": _has_cjk, "none": lambda t: True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tar-dir", required=True, type=Path,
                   help="Dir containing Emilia-YODAS <LANG>-B*.tar files")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--language", required=True, choices=["pl", "zh", "en", "de", "fr", "ja", "ko", "none"])
    p.add_argument("--min-duration", type=float, default=2.0)
    p.add_argument("--max-duration", type=float, default=30.0)
    p.add_argument("--min-chars", type=int, default=None,
                   help="Default: pl=20, zh=5, otherwise=20")
    p.add_argument("--max-chars", type=int, default=280)
    p.add_argument("--min-dnsmos", type=float, default=3.0,
                   help="Drop clips with dnsmos below this threshold (Emilia paper recommends 3.0)")
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--max-hours", type=float, default=None)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lang-filter", action="store_true", default=True)
    p.add_argument("--no-lang-filter", dest="lang_filter", action="store_false")
    p.add_argument("--skip-ids-from", type=Path, default=None,
                   help="Plain-text file with one ID per line — drop these (already-synthesized dedup)")
    args = p.parse_args()

    if args.min_chars is None:
        args.min_chars = {"pl": 20, "zh": 5}.get(args.language, 20)

    pred = LANG_PRED.get(args.language)
    if pred is None:
        pred = lambda t: True

    skip = set()
    if args.skip_ids_from and args.skip_ids_from.exists():
        with open(args.skip_ids_from) as f:
            for line in f:
                s = line.strip()
                if s:
                    skip.add(s)
        print(f"[skip] loaded {len(skip)} ids")

    tars = sorted(args.tar_dir.glob("*.tar"))
    if not tars:
        raise SystemExit(f"No .tar files in {args.tar_dir}")
    print(f"[tars] {len(tars)} archives to scan")

    items = []
    stats = {
        "seen": 0,
        "kept": 0,
        "skip_duration": 0,
        "skip_chars": 0,
        "skip_language": 0,
        "skip_dnsmos": 0,
        "skip_in_skipset": 0,
        "total_source_hours": 0.0,
    }

    for tar_path in tars:
        with tarfile.open(tar_path) as t:
            for m in t:
                if not m.name.endswith(".json"):
                    continue
                stats["seen"] += 1
                try:
                    j = json.loads(t.extractfile(m).read())
                except Exception:
                    continue

                _id = j.get("_id") or m.name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
                if _id in skip:
                    stats["skip_in_skipset"] += 1
                    continue

                text = (j.get("text") or "").strip()
                dur = float(j.get("duration", 0))
                lang = j.get("language", "")
                dnsmos = float(j.get("dnsmos", 0))

                if dur < args.min_duration or dur > args.max_duration:
                    stats["skip_duration"] += 1
                    continue
                if len(text) < args.min_chars or len(text) > args.max_chars:
                    stats["skip_chars"] += 1
                    continue
                if args.lang_filter and lang and lang != args.language:
                    stats["skip_language"] += 1
                    continue
                if args.lang_filter and not pred(text):
                    stats["skip_language"] += 1
                    continue
                if dnsmos < args.min_dnsmos:
                    stats["skip_dnsmos"] += 1
                    continue

                items.append({
                    "id": _id,
                    "text": text,
                    "language": args.language,
                    "source_duration": dur,
                })
                stats["kept"] += 1
                stats["total_source_hours"] += dur / 3600.0

                if args.max_items and stats["kept"] >= args.max_items:
                    break
                if args.max_hours and stats["total_source_hours"] >= args.max_hours:
                    break
        print(f"[tar] {tar_path.name}: cumulative kept={stats['kept']:,}  "
              f"hours={stats['total_source_hours']:.1f}")
        if args.max_items and stats["kept"] >= args.max_items:
            break
        if args.max_hours and stats["total_source_hours"] >= args.max_hours:
            break

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(items)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, indent=2))
    print(f"[out] {args.output} — {stats['kept']:,} items / "
          f"{stats['total_source_hours']:.1f} h")


if __name__ == "__main__":
    main()
