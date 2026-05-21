#!/usr/bin/env python3
"""YODAS2 tar.zst → JSON text manifest for synthesize_to_shar.py.

YODAS2 ships as language-coded archives (zh000.tar.zst, pl000.tar.zst, ...)
where each archive contains:
    <lang>/text/NNNNNNNN.json   -- per-channel transcripts
    <lang>/audio/...            -- audio (we don't need it)

Each JSON is a flat dict mapping segment_id → text:
    {
        "<video_id>-<seg_idx>-<start_cs>-<end_cs>": "transcript text",
        ...
    }

Timestamps are centiseconds (1/100s). We parse them to compute duration
per segment, apply standard filters (CJK char predicate for zh, length,
duration), and emit our standard manifest.

Streams via zstandard + tarfile so we never decompress the full 57+ GB
to disk.
"""

import argparse
import io
import json
import random
import re
import tarfile
from pathlib import Path

import zstandard as zstd


PL_CHARS = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")
def _has_pl(t): return any(c in PL_CHARS for c in t)
def _has_cjk(t): return any("一" <= c <= "鿿" for c in t)
LANG_PRED = {"pl": _has_pl, "zh": _has_cjk, "none": lambda t: True}


# segment_id format: <video_id>-<seg_idx>-<start_cs>-<end_cs>
_SEG_ID_RE = re.compile(r"^(?P<vid>.+)-(?P<seg>\d+)-(?P<start>\d+)-(?P<end>\d+)$")


def parse_segment_id(seg_id):
    m = _SEG_ID_RE.match(seg_id)
    if not m:
        return None
    start_cs = int(m.group("start"))
    end_cs = int(m.group("end"))
    duration_s = (end_cs - start_cs) / 100.0
    return seg_id, duration_s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tar-zst", required=True, type=Path,
                   help="YODAS2 language archive (e.g. zh000.tar.zst)")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--language", default="zh", choices=["pl", "zh", "none"])
    p.add_argument("--min-duration", type=float, default=2.0)
    p.add_argument("--max-duration", type=float, default=30.0)
    p.add_argument("--min-chars", type=int, default=None,
                   help="Default: pl=20, zh=5")
    p.add_argument("--max-chars", type=int, default=280)
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--max-hours", type=float, default=None)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lang-filter", action="store_true", default=True)
    p.add_argument("--no-lang-filter", dest="lang_filter", action="store_false")
    p.add_argument("--skip-ids-from", type=Path, default=None)
    args = p.parse_args()

    if args.min_chars is None:
        args.min_chars = {"pl": 20, "zh": 5}.get(args.language, 20)

    pred = LANG_PRED.get(args.language)

    skip_ids = set()
    if args.skip_ids_from is not None:
        prior = json.load(open(args.skip_ids_from, "r", encoding="utf-8"))
        skip_ids = {it["id"] for it in prior}
        print(f"Loaded {len(skip_ids)} IDs to skip from {args.skip_ids_from}")

    items = []
    counters = {"too_short_dur": 0, "too_long_dur": 0,
                "too_short_text": 0, "too_long_text": 0,
                "wrong_language": 0, "already_done": 0,
                "bad_seg_id": 0, "n_text_files": 0}
    total_seconds = 0.0

    print(f"Streaming {args.tar_zst} ...")
    dctx = zstd.ZstdDecompressor()
    with open(args.tar_zst, "rb") as f:
        with dctx.stream_reader(f) as reader:
            # tarfile needs a seekable-ish stream; use streaming mode "r|"
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    if "/text/" not in member.name or not member.name.endswith(".json"):
                        continue
                    counters["n_text_files"] += 1
                    try:
                        fobj = tar.extractfile(member)
                        if fobj is None:
                            continue
                        data = json.loads(fobj.read())
                    except Exception:
                        continue
                    # YODAS2 stores list of {audio_id, text: dict[seg_id → text]}.
                    # Older single-channel layout could be a flat dict; accept both.
                    if isinstance(data, list):
                        entries = data
                    elif isinstance(data, dict):
                        entries = [{"audio_id": None, "text": data}]
                    else:
                        continue
                    for entry in entries:
                        text_dict = entry.get("text", {}) if isinstance(entry, dict) else None
                        if not isinstance(text_dict, dict):
                            continue
                        for seg_id, text in text_dict.items():
                            if not isinstance(text, str):
                                continue
                            text = text.strip()
                            if not text:
                                continue
                            parsed = parse_segment_id(seg_id)
                            if parsed is None:
                                counters["bad_seg_id"] += 1
                                continue
                            _, dur = parsed
                            if dur < args.min_duration:
                                counters["too_short_dur"] += 1
                                continue
                            if dur > args.max_duration:
                                counters["too_long_dur"] += 1
                                continue
                            if len(text) < args.min_chars:
                                counters["too_short_text"] += 1
                                continue
                            if len(text) > args.max_chars:
                                counters["too_long_text"] += 1
                                continue
                            if args.lang_filter and pred is not None and not pred(text):
                                counters["wrong_language"] += 1
                                continue
                            if seg_id in skip_ids:
                                counters["already_done"] += 1
                                continue
                            items.append({
                                "id": seg_id,
                                "text": text,
                                "language": args.language,
                                "source_duration": dur,
                            })
                            total_seconds += dur
                    if counters["n_text_files"] % 100 == 0:
                        print(f"  processed {counters['n_text_files']} text files, "
                              f"kept {len(items)} segs so far "
                              f"({total_seconds/3600:.1f} h)")

    print(f"\nScanned {counters['n_text_files']} text files. "
          f"Kept {len(items)} items (~{total_seconds/3600:.1f} h).")
    print(f"Skipped: short_dur={counters['too_short_dur']}, "
          f"long_dur={counters['too_long_dur']}, "
          f"short_text={counters['too_short_text']}, "
          f"long_text={counters['too_long_text']}, "
          f"wrong_lang={counters['wrong_language']}, "
          f"done={counters['already_done']}, "
          f"bad_id={counters['bad_seg_id']}")

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
