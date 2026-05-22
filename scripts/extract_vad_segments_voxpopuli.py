#!/usr/bin/env python3
"""VoxPopuli VAD-merged JSONL → per-segment audio manifest for Whisper ASR.

Reads `per_lang_year_dedup/<lang>_YYYY.jsonl` (one line per plenary session)
and emits a JSONL manifest with one entry per VAD sub-segment, suitable for
sharded Whisper inference.

Source schema (one line per session):
    {
      "20171005-0900-PLENARY-3_pl": {
        "timestamps": [[start_sample, end_sample], ...],
        "duration_sec": 4584.94,
        "sample_rate": 16000,
        "lang": "pl"
      }
    }

Output schema (one line per VAD sub-segment):
    {
      "id": "20171005-0900-PLENARY-3_pl_0042",
      "session_id": "20171005-0900-PLENARY-3_pl",
      "audio_path": "/.../raw_audios/pl/2017/20171005-0900-PLENARY-3_pl.ogg",
      "year": 2017,
      "start_sec": 12.34,
      "end_sec": 18.92,
      "duration_sec": 6.58
    }

Filters by [min_duration, max_duration]. Skips IDs in --skip-ids-from
(plain text file, one ID per line) for Granary dedup.
"""

import argparse
import json
import re
from pathlib import Path


_SESSION_RE = re.compile(r"^(\d{8})-")  # leading YYYYMMDD → year


def session_year(session_id):
    m = _SESSION_RE.match(session_id)
    return int(m.group(1)[:4]) if m else None


def audio_path_for(session_id, lang, audio_root):
    year = session_year(session_id)
    if year is None:
        return None
    return audio_root / lang / str(year) / f"{session_id}.ogg"


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vad-dir", type=Path, required=True,
                   help="Dir with per_lang_year_dedup/<lang>_YYYY.jsonl files")
    p.add_argument("--audio-root", type=Path, required=True,
                   help="VoxPopuli raw_audios/ root (contains <lang>/<year>/...)")
    p.add_argument("--language", required=True)
    p.add_argument("--years", default=None,
                   help="Comma-separated year filter, e.g. 2017,2018,2019")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--min-duration", type=float, default=2.0)
    p.add_argument("--max-duration", type=float, default=30.0)
    p.add_argument("--skip-ids-from", type=Path, default=None,
                   help="Plain-text file: one session_id or segment_id per line to drop")
    p.add_argument("--require-audio-exists", action="store_true",
                   help="Stat each session audio; drop missing files (slow on cold cache)")
    p.add_argument("--max-segments", type=int, default=None)
    args = p.parse_args()

    skip = set()
    if args.skip_ids_from and args.skip_ids_from.exists():
        with open(args.skip_ids_from) as f:
            for line in f:
                s = line.strip()
                if s:
                    skip.add(s)
        print(f"[skip] loaded {len(skip)} ids")

    year_filter = None
    if args.years:
        year_filter = {int(y) for y in args.years.split(",")}

    jsonl_files = sorted(args.vad_dir.glob(f"{args.language}_*.jsonl"))
    if not jsonl_files:
        raise SystemExit(f"No {args.language}_*.jsonl in {args.vad_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "sessions_seen": 0,
        "sessions_kept": 0,
        "sessions_skip_year": 0,
        "sessions_skip_audio_missing": 0,
        "sessions_skip_in_skipset": 0,
        "segments_raw": 0,
        "segments_kept": 0,
        "segments_skip_duration": 0,
        "segments_skip_in_skipset": 0,
        "total_hours": 0.0,
    }

    with open(args.output, "w", encoding="utf-8") as out:
        for jf in jsonl_files:
            for record in iter_jsonl(jf):
                for session_id, meta in record.items():
                    stats["sessions_seen"] += 1
                    year = session_year(session_id)
                    if year_filter is not None and year not in year_filter:
                        stats["sessions_skip_year"] += 1
                        continue
                    if session_id in skip:
                        stats["sessions_skip_in_skipset"] += 1
                        continue
                    audio = audio_path_for(session_id, args.language, args.audio_root)
                    if audio is None:
                        continue
                    if args.require_audio_exists and not audio.exists():
                        stats["sessions_skip_audio_missing"] += 1
                        continue

                    sr = meta.get("sample_rate", 16000)
                    timestamps = meta.get("timestamps", [])
                    kept_in_session = 0
                    for idx, (s_samp, e_samp) in enumerate(timestamps):
                        stats["segments_raw"] += 1
                        start_sec = s_samp / sr
                        end_sec = e_samp / sr
                        dur = end_sec - start_sec
                        if dur < args.min_duration or dur > args.max_duration:
                            stats["segments_skip_duration"] += 1
                            continue
                        seg_id = f"{session_id}_{idx:05d}"
                        if seg_id in skip:
                            stats["segments_skip_in_skipset"] += 1
                            continue
                        rec = {
                            "id": seg_id,
                            "session_id": session_id,
                            "audio_path": str(audio),
                            "year": year,
                            "start_sec": round(start_sec, 3),
                            "end_sec": round(end_sec, 3),
                            "duration_sec": round(dur, 3),
                        }
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        stats["segments_kept"] += 1
                        stats["total_hours"] += dur / 3600.0
                        kept_in_session += 1
                        if args.max_segments and stats["segments_kept"] >= args.max_segments:
                            break
                    if kept_in_session > 0:
                        stats["sessions_kept"] += 1
                    if args.max_segments and stats["segments_kept"] >= args.max_segments:
                        break
                if args.max_segments and stats["segments_kept"] >= args.max_segments:
                    break
            if args.max_segments and stats["segments_kept"] >= args.max_segments:
                break

    print(json.dumps(stats, indent=2))
    print(f"[out] {args.output} — {stats['segments_kept']} segments / {stats['total_hours']:.1f} h")


if __name__ == "__main__":
    main()
