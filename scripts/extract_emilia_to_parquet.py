#!/usr/bin/env python3
"""Emilia WDS tar → parquet matching our synth-delivery schema.

Each Emilia tar holds flat <key>.mp3 + <key>.json pairs (no nested tars).
Sidecar JSON has at least: {text, language, duration} (and usually speaker/dnsmos).

Output parquet schema:
    id        : string
    text      : string
    duration  : float64
    audio     : struct<bytes: binary, sampling_rate: int64>   (mp3 bytes inlined)
    language  : string
"""
import argparse, io, json, tarfile
from pathlib import Path

import soundfile as sf
import pyarrow as pa
import pyarrow.parquet as pq


def extract_tar(tar_path: Path, language: str,
                min_dur: float, max_dur: float,
                rows_out: list, stats: dict):
    with tarfile.open(tar_path, "r") as tf:
        members = {}
        for m in tf:
            if not m.isfile():
                continue
            stem, _, ext = m.name.rpartition(".")
            members.setdefault(stem, {})[ext.lower()] = m
        for stem, parts in members.items():
            if "json" not in parts or "mp3" not in parts:
                stats["unpaired"] += 1
                continue
            try:
                meta = json.loads(tf.extractfile(parts["json"]).read())
            except Exception:
                stats["meta_error"] += 1
                continue
            text = (meta.get("text") or "").strip()
            dur = float(meta.get("duration", 0.0))
            if not text:
                stats["no_text"] += 1
                continue
            if dur < min_dur or dur > max_dur:
                stats["bad_dur"] += 1
                continue
            audio_bytes = tf.extractfile(parts["mp3"]).read()
            try:
                info = sf.info(io.BytesIO(audio_bytes))
                sr = int(info.samplerate)
            except Exception:
                sr = 24000
            rows_out.append({
                "id": Path(stem).name,
                "text": text,
                "duration": dur,
                "bytes": audio_bytes,
                "sr": sr,
            })
            stats["kept"] += 1


def flush(rows, out_path, language):
    if not rows:
        return
    audio_struct = pa.array(
        [{"bytes": r["bytes"], "sampling_rate": r["sr"]} for r in rows],
        type=pa.struct([pa.field("bytes", pa.binary()),
                        pa.field("sampling_rate", pa.int64())]))
    table = pa.table({
        "id": pa.array([r["id"] for r in rows], type=pa.string()),
        "text": pa.array([r["text"] for r in rows], type=pa.string()),
        "duration": pa.array([r["duration"] for r in rows], type=pa.float64()),
        "audio": audio_struct,
        "language": pa.array([language] * len(rows), type=pa.string()),
    })
    pq.write_table(table, out_path, compression="snappy")
    print(f"  wrote {len(rows)} rows -> {out_path.name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar-glob", required=True,
                    help="glob for Emilia tars (e.g. /.../ZH-B*.tar)")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--language", required=True)
    ap.add_argument("--min-duration", type=float, default=2.0)
    ap.add_argument("--max-duration", type=float, default=30.0)
    args = ap.parse_args()

    import glob as _g
    tars = sorted(_g.glob(args.tar_glob))
    if not tars:
        raise SystemExit(f"no tars matched: {args.tar_glob}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(tars)} tars", flush=True)

    stats = {"kept": 0, "unpaired": 0, "meta_error": 0,
             "no_text": 0, "bad_dur": 0}
    total_dur = 0.0
    for i, t in enumerate(tars):
        rows = []
        print(f"[{i+1}/{len(tars)}] {Path(t).name}", flush=True)
        extract_tar(Path(t), args.language,
                    args.min_duration, args.max_duration, rows, stats)
        out_path = args.output_dir / f"train-{i:05d}.parquet"
        flush(rows, out_path, args.language)
        total_dur += sum(r["duration"] for r in rows)
        print(f"  running total: {stats['kept']:,} cuts, {total_dur/3600:.1f} h",
              flush=True)

    print(f"\n=== done ===")
    print(f"  kept       : {stats['kept']:,} ({total_dur/3600:,.1f} h)")
    print(f"  unpaired   : {stats['unpaired']:,}")
    print(f"  meta_error : {stats['meta_error']:,}")
    print(f"  no_text    : {stats['no_text']:,}")
    print(f"  bad_dur    : {stats['bad_dur']:,}")


if __name__ == "__main__":
    main()
