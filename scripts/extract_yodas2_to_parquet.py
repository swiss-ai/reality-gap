#!/usr/bin/env python3
"""YODAS2 tar.zst → per-segment parquet matching our synth-delivery schema.

Outer tar structure:
    <lang>000/text/<chunk>.json       — per-chunk transcripts (list of {audio_id, text: {seg_id: text}})
    <lang>000/audio/<chunk>.tar.gz    — nested gzip tar containing <video_id>.wav files (full videos)
    <lang>000/duration/<chunk>.txt    — ignored (durations are in segment_id timestamps)

Segment id format: <video_id>-<seg_idx>-<start_cs>-<end_cs>  (centiseconds)

Output: parquet shards under output_dir/train-NNNNN.parquet, one row per segment with
columns matching our schema:
    id        : string
    text      : string
    duration  : float64
    audio     : struct<bytes: binary, sampling_rate: int64>   (WAV-encoded slice)
    language  : string

Assumes outer tar is already zstd-decompressed (input is plain .tar). For .tar.zst,
decompress first via:  zstd -d <file>.tar.zst -o <file>.tar
"""
import argparse, io, json, re, tarfile, sys
from pathlib import Path

import numpy as np
import soundfile as sf
import pyarrow as pa
import pyarrow.parquet as pq

PL_CHARS = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")
def _has_pl(t): return any(c in PL_CHARS for c in t)
def _has_cjk(t): return any("一" <= c <= "鿿" for c in t)
LANG_PRED = {"pl": _has_pl, "zh": _has_cjk, "none": lambda t: True}

_SEG_ID_RE = re.compile(r"^(?P<vid>.+)-(?P<seg>\d+)-(?P<start>\d+)-(?P<end>\d+)$")

def parse_seg(seg_id):
    m = _SEG_ID_RE.match(seg_id)
    if not m: return None
    return m.group("vid"), int(m.group("start")), int(m.group("end"))


def encode_wav(slice_arr, sr):
    """Encode a numpy float array to WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, slice_arr, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tar", required=True, type=Path,
                    help="Decompressed YODAS2 tar (e.g. pl000.tar)")
    ap.add_argument("--output-dir", required=True, type=Path,
                    help="Parquet output dir (train-NNNNN.parquet shards)")
    ap.add_argument("--language", required=True, choices=["pl", "zh", "none"])
    ap.add_argument("--min-duration", type=float, default=2.0)
    ap.add_argument("--max-duration", type=float, default=30.0)
    ap.add_argument("--min-chars", type=int, default=None)
    ap.add_argument("--max-chars", type=int, default=280)
    ap.add_argument("--no-lang-filter", action="store_true")
    ap.add_argument("--shard-rows", type=int, default=1000,
                    help="Rows per output parquet shard")
    ap.add_argument("--max-items", type=int, default=None,
                    help="Cap total emitted rows (for testing)")
    args = ap.parse_args()

    if args.min_chars is None:
        args.min_chars = {"pl": 20, "zh": 5}.get(args.language, 5)

    pred = LANG_PRED.get(args.language) if not args.no_lang_filter else None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    shard_idx = 0
    total_dur_s = 0.0
    stats = {"chunks_seen": 0, "kept": 0, "skipped_dur": 0, "skipped_text": 0,
             "skipped_lang": 0, "no_audio_in_chunk": 0, "bad_seg_id": 0,
             "video_missing": 0, "decode_error": 0}

    def flush():
        nonlocal rows, shard_idx
        if not rows: return
        audio_struct = pa.array(
            [{"bytes": r["bytes"], "sampling_rate": r["sr"]} for r in rows],
            type=pa.struct([pa.field("bytes", pa.binary()),
                            pa.field("sampling_rate", pa.int64())]))
        table = pa.table({
            "id": pa.array([r["id"] for r in rows], type=pa.string()),
            "text": pa.array([r["text"] for r in rows], type=pa.string()),
            "duration": pa.array([r["duration"] for r in rows], type=pa.float64()),
            "audio": audio_struct,
            "language": pa.array([args.language] * len(rows), type=pa.string()),
        })
        out = args.output_dir / f"train-{shard_idx:05d}.parquet"
        pq.write_table(table, out, compression="snappy")
        print(f"  shard {shard_idx}: wrote {len(rows)} rows to {out.name}", flush=True)
        rows.clear()
        shard_idx += 1

    print(f"Opening outer tar: {args.tar}", flush=True)
    outer = tarfile.open(args.tar, mode="r")

    # Index outer tar by chunk_id → (text_member, audio_member)
    text_by_chunk = {}
    audio_by_chunk = {}
    for m in outer.getmembers():
        if not m.isfile(): continue
        parts = m.name.split("/")
        if len(parts) < 3: continue
        sub = parts[1]   # "text", "audio", "duration"
        fname = parts[2]
        chunk_id = fname.rsplit(".", 1)[0].split(".")[0]  # 00000011.json -> 00000011
        if sub == "text" and fname.endswith(".json"):
            text_by_chunk[chunk_id] = m
        elif sub == "audio" and fname.endswith(".tar.gz"):
            audio_by_chunk[chunk_id] = m
    print(f"Found {len(text_by_chunk)} text chunks, {len(audio_by_chunk)} audio chunks", flush=True)

    common = sorted(set(text_by_chunk) & set(audio_by_chunk))
    print(f"Processing {len(common)} chunks with both text + audio", flush=True)

    for chunk_id in common:
        stats["chunks_seen"] += 1
        # Read text
        try:
            data = json.loads(outer.extractfile(text_by_chunk[chunk_id]).read())
        except Exception as e:
            print(f"  chunk {chunk_id}: text load failed: {e}", flush=True)
            continue
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict):
            entries = [{"audio_id": None, "text": data}]
        else:
            continue
        # Group segments by video_id (so we read each video's wav once)
        per_video = {}  # video_id -> list of (seg_id, text, start_cs, end_cs)
        for entry in entries:
            text_dict = entry.get("text", {}) if isinstance(entry, dict) else None
            if not isinstance(text_dict, dict): continue
            for seg_id, text in text_dict.items():
                if not isinstance(text, str): continue
                text = text.strip()
                if not text: continue
                parsed = parse_seg(seg_id)
                if parsed is None:
                    stats["bad_seg_id"] += 1
                    continue
                vid, start_cs, end_cs = parsed
                dur = (end_cs - start_cs) / 100.0
                if dur < args.min_duration or dur > args.max_duration:
                    stats["skipped_dur"] += 1
                    continue
                if len(text) < args.min_chars or len(text) > args.max_chars:
                    stats["skipped_text"] += 1
                    continue
                if pred is not None and not pred(text):
                    stats["skipped_lang"] += 1
                    continue
                per_video.setdefault(vid, []).append((seg_id, text, start_cs, end_cs))
        if not per_video:
            stats["no_audio_in_chunk"] += 1
            continue

        # Read inner audio tar.gz into memory once, then process all videos
        try:
            audio_blob = outer.extractfile(audio_by_chunk[chunk_id]).read()
        except Exception as e:
            print(f"  chunk {chunk_id}: audio blob read failed: {e}", flush=True)
            continue
        try:
            inner = tarfile.open(fileobj=io.BytesIO(audio_blob), mode="r:gz")
        except Exception as e:
            print(f"  chunk {chunk_id}: inner tar open failed: {e}", flush=True)
            continue
        wav_members = {}
        for im in inner.getmembers():
            if not im.isfile() or not im.name.endswith(".wav"): continue
            stem = Path(im.name).stem  # "./x-bW3OaTtjc.wav" -> "x-bW3OaTtjc"
            wav_members[stem] = im

        for vid, segs in per_video.items():
            if vid not in wav_members:
                stats["video_missing"] += len(segs)
                continue
            try:
                wav_bytes = inner.extractfile(wav_members[vid]).read()
                audio_arr, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
                if audio_arr.ndim > 1:
                    audio_arr = audio_arr.mean(axis=1)  # mono
            except Exception as e:
                stats["decode_error"] += len(segs)
                continue

            for seg_id, text, start_cs, end_cs in segs:
                start_smp = int(start_cs * sr / 100.0)
                end_smp = int(end_cs * sr / 100.0)
                if end_smp > len(audio_arr):
                    end_smp = len(audio_arr)
                if start_smp >= end_smp:
                    continue
                slice_arr = audio_arr[start_smp:end_smp]
                if slice_arr.size == 0: continue
                wav = encode_wav(slice_arr, sr)
                dur = (end_cs - start_cs) / 100.0
                rows.append({"id": seg_id, "text": text, "duration": dur,
                             "bytes": wav, "sr": int(sr)})
                stats["kept"] += 1
                total_dur_s += dur
                if len(rows) >= args.shard_rows:
                    flush()
                if args.max_items and stats["kept"] >= args.max_items:
                    flush()
                    print(f"\nReached max_items={args.max_items}, stopping.")
                    print_stats(stats, total_dur_s)
                    return
        inner.close()
        if stats["chunks_seen"] % 5 == 0:
            print(f"  …processed {stats['chunks_seen']}/{len(common)} chunks, "
                  f"kept {stats['kept']} segs ({total_dur_s/3600:.1f} h so far)",
                  flush=True)

    flush()
    print_stats(stats, total_dur_s)


def print_stats(stats, total_dur_s):
    print(f"\n=== done ===")
    print(f"  chunks_seen      : {stats['chunks_seen']}")
    print(f"  kept             : {stats['kept']:,}  ({total_dur_s/3600:.1f} h)")
    print(f"  skipped_dur      : {stats['skipped_dur']:,}")
    print(f"  skipped_text     : {stats['skipped_text']:,}")
    print(f"  skipped_lang     : {stats['skipped_lang']:,}")
    print(f"  bad_seg_id       : {stats['bad_seg_id']:,}")
    print(f"  video_missing    : {stats['video_missing']:,}")
    print(f"  decode_error     : {stats['decode_error']:,}")
    print(f"  no_audio_in_chunk: {stats['no_audio_in_chunk']:,}")


if __name__ == "__main__":
    main()
