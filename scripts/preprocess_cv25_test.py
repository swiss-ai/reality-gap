#!/usr/bin/env python3
"""Repack CV25 test.parquet into the audio_inference.py-expected schema.

Input schema (CV25 raw):
    audio_bytes : binary           (raw MP3)
    sentence    : string           (transcript)
    clip_id     : string           (utterance id)
    locale      : string           (language code)
    + a dozen other CommonVoice metadata columns we ignore

Output schema (matches our delivery + load_parquet_dataset expectations):
    id        : string
    text      : string
    duration  : float64
    audio     : struct<bytes: binary, sampling_rate: int64>
    language  : string

Reads MP3 bytes via soundfile to recover sampling_rate + duration, but
keeps the raw MP3 bytes as the `audio.bytes` payload (no re-encoding).
audio_inference.py decodes via soundfile downstream so this round-trips.
"""
from __future__ import annotations

import argparse, io
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf


def repack(in_path: Path, out_path: Path, language: str) -> dict:
    pf = pq.ParquetFile(in_path)
    n_rows = pf.metadata.num_rows
    print(f"Reading {in_path} ({n_rows} rows)...", flush=True)

    ids, texts, durs, blobs, srs, langs = [], [], [], [], [], []
    n_decode_err = 0
    for batch in pf.iter_batches(batch_size=512,
                                 columns=["audio_bytes", "sentence", "clip_id"]):
        d = batch.to_pydict()
        for ab, sent, cid in zip(d["audio_bytes"], d["sentence"], d["clip_id"]):
            if not ab:
                n_decode_err += 1
                continue
            try:
                info = sf.info(io.BytesIO(ab))
                sr = int(info.samplerate)
                duration = float(info.frames) / sr if sr else 0.0
            except Exception:
                n_decode_err += 1
                continue
            ids.append(cid)
            texts.append(sent or "")
            durs.append(duration)
            blobs.append(ab)
            srs.append(sr)
            langs.append(language)

    audio_type = pa.struct([
        pa.field("bytes", pa.binary()),
        pa.field("sampling_rate", pa.int64()),
    ])
    audio = pa.array(
        [{"bytes": b, "sampling_rate": s} for b, s in zip(blobs, srs)],
        type=audio_type,
    )
    table = pa.table({
        "id": pa.array(ids, type=pa.string()),
        "text": pa.array(texts, type=pa.string()),
        "duration": pa.array(durs, type=pa.float64()),
        "audio": audio,
        "language": pa.array(langs, type=pa.string()),
    })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="snappy")
    sr_unique = sorted(set(srs)) if srs else []
    return {
        "n_in": n_rows,
        "n_out": len(ids),
        "n_decode_err": n_decode_err,
        "sample_rates": sr_unique,
        "total_hours": round(sum(durs) / 3600, 2),
        "size_mb": round(out_path.stat().st_size / 1024 / 1024, 1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path,
                    help="CV25 test.parquet (e.g. /capstor/.../commonvoice25/pl/test.parquet)")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output parquet path (e.g. .../eval_test_sets/cv25_pl/test.parquet)")
    ap.add_argument("--language", required=True, choices=["pl", "zh"])
    args = ap.parse_args()

    stats = repack(args.input, args.output, args.language)
    print(f"\nDone: {stats}")


if __name__ == "__main__":
    main()
