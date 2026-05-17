#!/usr/bin/env python3
"""Verify audio.bytes in a parquet contains a real encoded audio file."""
import argparse
import pyarrow.parquet as pq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("parquet")
    p.add_argument("--n-rows", type=int, default=3)
    args = p.parse_args()

    tbl = pq.read_table(args.parquet, columns=["id", "audio"])
    print(f"Total rows: {tbl.num_rows}")
    for i in range(min(args.n_rows, tbl.num_rows)):
        row = tbl.slice(i, 1).to_pydict()
        rid = row["id"][0]
        audio = row["audio"][0]
        if not isinstance(audio, dict) or "bytes" not in audio:
            print(f"  row {i} {rid}: BAD - audio not a dict with bytes")
            continue
        b = audio["bytes"]
        if not b:
            print(f"  row {i} {rid}: BAD - empty bytes")
            continue
        magic = b[:4]
        sr = audio.get("sampling_rate")
        if magic == b"RIFF":
            kind = "WAV"
        elif magic == b"fLaC":
            kind = "FLAC"
        elif magic == b"OggS":
            kind = "OGG"
        else:
            kind = f"UNKNOWN magic={magic!r}"
        print(f"  row {i} {rid}: {kind}, {len(b):,} bytes, sr={sr}")


if __name__ == "__main__":
    main()
