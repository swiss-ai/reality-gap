#!/usr/bin/env python3
"""Restore the CV25 PL + ZH-CN test.parquets I corrupted on 2026-05-27.

Downloads raw CV25 from HF (mozilla-foundation/common_voice_25_0) for each
language, transforms to match the original schema observed in untouched
sister files (sl/test.parquet as control), writes to /tmp first, schema-
verifies, then atomically replaces the corrupted store-side file.

Fails loud if the target path is a symlink (it shouldn't be in /capstor/store/
but I'm not making that mistake twice).
"""
from __future__ import annotations

import argparse, os, shutil, sys, tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


CONTROL = "/capstor/store/cscs/swissai/infra01/audio-datasets/processed/commonvoice25/sl/test.parquet"

TARGETS = {
    "pl": "/capstor/store/cscs/swissai/infra01/audio-datasets/processed/commonvoice25/pl/test.parquet",
    "zh-CN": "/capstor/store/cscs/swissai/infra01/audio-datasets/processed/commonvoice25/zh-CN/test.parquet",
}


def load_target_schema() -> pa.Schema:
    """Read sl/test.parquet's schema as the authoritative target."""
    pf = pq.ParquetFile(CONTROL)
    schema = pf.schema_arrow
    print(f"Target schema (from {CONTROL}):")
    for i, f in enumerate(schema):
        print(f"  [{i}] {f.name}: {f.type}")
    return schema


def download_hf(language: str):
    """Pull CV25 test split for one language from HuggingFace."""
    import datasets
    print(f"\n--- Downloading mozilla-foundation/common_voice_25_0 ({language}, test) ---")
    ds = datasets.load_dataset(
        "mozilla-foundation/common_voice_25_0",
        language,
        split="test",
    )
    # We must avoid HF's audio decoding to keep the MP3 bytes raw.
    ds = ds.cast_column("audio", datasets.Audio(decode=False))
    print(f"  rows: {len(ds)}  cols: {ds.column_names}")
    return ds


def transform(ds, target_schema: pa.Schema):
    """Reshape HF raw dataset into the target schema's columns.

    Best effort: fills columns the HF schema provides, leaves the rest empty.
    """
    rows_out = {f.name: [] for f in target_schema}
    have = set(ds.column_names)

    for row in ds:
        # Raw HF audio column is a dict: {'path': str, 'bytes': bytes}
        # We want raw MP3 bytes in audio_bytes.
        audio_bytes = b""
        if "audio" in row and isinstance(row["audio"], dict):
            audio_bytes = row["audio"].get("bytes") or b""

        for f in target_schema:
            name = f.name
            if name == "audio_bytes":
                rows_out[name].append(audio_bytes)
            elif name == "clip_id":
                # Original processed pipeline appears to set clip_id = file path/name
                rows_out[name].append(row.get("path") or "")
            elif name == "split":
                rows_out[name].append("test")
            elif name == "sentence_domain":
                # New in CV25; pass through if HF provides it, else empty
                rows_out[name].append(row.get("sentence_domain") or "")
            elif name == "sentence_id":
                rows_out[name].append(row.get("sentence_id") or "")
            elif name in have:
                rows_out[name].append(row.get(name))
            elif name == "accents":
                # HF sometimes calls this "accent" (singular)
                rows_out[name].append(row.get("accent") or row.get("accents") or "")
            else:
                rows_out[name].append(None)

    # Cast each list to the corresponding arrow type
    arrays = []
    for f in target_schema:
        col = rows_out[f.name]
        try:
            arr = pa.array(col, type=f.type)
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            # Type coercion fallback (e.g. None vs int)
            arr = pa.array(col).cast(f.type, safe=False)
        arrays.append(arr)
    return pa.table(arrays, schema=target_schema)


def safe_replace(table: pa.Table, dest: Path):
    """Write table to a tmp file, verify schema, then atomically replace dest.

    Refuses if dest is a symlink.
    """
    if dest.is_symlink():
        raise RuntimeError(
            f"REFUSE: {dest} is a symlink. Won't write through it.")

    # Write tmp next to dest so the atomic-rename stays on the same filesystem
    # (and so it survives a node-local /tmp clean).
    tmp = dest.with_name(dest.name + ".restore_tmp")
    try:
        pq.write_table(table, tmp, compression="snappy")
        # Verify the tmp read-back matches target schema
        verify = pq.ParquetFile(tmp)
        if verify.schema_arrow.names != table.schema.names:
            raise RuntimeError(
                f"Schema mismatch on write-back: got {verify.schema_arrow.names}")
        print(f"  Wrote tmp {tmp} ({verify.metadata.num_rows} rows, "
              f"{tmp.stat().st_size / 1024 / 1024:.1f} MB)")
        shutil.move(str(tmp), str(dest))
        print(f"  Replaced: {dest}")
    finally:
        if tmp.exists():
            tmp.unlink()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=list(TARGETS), required=True,
                    help="Language to restore: 'pl' or 'zh-CN'")
    ap.add_argument("--dry-run", action="store_true",
                    help="Download + transform + write to /tmp, but don't replace the store file.")
    args = ap.parse_args()

    dest = Path(TARGETS[args.lang])
    print(f"Restoring {args.lang} → {dest}")
    print(f"Dry-run: {args.dry_run}")

    target = load_target_schema()
    ds = download_hf(args.lang)
    table = transform(ds, target)

    print(f"\nBuilt table: {table.num_rows} rows, schema:")
    for i, f in enumerate(table.schema):
        print(f"  [{i}] {f.name}: {f.type}")

    if args.dry_run:
        # /tmp is node-local on Clariden — write to shared scratch so the verify
        # job (on a different node) can read it.
        out_dir = Path(os.environ.get(
            "SCRATCH",
            f"/capstor/scratch/cscs/{os.environ.get('USER', 'unknown')}"
        )) / "cv25_restore_dryrun"
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / f"cv25_{args.lang.replace('-', '_')}_test_RESTORED.parquet"
        pq.write_table(table, tmp, compression="snappy")
        print(f"\nDRY RUN — wrote to {tmp} ({tmp.stat().st_size / 1024 / 1024:.1f} MB)")
        print("Verify it by hand, then re-run without --dry-run to atomic-replace the store file.")
        return

    safe_replace(table, dest)
    print(f"\nDone. {dest} restored.")


if __name__ == "__main__":
    main()
