#!/usr/bin/env python3
"""Convert a Lhotse Shar directory to a HF-style parquet.

Reads:
  <shar-in>/
    shar_index.json
    cuts.000000.jsonl.gz   - per-utt metadata (id, text, supervision)
    recording.000000.tar   - WAV bytes per utt

Writes:
  <parquet-out>            - one row per utterance, columns:
    id        : string                                    - utterance id
    text      : string                                    - raw Polish text
    duration  : float64                                   - seconds (from cut)
    audio     : struct<bytes: binary, sampling_rate: int64> - HF audio struct
    language  : string                                    - "pl"

Schema mirrors what audio_tokenization/prepare/prepare_parquet_to_shar.py
(supervisor's batch_tok branch) expects. Notable: NO `text_tokens` column —
his pipeline tokenizes from `text` via load_text_tokenizer. NO `sample_rate`
sibling column — it lives inside the audio struct.
"""

import argparse
import gzip
import io
import json
import logging
import sys
import tarfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def load_cuts(cuts_path: Path) -> list[dict]:
    """Read the per-utt JSONL from cuts.000000.jsonl.gz."""
    cuts = []
    with gzip.open(cuts_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cuts.append(json.loads(line))
    return cuts


def cut_text(cut: dict) -> str:
    """Extract the text from a cut's first supervision."""
    sups = cut.get("supervisions") or []
    if not sups:
        raise ValueError(f"Cut {cut.get('id')} has no supervisions")
    text = sups[0].get("text")
    if text is None:
        raise ValueError(f"Cut {cut.get('id')} supervision has no text")
    return text


def cut_recording_id(cut: dict) -> str:
    """The recording id inside the tar — what filenames are keyed on."""
    rec = cut.get("recording") or {}
    return rec.get("id") or cut.get("id")


_AUDIO_SUFFIXES = {".wav", ".flac", ".opus", ".ogg", ".mp3"}


def index_recording_tar(tar_path: Path) -> dict[str, bytes]:
    """Read recording.000000.tar once into memory: {member_basename_noext: wav_bytes}.

    Lhotse Shar tars typically name members like <recording_id>.flac or
    <recording_id>.wav. We key by stem so callers can look up by recording_id.

    Filter to audio extensions only: Lhotse SharWriter also writes a tiny
    `<recording_id>.json` sidecar into the same tar. Both share `Path.stem`,
    so an unfiltered walk silently overwrites the audio bytes with the JSON
    metadata (~200 B). Confirmed bug 2026-05-17 on synthesize_to_shar output.
    """
    out: dict[str, bytes] = {}
    with tarfile.open(tar_path, "r") as tf:
        for member in tf:
            if not member.isfile():
                continue
            if Path(member.name).suffix.lower() not in _AUDIO_SUFFIXES:
                continue
            stem = Path(member.name).stem
            fh = tf.extractfile(member)
            if fh is None:
                continue
            out[stem] = fh.read()
    return out


def _convert_one_shard(cuts_path: Path, tar_path: Path,
                       parquet_out: Path) -> dict:
    """Convert one (cuts.NNNNNN.jsonl.gz, recording.NNNNNN.tar) pair."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    logger.info("Reading cuts manifest: %s", cuts_path)
    cuts = load_cuts(cuts_path)
    logger.info("  %d cuts", len(cuts))

    logger.info("Indexing recording tar: %s", tar_path)
    wav_index = index_recording_tar(tar_path)
    logger.info("  %d wav members", len(wav_index))

    ids: list[str] = []
    texts: list[str] = []
    audio_bytes_list: list[bytes] = []
    audio_srs: list[int] = []
    durs: list[float] = []
    langs: list[str] = []

    missing_audio = 0
    for cut in cuts:
        cid = cut["id"]
        text = cut_text(cut)
        rec_id = cut_recording_id(cut)
        wav_bytes = wav_index.get(rec_id) or wav_index.get(cid)
        if wav_bytes is None:
            missing_audio += 1
            logger.warning("No WAV for cut %s (recording_id=%s)", cid, rec_id)
            continue

        rec = cut.get("recording") or {}
        sr = int(rec.get("sampling_rate") or 24000)
        duration = float(cut.get("duration") or 0.0)
        lang = (cut.get("supervisions") or [{}])[0].get("language") or "pl"

        ids.append(cid)
        texts.append(text)
        audio_bytes_list.append(wav_bytes)
        audio_srs.append(sr)
        durs.append(duration)
        langs.append(lang)

    if missing_audio:
        logger.warning("%d cuts missing audio — skipped", missing_audio)

    # Schema mirrors batch_tok prepare_parquet_to_shar.py expectations:
    #   audio: struct<bytes: Binary, sampling_rate: Int64>
    #   id (str), text (optional), duration (optional), language (optional)
    # No text_tokens — his pipeline tokenizes from `text` via load_text_tokenizer.
    audio_type = pa.struct([
        pa.field("bytes", pa.binary()),
        pa.field("sampling_rate", pa.int64()),
    ])
    audio_struct = pa.array(
        [{"bytes": b, "sampling_rate": sr} for b, sr in zip(audio_bytes_list, audio_srs)],
        type=audio_type,
    )

    table = pa.table({
        "id": pa.array(ids, type=pa.string()),
        "text": pa.array(texts, type=pa.string()),
        "duration": pa.array(durs, type=pa.float64()),
        "audio": audio_struct,
        "language": pa.array(langs, type=pa.string()),
    })

    parquet_out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, parquet_out, compression="snappy")
    logger.info("Wrote %s (%d rows, %.1f MB)",
                parquet_out, len(ids), parquet_out.stat().st_size / 1024 / 1024)

    return {
        "n_rows": len(ids),
        "n_missing_audio": missing_audio,
        "parquet_size_mb": round(parquet_out.stat().st_size / 1024 / 1024, 2),
    }


def convert(shar_in: Path, parquet_out_dir: Path,
            shard_idx: int | None = None,
            total_shards: int | None = None) -> dict:
    """Convert (cuts, recording) pairs under shar_in to per-shard parquets.

    Walks **/cuts.*.jsonl.gz so it handles flat layout, shard_NNNN/ subdirs
    (synthesize_to_shar output), and Spark-style part-NNNNN/ layouts.

    Sequential mode (shard_idx=None): processes every discovered pair.
        Output: <parquet_out_dir>/train-NNNNN-of-MMMMM.parquet for each.

    Single-shard mode (shard_idx given): processes ONE pair — pair[shard_idx]
    after sorting. Output filename is train-{shard_idx:05d}-of-{total_shards:05d}.parquet.
    This is the array-job task path: one SLURM task per shard, memory-bounded,
    runs in parallel.
    """
    import re

    cuts_files = sorted(shar_in.rglob("cuts.*.jsonl.gz"))
    pairs: list[tuple[Path, Path]] = []
    for cuts_path in cuts_files:
        m = re.match(r"cuts\.(\d+)\.jsonl\.gz$", cuts_path.name)
        if not m:
            continue
        rec_tar = cuts_path.parent / f"recording.{m.group(1)}.tar"
        if not rec_tar.exists():
            logger.warning("No matching tar for %s, skipping", cuts_path)
            continue
        pairs.append((cuts_path, rec_tar))

    if not pairs:
        raise FileNotFoundError(f"No (cuts, recording) pairs found under {shar_in}")

    logger.info("Found %d Shar shard(s) under %s", len(pairs), shar_in)
    parquet_out_dir.mkdir(parents=True, exist_ok=True)

    # Slice to one pair if single-shard mode; mirror naming behaviour.
    if shard_idx is not None:
        if total_shards is None:
            raise ValueError("total_shards required when shard_idx is set")
        if shard_idx < 0 or shard_idx >= len(pairs):
            raise IndexError(
                f"shard_idx={shard_idx} out of range; found {len(pairs)} pairs")
        pairs_to_process = [(shard_idx, pairs[shard_idx])]
        n_for_name = total_shards
    else:
        pairs_to_process = list(enumerate(pairs))
        n_for_name = len(pairs)

    total_rows = 0
    total_missing = 0
    total_mb = 0.0
    for idx, (cuts_path, tar_path) in pairs_to_process:
        out_file = parquet_out_dir / f"train-{idx:05d}-of-{n_for_name:05d}.parquet"
        stats = _convert_one_shard(cuts_path, tar_path, out_file)
        total_rows += stats["n_rows"]
        total_missing += stats["n_missing_audio"]
        total_mb += stats["parquet_size_mb"]

    return {
        "n_shards": len(pairs_to_process),
        "n_rows": total_rows,
        "n_missing_audio": total_missing,
        "parquet_total_mb": round(total_mb, 2),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shar-in", required=True, type=Path,
                   help="Lhotse Shar directory to read (walks subdirs).")
    p.add_argument("--parquet-out-dir", required=True, type=Path,
                   help="Output directory for per-shard parquets.")
    p.add_argument("--shard-idx", type=int, default=None,
                   help="Single-shard mode: 0-based index into the sorted list "
                        "of discovered shar shards. Used by array SLURM jobs "
                        "(one task per shard). Requires --total-shards.")
    p.add_argument("--total-shards", type=int, default=None,
                   help="Required with --shard-idx. The total shard count "
                        "across the array — appears in the output filename's "
                        "of-NNNNN suffix.")
    # Back-compat: --parquet-out is now treated as a dir
    p.add_argument("--parquet-out", dest="parquet_out_dir", type=Path,
                   help=argparse.SUPPRESS)
    args = p.parse_args()

    if not args.shar_in.exists():
        logger.error("Shar dir not found: %s", args.shar_in)
        sys.exit(2)

    stats = convert(args.shar_in, args.parquet_out_dir,
                    shard_idx=args.shard_idx, total_shards=args.total_shards)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
