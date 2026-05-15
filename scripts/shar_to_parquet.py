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


def index_recording_tar(tar_path: Path) -> dict[str, bytes]:
    """Read recording.000000.tar once into memory: {member_basename_noext: wav_bytes}.

    Lhotse Shar tars typically name members like <recording_id>.flac or
    <recording_id>.wav. We key by stem so callers can look up by recording_id.
    """
    out: dict[str, bytes] = {}
    with tarfile.open(tar_path, "r") as tf:
        for member in tf:
            if not member.isfile():
                continue
            stem = Path(member.name).stem
            fh = tf.extractfile(member)
            if fh is None:
                continue
            out[stem] = fh.read()
    return out


def convert(shar_in: Path, parquet_out: Path) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    cuts_path = shar_in / "cuts.000000.jsonl.gz"
    tar_path = shar_in / "recording.000000.tar"
    if not cuts_path.exists() or not tar_path.exists():
        raise FileNotFoundError(f"Expected cuts + recording in {shar_in}")

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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shar-in", required=True, type=Path,
                   help="Lhotse Shar directory to read")
    p.add_argument("--parquet-out", required=True, type=Path,
                   help="Output .parquet file path")
    args = p.parse_args()

    if not args.shar_in.exists():
        logger.error("Shar dir not found: %s", args.shar_in)
        sys.exit(2)

    stats = convert(args.shar_in, args.parquet_out)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
