#!/usr/bin/env python3
"""Convert a Lhotse Shar directory to a single parquet with tokenized text.

Reads:
  <shar-in>/
    shar_index.json
    cuts.000000.jsonl.gz   - per-utt metadata (id, text, supervision)
    recording.000000.tar   - WAV bytes per utt

Writes:
  <parquet-out>            - one row per utterance, columns:
    id            : str          - utterance id
    text          : str          - raw text (unchanged from supervision)
    text_tokens   : list[int]    - joint tokenizer encode(text, add_special_tokens=False)
    audio         : bytes        - WAV bytes verbatim (24 kHz PCM_16 mono)
    sample_rate   : int          - 24000
    duration_s    : float        - from cut.duration
    language      : str          - "pl"

Schema follows the supervisor's spec (2026-05-14):
- Pre-tokenized text in `text_tokens` so the downstream pipeline doesn't
  re-tokenize.
- No special-token wrapping (BOS / <|speech_transcribe|> / EOS) — the
  pipeline adds those. Hence add_special_tokens=False.
- The text_tokens column matches the cut.custom["text_tokens"] convention
  used by his Lhotse-Shar path; parquet just flattens it to a column.
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

DEFAULT_TOKENIZER = "/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok"


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


def convert(shar_in: Path, parquet_out: Path, tokenizer_path: str) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq
    # Use the raw `tokenizers` library, not transformers' AutoTokenizer. The
    # voxcpm2 venv ships transformers 4.55 which pulls in `masking_utils` ->
    # `torch._dynamo._trace_wrapped_higher_order_op.TransformGetItemToIndex`,
    # a torch 2.7+ symbol that NGC 24.11 (torch 2.6) lacks. Raw `tokenizers`
    # has no torch dependency. We don't need special-token wrapping anyway
    # (supervisor said add_special_tokens=False), so vocab+encode is enough.
    from tokenizers import Tokenizer

    cuts_path = shar_in / "cuts.000000.jsonl.gz"
    tar_path = shar_in / "recording.000000.tar"
    if not cuts_path.exists() or not tar_path.exists():
        raise FileNotFoundError(f"Expected cuts + recording in {shar_in}")

    tokenizer_json = Path(tokenizer_path) / "tokenizer.json"
    if not tokenizer_json.exists():
        raise FileNotFoundError(f"tokenizer.json not found in {tokenizer_path}")
    logger.info("Loading tokenizer: %s", tokenizer_json)
    tokenizer = Tokenizer.from_file(str(tokenizer_json))

    logger.info("Reading cuts manifest: %s", cuts_path)
    cuts = load_cuts(cuts_path)
    logger.info("  %d cuts", len(cuts))

    logger.info("Indexing recording tar: %s", tar_path)
    wav_index = index_recording_tar(tar_path)
    logger.info("  %d wav members", len(wav_index))

    ids: list[str] = []
    texts: list[str] = []
    token_lists: list[list[int]] = []
    audios: list[bytes] = []
    srs: list[int] = []
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

        # `tokenizers.Tokenizer.encode` does not add special tokens by default
        # (parity with `AutoTokenizer.encode(..., add_special_tokens=False)`).
        # Returns an Encoding object; we take .ids.
        token_ids = tokenizer.encode(text).ids

        # Sample rate + duration: prefer the recording's sampling_rate; fall back
        # to cut.duration. Lhotse stores both, but the canonical source is the
        # Recording for sample_rate.
        rec = cut.get("recording") or {}
        sr = int(rec.get("sampling_rate") or 24000)
        duration = float(cut.get("duration") or 0.0)
        lang = (cut.get("supervisions") or [{}])[0].get("language") or "pl"

        ids.append(cid)
        texts.append(text)
        token_lists.append(token_ids)
        audios.append(wav_bytes)
        srs.append(sr)
        durs.append(duration)
        langs.append(lang)

    if missing_audio:
        logger.warning("%d cuts missing audio — skipped", missing_audio)

    table = pa.table({
        "id": pa.array(ids, type=pa.string()),
        "text": pa.array(texts, type=pa.string()),
        "text_tokens": pa.array(token_lists, type=pa.list_(pa.int32())),
        "audio": pa.array(audios, type=pa.binary()),
        "sample_rate": pa.array(srs, type=pa.int32()),
        "duration_s": pa.array(durs, type=pa.float32()),
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
    p.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER,
                   help=f"Joint tokenizer path (default: {DEFAULT_TOKENIZER})")
    args = p.parse_args()

    if not args.shar_in.exists():
        logger.error("Shar dir not found: %s", args.shar_in)
        sys.exit(2)

    stats = convert(args.shar_in, args.parquet_out, args.tokenizer_path)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
