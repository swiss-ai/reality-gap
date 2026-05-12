#!/usr/bin/env python3
"""Verify a Polish dataset is loadable and pull a TTS test-set JSON from it.

Same JSON schema as `data/tts_bench/pl_50.json`:
    [{"id": "...", "category": "...", "text": "..."}, ...]

Supports two source formats (auto-detected):
  - Lhotse Shar  (directory with shar_index.json)
  - HF Dataset   (directory with dataset_info.json / dataset_dict.json)

Usage:
    # Dry-run (load, print stats, no JSON written)
    python scripts/extract_tts_test_set.py --source PATH --verify-only

    # Pull N samples and write a test set
    python scripts/extract_tts_test_set.py \\
        --source /capstor/.../SHAR/stage_2/voxpopuli_asr/pl \\
        --n 100 \\
        --output data/tts_bench/pl_voxpopuli_100.json \\
        --category voxpopuli \\
        --min-words 3 --max-words 60

Recommended verification batch (all the candidates):
    for D in \\
        /capstor/store/cscs/swissai/infra01/audio-datasets/benchmark/fleurs_cache/pl_pl \\
        /capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/voxpopuli_asr/pl \\
        /capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/mls/mls_polish_train \\
        /capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/commonvoice/pl_train \\
        /capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/granary_ytc_asr/pl ; do
      python scripts/extract_tts_test_set.py --source "$D" --verify-only
    done
"""

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)


# ── Loaders ──────────────────────────────────────────────────────────
def _looks_like_shar(p: Path) -> bool:
    return (p / "shar_index.json").exists() or (p / "_SUCCESS").exists()


def _looks_like_hf(p: Path) -> bool:
    return (p / "dataset_info.json").exists() or (p / "dataset_dict.json").exists()


def iter_lhotse_shar(path: Path, limit: int):
    """Yield (id, text, duration_seconds) tuples from a Lhotse Shar dir."""
    from lhotse import CutSet

    cuts = CutSet.from_shar(in_dir=str(path))

    n = 0
    for cut in cuts:
        if n >= limit and limit > 0:
            return
        text = None
        if cut.supervisions:
            text = cut.supervisions[0].text
        if not text:
            continue
        yield (cut.id, text.strip(), float(cut.duration))
        n += 1


def iter_hf(path: Path, limit: int):
    """Yield (id, text, duration_seconds) from an HF dataset dir."""
    from datasets import Audio, load_from_disk

    ds = load_from_disk(str(path))
    # Handle DatasetDict — pick a sensible split
    if hasattr(ds, "keys") and not hasattr(ds, "column_names"):
        for k in ["train", "test", "validation"]:
            if k in ds:
                ds = ds[k]
                break
        else:
            ds = ds[next(iter(ds))]

    cols = ds.column_names
    text_col = next(
        (c for c in ("transcription", "sentence", "text", "raw_transcription") if c in cols),
        None,
    )
    if text_col is None:
        raise RuntimeError(f"No known text column in {cols}")

    # Don't decode audio (torchcodec missing on most containers); we only need text.
    if "audio" in cols:
        ds = ds.cast_column("audio", Audio(decode=False))

    n = 0
    for i, sample in enumerate(ds):
        if n >= limit and limit > 0:
            return
        text = sample.get(text_col)
        if not text:
            continue
        # Duration: HF audio dicts may carry sampling_rate + array length, but
        # we asked them not to decode. Best-effort from `duration` field if present.
        dur = float(sample.get("duration") or sample.get("duration_seconds") or 0.0)
        yield (f"{path.name}_{i:06d}", text.strip(), dur)
        n += 1


def detect_and_iter(source: Path, limit: int):
    if _looks_like_shar(source):
        logger.info("Source detected as Lhotse Shar: %s", source)
        return iter_lhotse_shar(source, limit), "shar"
    if _looks_like_hf(source):
        logger.info("Source detected as HF dataset: %s", source)
        return iter_hf(source, limit), "hf"
    raise RuntimeError(
        f"Could not detect source format at {source} "
        "(no shar_index.json/_SUCCESS, no dataset_info.json/dataset_dict.json)."
    )


# ── Stats ────────────────────────────────────────────────────────────
def _summarize(samples: list[dict]) -> dict:
    if not samples:
        return {"n": 0}
    word_counts = [len(WORD_RE.findall(s["text"])) for s in samples]
    return {
        "n": len(samples),
        "median_words": int(sorted(word_counts)[len(word_counts) // 2]),
        "min_words": min(word_counts),
        "max_words": max(word_counts),
        "median_duration_s": (
            round(sorted(s["duration_seconds"] for s in samples)[len(samples) // 2], 2)
            if samples[0].get("duration_seconds") else None
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, type=Path, help="Path to dataset")
    p.add_argument("--n", type=int, default=100, help="Number of utterances to extract")
    p.add_argument("--output", type=Path, default=None, help="Output JSON path (omit for verify-only)")
    p.add_argument("--category", default="extracted", help="Category label for the test set")
    p.add_argument("--min-words", type=int, default=3, help="Drop utterances shorter than this")
    p.add_argument("--max-words", type=int, default=60, help="Drop utterances longer than this")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verify-only", action="store_true",
                   help="Just verify loading + print stats, write nothing")
    p.add_argument("--scan-limit", type=int, default=20000,
                   help="Max records to scan when picking N (to bound time on huge datasets)")
    args = p.parse_args()

    if not args.source.exists():
        logger.error("Source path does not exist: %s", args.source)
        sys.exit(2)

    # Pull a wide pool, filter for length, then sample N.
    pool: list[dict] = []
    try:
        it, kind = detect_and_iter(args.source, args.scan_limit)
        for sid, text, dur in it:
            wc = len(WORD_RE.findall(text))
            if args.min_words <= wc <= args.max_words:
                pool.append({"id": sid, "text": text, "duration_seconds": dur})
    except Exception as e:
        logger.error("FAILED to load %s: %s", args.source, e, exc_info=True)
        sys.exit(3)

    logger.info("Loaded %d eligible records from %s (%s)", len(pool), args.source, kind)

    if not pool:
        logger.error("No eligible utterances after filtering. Source may be empty or all text out of range.")
        sys.exit(4)

    # Print examples + stats
    rng = random.Random(args.seed)
    print(f"\n=== {args.source} ===")
    print(f"  format:        {kind}")
    print(f"  eligible_pool: {len(pool)} (scan_limit={args.scan_limit})")
    stats = _summarize(pool)
    for k, v in stats.items():
        if v is not None:
            print(f"  {k:<15}: {v}")
    print("  examples:")
    for s in rng.sample(pool, min(5, len(pool))):
        snippet = s["text"][:140] + ("…" if len(s["text"]) > 140 else "")
        print(f"    [{s['id']}] {snippet}")

    if args.verify_only or args.output is None:
        return

    n = min(args.n, len(pool))
    chosen = rng.sample(pool, n)
    out = [
        {
            "id": f"{args.category}_{i:04d}",
            "category": args.category,
            "text": s["text"],
            "source_id": s["id"],
        }
        for i, s in enumerate(chosen)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %d records to %s", n, args.output)


if __name__ == "__main__":
    main()
