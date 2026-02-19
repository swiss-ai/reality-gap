#!/usr/bin/env python3
"""Shared helpers for dataset preparation scripts (HF/WDS -> Shar)."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SUCCESS_MARKER_FILE = "_SUCCESS"


def load_text_tokenizer(tokenizer_path: str | Path):
    """Load a Rust fast tokenizer from a tokenizer.json file.
    Returns None if tokenizer_path is None.
    """
    if tokenizer_path is None:
        return None
    import logging
    from tokenizers import Tokenizer
    path = Path(tokenizer_path)
    if not path.is_file():
        raise FileNotFoundError(f"Text tokenizer not found: {path}")
    tok = Tokenizer.from_file(str(path))
    logging.getLogger(__name__).info(f"Text pre-tokenization enabled: {path}")
    return tok


def make_text_tokenize_fn(tokenizer):
    """Return a lhotse cut map function that tokenizes supervision text.
    Stores result as cut.custom["text_tokens"] (list[int]).
    """
    import logging
    _logger = logging.getLogger(__name__)

    def _tokenize_text(cut):
        texts = [s.text for s in (cut.supervisions or []) if s.text]
        if not texts:
            return cut
        if len(texts) > 1:
            _logger.debug(
                "Cut %s: merging %d supervision texts into one", cut.id, len(texts)
            )
        ids = tokenizer.encode(" ".join(texts), add_special_tokens=False).ids
        cut.custom = cut.custom or {}
        cut.custom["text_tokens"] = ids
        return cut
    return _tokenize_text


def to_mono(cut, mono_downmix=True, stats=None):
    """Convert a multi-channel cut to mono.

    If ``mono_downmix`` is True, tries averaging channels first; falls back to
    channel 0 on decode errors.  If False, always takes channel 0.

    ``stats`` is an optional ``Counter`` for tracking fallback events.
    """
    if cut.num_channels <= 1:
        return cut
    if mono_downmix:
        try:
            result = cut.to_mono(mono_downmix=True)
            # Force-load to catch AudioLoadingError from broken headers now,
            # rather than letting it bubble up later in SharWriter.
            result.load_audio()
            return result
        except Exception:
            if stats is not None:
                stats["downmix_fallback_ch0"] += 1
            # Broken header — fall back to channel 0.
    result = cut.to_mono(mono_downmix=False)
    if isinstance(result, list):
        return result[0]  # Take first channel (channel 0)
    return result


def setup_partition_dir(
    part_dir: Path,
    *,
    success_marker_name: str = SUCCESS_MARKER_FILE,
    reuse_log: str | None = None,
    reset_log: str | None = None,
    logger=None,
) -> bool:
    """Prepare a partition directory for resume-safe writing.

    Returns:
        True if the partition is already complete and should be reused.
        False if caller should (re)process and write this partition.
    """
    success_marker = part_dir / success_marker_name
    if success_marker.is_file():
        if logger and reuse_log:
            logger.info(reuse_log)
        return True

    # If marker is missing, any leftover files are considered partial output.
    if part_dir.is_dir():
        if logger and reset_log:
            logger.warning(reset_log)
        shutil.rmtree(part_dir)

    part_dir.mkdir(parents=True, exist_ok=True)
    return False


def mark_partition_success(
    part_dir: Path,
    *,
    success_marker_name: str = SUCCESS_MARKER_FILE,
) -> None:
    """Atomically mark a partition as fully prepared."""
    (part_dir / success_marker_name).write_text("ok\n")


def validate_or_write_prepare_state(
    state_path: Path,
    *,
    expected: Mapping[str, object],
    invariant_keys: Sequence[str],
    guidance: str,
) -> bool:
    """Persist first-run state or assert resume invariants on later runs."""
    if state_path.is_file():
        payload = json.loads(state_path.read_text())
        if not isinstance(payload, dict):
            raise RuntimeError(f"Invalid prepare state format: {state_path}")

        for key in invariant_keys:
            prev = payload.get(key)
            cur = expected.get(key)
            if prev != cur:
                raise AssertionError(
                    "Unsafe resume detected: persisted configuration changed.\n"
                    f"State file: {state_path}\n"
                    f"Key: {key}\n"
                    f"Existing value: {prev!r}\n"
                    f"Current value: {cur!r}\n"
                    f"{guidance}"
                )
        return False

    state_path.write_text(json.dumps(dict(expected), indent=2) + "\n")
    return True


def build_shar_index_from_parts(
    *,
    shar_root: Path,
    part_dirs: Iterable[Path],
    index_filename: str,
    success_marker_name: str = SUCCESS_MARKER_FILE,
) -> tuple[Path, int]:
    """Build a merged ``shar_index.json`` from expected partition directories."""
    fields = defaultdict(list)

    for part_dir in part_dirs:
        if not part_dir.is_dir():
            raise FileNotFoundError(f"Missing partition directory: {part_dir}")

        success_marker = part_dir / success_marker_name
        if not success_marker.is_file():
            raise RuntimeError(
                f"Missing completion marker in {part_dir}. "
                "Partial partition detected; resume is unsafe."
            )

        for p in sorted(part_dir.iterdir()):
            if not p.is_file() or p.name == success_marker_name:
                continue
            field = p.name.split(".")[0]
            if field == "cuts" and p.suffix == ".gz":
                fields["cuts"].append(str(p))
            elif p.suffix in (".tar", ".gz"):
                fields[field].append(str(p))

    if not fields.get("cuts"):
        raise FileNotFoundError(f"No Shar cuts found under {shar_root}")

    payload = {
        "version": 1,
        "fields": {k: sorted(v) for k, v in fields.items()},
    }
    index_path = shar_root / index_filename
    index_path.write_text(json.dumps(payload, indent=2))
    return index_path, len(fields["cuts"])


# ---------------------------------------------------------------------------
# Shared helpers for parallel prepare scripts (WDS, audio-dir, etc.)
# ---------------------------------------------------------------------------

WORKER_ASSIGNMENT_FILE = "_worker_assignment.json"
WORKER_STATS_FILE = "worker_stats.json"
PREPARE_SUMMARY_FILE = "prepare_summary.json"

logger = logging.getLogger(__name__)


def audio_md5(path: str) -> str:
    """MD5 of decoded audio waveform (float32 PCM, not raw file bytes)."""
    import soundfile as sf

    data, _ = sf.read(path, dtype="float32")
    return hashlib.md5(data.tobytes()).hexdigest()


def build_audio_index(audio_root: Path, pattern: str = "**/*.ogg") -> dict[str, str]:
    """Map lowercased file stems to full paths (recursive glob).

    Keys are lowercased for case-insensitive matching with
    ``canonical_sample_key`` used by the VAD / chunking pipeline.
    """
    return {p.stem.lower(): str(p) for p in audio_root.glob(pattern)}


def distribute_round_robin(items: Sequence, num_workers: int) -> list[list]:
    """Distribute items across workers in round-robin order."""
    buckets: list[list] = [[] for _ in range(num_workers)]
    for i, item in enumerate(items):
        buckets[i % num_workers].append(item)
    return buckets


def build_shar_index(
    shar_root: Path,
    num_workers: int,
    index_filename: str = "shar_index.json",
    worker_dir_fmt: str = "worker_{:02d}",
) -> None:
    """Build a merged ``shar_index.json`` from all ``worker_*`` directories.

    The index maps field names (``cuts``, ``recording``, ...) to sorted lists
    of absolute file paths, so that ``CutSet.from_shar(fields=...)`` can load
    all worker outputs as a single logical CutSet.
    """
    worker_dirs = [shar_root / worker_dir_fmt.format(wid) for wid in range(num_workers)]
    index_path, cuts_count = build_shar_index_from_parts(
        shar_root=shar_root,
        part_dirs=worker_dirs,
        index_filename=index_filename,
        success_marker_name=SUCCESS_MARKER_FILE,
    )
    logger.info(f"Wrote merged index: {index_path} ({cuts_count} cut shards)")


def load_worker_assignment(
    shar_dir: Path,
    *,
    items_key: str = "resolved_items",
) -> dict | None:
    """Load a persisted worker assignment from ``_worker_assignment.json``.

    *items_key* is the JSON key storing the list of input items
    (e.g. ``"resolved_shards"`` for WDS, ``"resolved_jsonls"`` for audio-dir).
    """
    path = shar_dir / WORKER_ASSIGNMENT_FILE
    if not path.is_file():
        return None

    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid assignment file format: {path}")

    try:
        num_workers = int(payload["num_workers"])
        resolved = payload[items_key]
    except KeyError as e:
        raise RuntimeError(
            f"Invalid assignment file (missing key {e.args[0]}): {path}"
        ) from e

    if num_workers < 1:
        raise RuntimeError(f"Invalid num_workers in assignment file: {path}")
    if not isinstance(resolved, list):
        raise RuntimeError(f"Invalid {items_key} in assignment file: {path}")

    return {
        "path": path,
        "num_workers": num_workers,
        items_key: [str(p) for p in resolved],
    }


def write_worker_assignment(
    shar_dir: Path,
    num_workers: int,
    resolved_items: Sequence,
    *,
    items_key: str = "resolved_items",
) -> Path:
    """Persist worker assignment for resume safety."""
    path = shar_dir / WORKER_ASSIGNMENT_FILE
    payload = {
        "version": 1,
        "num_workers": int(num_workers),
        items_key: list(resolved_items),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def run_aggregate(shar_root: Path) -> None:
    """Read prepare_summary.json from all node_*/ dirs, sum totals, and print."""
    node_dirs = sorted(shar_root.glob("node_*"))
    if not node_dirs:
        single = shar_root / PREPARE_SUMMARY_FILE
        if single.is_file():
            node_dirs = [shar_root]
        else:
            raise FileNotFoundError(
                f"No node_*/ dirs (or {PREPARE_SUMMARY_FILE}) found under {shar_root}"
            )

    summaries = []
    for nd in node_dirs:
        sp = nd / PREPARE_SUMMARY_FILE
        if not sp.is_file():
            logger.warning(f"Missing {sp}, skipping")
            continue
        summaries.append(json.loads(sp.read_text()))

    if not summaries:
        raise FileNotFoundError(f"No {PREPARE_SUMMARY_FILE} found in any node dir")

    total_written = 0
    total_skipped = 0
    total_errors = 0
    total_duration_sec = 0.0
    total_elapsed_sec = 0.0
    agg_reason: Counter = Counter()
    agg_runtime: Counter = Counter()

    for s in summaries:
        total_written += s.get("total_written", 0)
        total_skipped += s.get("total_skipped", 0)
        total_errors += s.get("total_errors", 0)
        total_duration_sec += s.get("total_duration_sec", 0.0)
        total_elapsed_sec = max(total_elapsed_sec, s.get("elapsed_sec", 0.0))
        agg_reason.update(s.get("reason_counts", {}))
        agg_runtime.update(s.get("runtime_counts", {}))

    total_hours = total_duration_sec / 3600.0

    print()
    print(f"=== Aggregate stats from {len(summaries)} node(s) under {shar_root} ===")
    print(f"  Samples written:  {total_written:>12d}")
    print(f"  Samples skipped:  {total_skipped:>12d}")
    print(f"  Errors:           {total_errors:>12d}")
    print(f"  Total hours:      {total_hours:>12.1f}")
    print(f"  Max wall-time:    {total_elapsed_sec:>12.1f}s")
    if agg_reason:
        print(f"  VAD reasons:      {dict(agg_reason)}")
    if agg_runtime:
        print(f"  Runtime counters: {dict(agg_runtime)}")
    print()
