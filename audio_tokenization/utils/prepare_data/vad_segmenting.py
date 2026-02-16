#!/usr/bin/env python3
"""VAD-aware chunking helpers for WDS -> Shar preparation.

Split long recordings into speech-aware chunks with hard caps:
- no chunk longer than ``max_chunk_sec``
- drop chunks shorter than ``min_chunk_sec``

VAD timestamps are read from per-shard JSONL files produced by
``filter_langid_vad.py``. Each worker loads only the per-shard files
that correspond to its assigned tar shards — no pre-build step needed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class VADChunkingConfig:
    """Config for speech-aware chunking.

    max_chunk_sec: Target maximum for packing multiple segments. Single
        continuous speech segments CAN exceed this and will be kept intact.
    min_chunk_sec: Drop chunks shorter than this (too fragmented).
    sample_rate: Sample rate VAD was run at (for timestamp conversion).
    max_merge_gap_sec: Merge segments if gap is <= this threshold.
    """

    max_chunk_sec: float = 200.0
    min_chunk_sec: float = 10.0
    sample_rate: int = 16000
    max_merge_gap_sec: float = 0.5


def canonical_sample_key(key: str) -> str:
    """Normalize sample keys for stable matching across WDS/metadata/VAD."""
    return key.strip().lower()


def shard_name_from_tar_path(tar_path: str | Path) -> str:
    """Match the shard naming convention used by the WDS pipeline."""
    p = Path(tar_path)
    return f"{p.parent.name}/{p.stem}".lower()


def vad_per_shard_file(vad_per_shard_dir: Path, shard_name: str) -> Path:
    """Return per-shard VAD JSONL path (audio/000001 -> audio_000001.jsonl)."""
    return vad_per_shard_dir / f"{shard_name.replace('/', '_')}.jsonl"


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

def _normalize_raw_timestamps(raw_timestamps: Any) -> List[Tuple[int, int]]:
    """Normalize timestamps into sorted [(start, end), ...] sample-index pairs."""
    normalized: List[Tuple[int, int]] = []
    if not isinstance(raw_timestamps, list):
        return normalized

    for item in raw_timestamps:
        if isinstance(item, dict):
            start = item.get("start")
            end = item.get("end")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            start, end = item[0], item[1]
        else:
            continue

        try:
            start_i = int(start)
            end_i = int(end)
        except (TypeError, ValueError):
            continue

        if end_i <= start_i:
            continue
        normalized.append((start_i, end_i))

    normalized.sort(key=lambda x: x[0])
    return normalized


def _parse_vad_jsonl_line(
    line: str,
    *,
    with_duration: bool = False,
    with_sample_rate: bool = False,
    with_lang: bool = False,
) -> Optional[Tuple]:
    """Parse one line from a per-shard VAD JSONL file.

    Returns ``(normalized_key, timestamps)`` by default.  When
    *with_duration* is True, returns ``(key, timestamps, duration_sec)``
    instead and skips lines that lack a positive ``duration_sec``.
    When *with_sample_rate* is also True, appends ``sample_rate``
    (int or None) to the tuple.  When *with_lang* is True, appends
    ``lang`` (str, defaults to ``"unknown"``) as the last element.
    """
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict) or len(payload) != 1:
        return None

    ((raw_key, raw_value),) = payload.items()
    if not isinstance(raw_value, dict):
        return None

    key = canonical_sample_key(str(raw_key))
    timestamps = _normalize_raw_timestamps(raw_value.get("timestamps", []))

    result = (key, timestamps)

    if with_duration:
        duration_sec = raw_value.get("duration_sec", 0.0)
        try:
            duration_sec = float(duration_sec)
        except (TypeError, ValueError):
            return None
        if duration_sec <= 0:
            return None
        result = result + (duration_sec,)
        if with_sample_rate:
            sr_raw = raw_value.get("sample_rate")
            try:
                sr = int(sr_raw) if sr_raw is not None else None
            except (TypeError, ValueError):
                sr = None
            result = result + (sr,)

    if with_lang:
        result = result + (raw_value.get("lang", "unknown"),)

    return result


# ---------------------------------------------------------------------------
# Per-shard VAD loading
# ---------------------------------------------------------------------------

def load_vad_from_per_shard_dir(
    vad_per_shard_dir: Path,
    tar_paths: Sequence[str],
    *,
    with_lang: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, List[Tuple[int, int]]] | Tuple[Dict[str, List[Tuple[int, int]]], Dict[str, str]]:
    """Load VAD entries from per-shard JSONL files for the given tar shards.

    Each worker calls this with its own tar_paths and gets only the
    relevant VAD entries. No pre-build step or cache needed.

    When *with_lang* is True, returns ``(vad_lookup, lang_lookup)`` where
    ``lang_lookup`` maps normalized sample keys to language codes.
    """
    lookup: Dict[str, List[Tuple[int, int]]] = {}
    lang_lookup: Dict[str, str] = {}
    files_read = 0
    for tar_path in tar_paths:
        shard_name = shard_name_from_tar_path(tar_path)
        vad_file = vad_per_shard_file(vad_per_shard_dir, shard_name)
        if not vad_file.is_file():
            continue
        files_read += 1
        with open(vad_file, "r", encoding="utf-8") as f:
            for line in f:
                parsed = _parse_vad_jsonl_line(line, with_lang=with_lang)
                if parsed is None:
                    continue
                if with_lang:
                    key, timestamps, lang = parsed
                    lang_lookup[key] = lang
                else:
                    key, timestamps = parsed
                lookup[key] = timestamps
    if logger is not None:
        logger.info(
            f"Loaded {len(lookup)} VAD entries from "
            f"{files_read}/{len(tar_paths)} shard files"
        )
    if with_lang:
        return lookup, lang_lookup
    return lookup


# ---------------------------------------------------------------------------
# Chunking logic
# ---------------------------------------------------------------------------

def _timestamps_to_seconds(
    timestamps: Sequence[Tuple[int, int]],
    *,
    sample_rate: int,
    audio_duration_sec: float,
) -> List[Tuple[float, float]]:
    spans_sec: List[Tuple[float, float]] = []
    sr = float(sample_rate)
    for start_i, end_i in timestamps:
        start = max(0.0, float(start_i) / sr)
        end = min(audio_duration_sec, float(end_i) / sr)
        if end <= start:
            continue
        spans_sec.append((start, end))
    spans_sec.sort(key=lambda x: x[0])
    return spans_sec


def _merge_spans(
    spans: Sequence[Tuple[float, float]],
    *,
    max_gap_sec: float,
) -> List[Tuple[float, float]]:
    if not spans:
        return []
    merged: List[List[float]] = [[spans[0][0], spans[0][1]]]
    for start, end in spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + max_gap_sec:
            merged[-1][1] = max(prev_end, end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def build_chunk_ranges_from_vad(
    timestamps: Sequence[Tuple[int, int]],
    *,
    audio_duration_sec: float,
    cfg: VADChunkingConfig,
) -> List[Tuple[float, float]]:
    """Build speech-aware chunk ranges as (offset_sec, duration_sec).

    Packs complete VAD segments into chunks up to max_chunk_sec, merging
    nearby segments (gap <= max_merge_gap_sec). Avoids cutting mid-segment
    by stopping before max_chunk_sec if adding the next segment would exceed it.

    Note: Single continuous speech segments CAN exceed max_chunk_sec and will
    be kept intact. max_chunk_sec only applies when packing multiple segments.
    """
    spans = _timestamps_to_seconds(
        timestamps,
        sample_rate=cfg.sample_rate,
        audio_duration_sec=audio_duration_sec,
    )
    if not spans:
        return []

    chunks: List[Tuple[float, float]] = []

    # Pack segments into chunks, merging nearby segments and respecting max_chunk_sec
    i = 0
    while i < len(spans):
        chunk_start = spans[i][0]
        chunk_end = spans[i][1]
        i += 1

        # Try to add more segments to this chunk
        while i < len(spans):
            seg_start, seg_end = spans[i]
            gap = seg_start - chunk_end

            # Check if we can merge this segment
            if gap <= cfg.max_merge_gap_sec:
                # Merging would extend to seg_end, check if it fits
                new_duration = seg_end - chunk_start
                if new_duration <= cfg.max_chunk_sec:
                    # Fits! Extend chunk to include gap + segment
                    chunk_end = seg_end
                    i += 1
                else:
                    # Would exceed max_chunk_sec, stop here
                    break
            else:
                # Gap too large, start new chunk
                break

        # Emit chunk if it meets minimum duration
        chunk_duration = chunk_end - chunk_start
        if chunk_duration >= cfg.min_chunk_sec:
            chunks.append((chunk_start, chunk_duration))

    return chunks


def split_cut_by_vad(
    *,
    cut: Any,
    sample_key: str,
    vad_lookup: Dict[str, List[Tuple[int, int]]],
    cfg: VADChunkingConfig,
) -> Tuple[List[Any], str]:
    """Split one recording cut with VAD-aware policy.

    Policy:
    - duration < min_chunk_sec: drop
    - min_chunk_sec <= duration < max_chunk_sec: keep full cut (no VAD needed)
    - duration >= max_chunk_sec:
      - missing VAD entry: drop
      - empty VAD timestamps: drop (non-speech)
      - otherwise: produce speech chunks <= max_chunk_sec and >= min_chunk_sec
    """
    duration = float(getattr(cut, "duration", 0.0) or 0.0)
    if duration < cfg.min_chunk_sec:
        return [], "too_short"

    if duration < cfg.max_chunk_sec:
        return [cut], "kept_full_short_audio"

    key = canonical_sample_key(sample_key)
    timestamps = vad_lookup.get(key)
    if timestamps is None:
        return [], "missing_vad"
    if not timestamps:
        return [], "empty_vad"

    ranges = build_chunk_ranges_from_vad(
        timestamps=timestamps,
        audio_duration_sec=duration,
        cfg=cfg,
    )
    if not ranges:
        return [], "chunks_below_min_duration"

    out = []
    for offset, chunk_duration in ranges:
        try:
            # preserve_id=False avoids duplicate IDs when one source cut is split.
            subcut = cut.truncate(
                offset=offset, duration=chunk_duration, preserve_id=False
            )
        except TypeError:
            # Older Lhotse versions may not expose preserve_id.
            subcut = cut.truncate(offset=offset, duration=chunk_duration)

        # Store global offset so we know where this chunk came from
        subcut.custom = subcut.custom or {}
        subcut.custom["global_offset_sec"] = offset
        out.append(subcut)
    return out, "chunked"
