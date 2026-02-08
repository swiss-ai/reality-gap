#!/usr/bin/env python3
"""Minimal utilities to merge Megatron indexed datasets (.bin/.idx)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Sequence, Tuple


def _import_indexed_dataset_symbols():
    """Import Megatron dataset symbols lazily at runtime."""
    try:
        # Keep this import local so callers that only inspect args/help
        # do not require Megatron at module import time.
        from megatron.core.datasets.indexed_dataset import (  # pylint: disable=import-error
            IndexedDataset,
            IndexedDatasetBuilder,
            get_bin_path,
            get_idx_path,
        )
    except Exception as exc:
        raise ImportError(
            "Failed to import megatron.core.datasets.indexed_dataset from current Python env. "
            "Install Megatron in this environment."
        ) from exc

    return IndexedDataset, IndexedDatasetBuilder, get_bin_path, get_idx_path


def discover_indexed_prefixes(
    input_dirs: Sequence[str],
    recursive: bool = False,
) -> Tuple[List[str], List[str]]:
    """Return valid shard prefixes from one or more directories.

    A valid prefix is any ``*.bin`` with a sibling ``*.idx``. Missing pairs are
    treated as hard errors to avoid silently merging partial outputs.
    """
    prefix_paths: set[Path] = set()
    missing_idx_paths: List[Path] = []
    # Recursive mode scans nested rank/chunk layouts.
    pattern = "**/*.bin" if recursive else "*.bin"

    for input_dir in input_dirs:
        # Resolve once so dedupe/sort are stable and absolute.
        root = Path(input_dir).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Input directory not found: {root}")

        for bin_path in root.glob(pattern):
            idx_path = bin_path.with_suffix(".idx")
            if not idx_path.is_file():
                missing_idx_paths.append(bin_path)
                continue
            prefix_paths.add(bin_path.with_suffix(""))

    prefixes = sorted(str(path) for path in prefix_paths)
    missing_idx = sorted(str(path) for path in missing_idx_paths)

    # strict_pairs is mandatory: fail fast on any incomplete shard.
    if missing_idx:
        preview = "\n".join(missing_idx[:20])
        raise RuntimeError(
            f"Found {len(missing_idx)} .bin files without .idx.\n"
            f"First entries:\n{preview}"
        )

    return prefixes, missing_idx


def merge_indexed_dataset(
    input_dirs: Sequence[str],
    output_prefix: str,
    recursive: bool = False,
    multimodal: bool = False,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Merge discovered indexed shards into a single Megatron dataset."""
    log = logger or logging.getLogger(__name__)
    # Accept either "<prefix>" or "<prefix>.bin/.idx" and normalize to prefix.
    output_prefix_path = Path(output_prefix).expanduser().resolve()
    if output_prefix_path.suffix in {".bin", ".idx"}:
        output_prefix_path = output_prefix_path.with_suffix("")
    output_prefix_path.parent.mkdir(parents=True, exist_ok=True)
    output_prefix_str = str(output_prefix_path)

    # Import Megatron symbols only when a merge is actually requested.
    IndexedDataset, IndexedDatasetBuilder, get_bin_path, get_idx_path = _import_indexed_dataset_symbols()

    out_bin = Path(get_bin_path(output_prefix_str))
    out_idx = Path(get_idx_path(output_prefix_str))

    prefixes, missing_idx = discover_indexed_prefixes(
        input_dirs=input_dirs,
        recursive=recursive,
    )
    # If input dir includes output dir, avoid self-appending during re-runs.
    prefixes = [p for p in prefixes if p != output_prefix_str]

    if not prefixes:
        raise RuntimeError("No valid input prefixes found (.bin + .idx).")

    # Guard against accidental overwrite unless explicitly forced.
    if not force and (out_bin.exists() or out_idx.exists()):
        raise FileExistsError(
            f"Output already exists: {out_bin} or {out_idx}. Set merge.force=true to overwrite."
        )
    if force:
        for path in (out_bin, out_idx):
            if path.exists():
                path.unlink()

    log.info(
        "Merging %d prefixes into %s (Megatron source: %s)",
        len(prefixes),
        output_prefix_str,
        "python-env",
    )
    # Use the first shard to recover dtype and preserve binary compatibility.
    first = IndexedDataset(prefixes[0], multimodal=multimodal)
    builder = IndexedDatasetBuilder(
        get_bin_path(output_prefix_str),
        dtype=first.index.dtype,
        multimodal=multimodal,
    )
    del first

    for i, prefix in enumerate(prefixes, start=1):
        # add_index appends both .idx metadata and .bin payload.
        builder.add_index(prefix)
        if i % 50 == 0 or i == len(prefixes):
            log.info("Merged %d/%d", i, len(prefixes))

    # finalize writes the merged .idx footer/arrays.
    builder.finalize(get_idx_path(output_prefix_str))

    return {
        "megatron_source": "python-env",
        "input_dirs": [str(Path(d).expanduser()) for d in input_dirs],
        "input_prefixes": len(prefixes),
        "ignored_missing_idx": len(missing_idx),
        "output_prefix": output_prefix_str,
        "output_bin": str(out_bin),
        "output_idx": str(out_idx),
        "multimodal": bool(multimodal),
    }
