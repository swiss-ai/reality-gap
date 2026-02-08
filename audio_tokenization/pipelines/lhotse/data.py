"""Lhotse data loading: load prepared Shar for tokenization.

Data preparation (HF/WDS -> Shar) is handled by standalone scripts:
    - audio_tokenization.utils.prepare_data.prepare_hf_to_shar
    - audio_tokenization.utils.prepare_data.prepare_wds_to_shar

This module only loads pre-built Shar and applies runtime filters.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

SHAR_INDEX_FILENAME = "shar_index.json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_cutset(cfg: Dict[str, Any], rank: int, world_size: int):
    """Load prepared Shar into a CutSet and apply post-load filters."""
    _set_resampling_backend(rank)

    cuts = _load_shar_cutset(cfg, rank)

    # Drop low sample-rate audio before resampling (e.g., 8kHz -> 24kHz = garbage).
    min_sr = cfg.get("min_sample_rate")
    if min_sr is not None:
        min_sr = int(min_sr)
        cuts = cuts.filter(
            lambda cut: getattr(cut, "sampling_rate", None) is not None
            and cut.sampling_rate >= min_sr
        )

    # Lazy safety-net resample (no-op when SR already matches).
    target_sr = cfg.get("target_sample_rate")
    if target_sr:
        cuts = cuts.resample(int(target_sr))

    min_dur = cfg.get("min_duration")
    max_dur = cfg.get("max_duration")
    if min_dur is not None or max_dur is not None:
        def _dur_filter(cut) -> bool:
            d = cut.duration
            if min_dur is not None and d < min_dur:
                return False
            if max_dur is not None and d > max_dur:
                return False
            return True

        cuts = cuts.filter(_dur_filter)

    return cuts


# ---------------------------------------------------------------------------
# Shar loading
# ---------------------------------------------------------------------------


def _load_shar_cutset(cfg, rank):
    """Load a CutSet from prepared Shar manifests."""
    from lhotse import CutSet

    shar_dir = cfg.get("shar_dir")
    if not shar_dir:
        raise ValueError("Lhotse tokenization requires 'shar_dir' with prepared Shar data.")

    shar_path = Path(shar_dir)
    if not shar_path.is_dir():
        raise FileNotFoundError(f"Shar directory does not exist: {shar_dir}")

    index_name = cfg.get("shar_index_filename", SHAR_INDEX_FILENAME)
    index_path = shar_path / index_name
    if index_path.is_file():
        with open(index_path) as f:
            fields = json.load(f).get("fields", {})
        if "cuts" not in fields:
            raise ValueError(f"Shar index missing required 'cuts' field: {index_path}")
        logger.info(f"[rank {rank}] Loading merged Shar index from {index_path}")
        return CutSet.from_shar(fields=fields, split_for_dataloading=False, shuffle_shards=True)
    elif _shar_exists(shar_dir):
        logger.info(f"[rank {rank}] Loading top-level Shar from {shar_dir}")
        return CutSet.from_shar(in_dir=shar_dir, split_for_dataloading=False, shuffle_shards=True)
    else:
        raise FileNotFoundError(
            f"No Shar manifests found in {shar_dir}. "
            "Run prepare_hf_to_shar or prepare_wds_to_shar first."
        )


def _shar_exists(shar_dir: str) -> bool:
    p = Path(shar_dir)
    if not p.is_dir():
        return False
    return any(p.glob("cuts*.jsonl.gz"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_resampling_backend(rank: int) -> None:
    try:
        from lhotse.audio.resampling_backend import set_current_resampling_backend

        set_current_resampling_backend("sox")
        logger.info(f"[rank {rank}] Using SoX resampling backend")
    except Exception:
        logger.warning(f"[rank {rank}] SoX backend unavailable, using default sinc interpolation")
