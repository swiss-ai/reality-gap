"""Checkpointing, micro-shard I/O, stats tracking, and W&B logging.

Design decisions:
- **Micro-shard chunking**: Each rank writes independent chunks named
  ``rank_XXXX_chunk_YYYY.{bin,idx}``.  Chunks are written to ``.tmp``
  files first and atomically renamed on finalize — no partial files on crash.
- **Sampler-state checkpointing**: Lhotse's ``DynamicBucketingSampler``
  supports ``state_dict()`` / ``load_state_dict()``. On resume the sampler
  restores progress through metadata bookkeeping (no audio decoding), so
  recovery is typically fast.
- **WorkerStats** is an inline dataclass (no Ray dependency from base.py).
- **SimpleWandbLogger**: Plain Python class (rank 0 only), rate-limited
  by a configurable interval.  No Ray actor overhead.
- **_aggregate_stats**: Uses ``dist.all_reduce`` so rank 0 can report
  global totals in the final metadata file.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist

from audio_tokenization.pipelines.shard_io import finalize_shard_writer
from audio_tokenization.utils.indexed_dataset import DType, IndexedDatasetBuilder

logger = logging.getLogger(__name__)

# Re-export finalize_shard_writer so pipeline.py only needs to import from here.
__all__ = [
    "WorkerStats",
    "open_chunk_writer",
    "save_checkpoint",
    "load_checkpoint",
    "SimpleWandbLogger",
    "aggregate_stats",
    "finalize_shard_writer",
    "is_cuda_oom",
]


# ---------------------------------------------------------------------------
# CUDA OOM detection
# ---------------------------------------------------------------------------


def is_cuda_oom(exc: BaseException) -> bool:
    """Return True if *exc* indicates a CUDA out-of-memory error.

    Checks both the dedicated ``torch.cuda.OutOfMemoryError`` (PyTorch ≥ 2.0)
    and the older ``RuntimeError("CUDA out of memory")`` pattern.
    """
    cuda_oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if cuda_oom_type is not None and isinstance(exc, cuda_oom_type):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return "cuda out of memory" in msg or "out of memory" in msg
    return False


# ---------------------------------------------------------------------------
# Per-rank statistics
# ---------------------------------------------------------------------------


@dataclass
class WorkerStats:
    """Cumulative statistics tracked per rank (no Ray dependency)."""

    samples_processed: int = 0
    tokens_generated: int = 0
    errors: int = 0
    samples_skipped: int = 0
    duration_skipped: int = 0
    frequency_skipped: int = 0
    start_time: float = field(default_factory=time.time)
    elapsed_time: float = 0.0
    throughput: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "samples_processed": self.samples_processed,
            "tokens_generated": self.tokens_generated,
            "errors": self.errors,
            "samples_skipped": self.samples_skipped,
            "duration_skipped": self.duration_skipped,
            "frequency_skipped": self.frequency_skipped,
            "elapsed_time": self.elapsed_time,
            "throughput": self.throughput,
        }

    def finalize(self) -> Dict[str, Any]:
        """Compute elapsed time and throughput, return final stats dict."""
        self.elapsed_time = time.time() - self.start_time
        self.throughput = (
            self.tokens_generated / self.elapsed_time if self.elapsed_time > 0 else 0
        )
        return self.to_dict()


# ---------------------------------------------------------------------------
# Micro-shard chunk writer
# ---------------------------------------------------------------------------


def open_chunk_writer(
    output_dir: str,
    rank: int,
    chunk_id: int,
    vocab_size: int,
) -> Tuple[IndexedDatasetBuilder, str, str, str, str]:
    """Open a Megatron IndexedDatasetBuilder for a micro-shard chunk.

    Naming: ``rank_XXXX_chunk_YYYY.{bin,idx}``
    Writes to ``.tmp`` suffix; call ``finalize_shard_writer()`` to atomically
    rename to the final paths.

    Returns:
        (builder, tmp_bin_path, tmp_idx_path, final_bin_path, final_idx_path)
    """
    output_prefix = Path(output_dir) / f"rank_{rank:04d}_chunk_{chunk_id:04d}"
    bin_path = str(output_prefix) + ".bin"
    idx_path = str(output_prefix) + ".idx"
    tmp_bin_path = bin_path + ".tmp"
    tmp_idx_path = idx_path + ".tmp"
    dtype = DType.optimal_dtype(vocab_size)
    builder = IndexedDatasetBuilder(tmp_bin_path, dtype=dtype)
    return builder, tmp_bin_path, tmp_idx_path, bin_path, idx_path


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------


def _checkpoint_path(output_dir: str, rank: int) -> Path:
    return Path(output_dir) / f"rank_{rank:04d}_checkpoint.pt"


def save_checkpoint(
    output_dir: str,
    rank: int,
    sampler_state: Dict[str, Any],
    chunk_id: int,
    stats: Dict[str, Any],
) -> None:
    """Atomically save checkpoint via ``.tmp`` + ``os.replace()``."""
    ckpt_path = _checkpoint_path(output_dir, rank)
    tmp_path = str(ckpt_path) + ".tmp"
    payload = {
        "sampler_state": sampler_state,
        "chunk_id": chunk_id,
        "stats": stats,
    }
    torch.save(payload, tmp_path)
    os.replace(tmp_path, str(ckpt_path))
    logger.debug(f"[rank {rank}] Saved checkpoint chunk_id={chunk_id}")


def load_checkpoint(output_dir: str, rank: int) -> Optional[Dict[str, Any]]:
    """Load checkpoint if it exists, else return None."""
    ckpt_path = _checkpoint_path(output_dir, rank)
    if not ckpt_path.exists():
        return None
    logger.info(f"[rank {rank}] Loading checkpoint from {ckpt_path}")
    return torch.load(str(ckpt_path), map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# W&B logger (rank 0 only)
# ---------------------------------------------------------------------------


class SimpleWandbLogger:
    """Lightweight W&B logger for rank 0.

    Logs running totals + throughput at a configurable interval.
    Calls are rate-limited: ``log()`` is a no-op unless ``log_interval_seconds``
    has elapsed since the last flush (or ``force=True``).
    """

    def __init__(
        self,
        project: str = "audio-tokenization",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[list] = None,
        config: Optional[dict] = None,
        log_interval_seconds: float = 10.0,
    ):
        import wandb

        self._run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            tags=tags or [],
            config=config or {},
            resume="allow",
        )
        self._interval = max(1.0, log_interval_seconds)
        self._last_flush = time.time()
        self._start_time = time.time()
        self._step = 0

    def log(
        self,
        samples: int,
        tokens: int,
        errors: int,
        skipped: int,
        batch_audio_seconds: float = 0.0,
        force: bool = False,
    ) -> None:
        """Log absolute totals if the flush interval has elapsed."""
        now = time.time()
        if not force and now - self._last_flush < self._interval:
            return
        import wandb

        elapsed = now - self._start_time
        wandb.log(
            {
                "samples_processed": samples,
                "tokens_generated": tokens,
                "errors": errors,
                "samples_skipped": skipped,
                "samples_per_second": samples / elapsed if elapsed > 0 else 0,
                "tokens_per_second": tokens / elapsed if elapsed > 0 else 0,
                "elapsed_seconds": elapsed,
                "batch_audio_seconds": batch_audio_seconds,
            },
            step=self._step,
        )
        self._step += 1
        self._last_flush = now

    def log_final(self, metrics: Dict[str, Any]) -> None:
        import wandb

        wandb.log({f"final/{k}": v for k, v in metrics.items()})

    def finish(self) -> None:
        import wandb

        wandb.finish()


# ---------------------------------------------------------------------------
# Cross-rank stats aggregation
# ---------------------------------------------------------------------------


def aggregate_stats(
    result: Dict[str, Any],
    rank: int,
    world_size: int,
) -> Dict[str, Any]:
    """Aggregate per-rank stats into global totals via ``dist.all_reduce``.

    Sums counters (samples, tokens, errors, skipped) and takes the max
    elapsed time (wall time = slowest rank).  Returns a new dict with
    global totals suitable for the metadata file.
    """
    if world_size <= 1 or not dist.is_initialized():
        return result

    keys_to_sum = ["samples_processed", "tokens_generated", "errors", "samples_skipped"]
    local_vals = torch.tensor(
        [float(result.get(k, 0)) for k in keys_to_sum],
        dtype=torch.float64,
        device="cuda",
    )
    dist.all_reduce(local_vals, op=dist.ReduceOp.SUM)
    global_vals = local_vals.cpu().tolist()

    global_result = dict(result)
    for k, v in zip(keys_to_sum, global_vals):
        global_result[k] = int(v)

    # Wall time = slowest rank
    elapsed_t = torch.tensor(
        [result.get("elapsed_time", 0.0)], dtype=torch.float64, device="cuda"
    )
    dist.all_reduce(elapsed_t, op=dist.ReduceOp.MAX)
    global_result["elapsed_time"] = elapsed_t.item()

    elapsed = global_result["elapsed_time"]
    global_result["throughput"] = (
        global_result["tokens_generated"] / elapsed if elapsed > 0 else 0
    )
    global_result["world_size"] = world_size

    logger.info(
        f"[rank {rank}] Global stats: {global_result['samples_processed']} samples, "
        f"{global_result['tokens_generated']} tokens, {global_result['errors']} errors"
    )

    return global_result
