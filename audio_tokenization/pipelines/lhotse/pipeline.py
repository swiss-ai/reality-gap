#!/usr/bin/env python3
"""Lhotse tokenization pipeline with DDP.

Loads pre-built Shar data and tokenizes on GPU. Data preparation (HF/WDS ->
Shar) is handled by standalone scripts — see ``prepare_hf_to_shar`` and
``prepare_wds_to_shar`` in ``audio_tokenization.utils.prepare_data``.

Architecture overview (3 files):
    data.py        — Shar loading + runtime filters
    checkpoint.py  — WorkerStats, chunk writer, checkpoint save/load, W&B, aggregation
    pipeline.py    — tokenize_loop (main per-rank loop) + run_lhotse_pipeline (entry point)

Launch examples::

    # Single node, 4 GPUs
    torchrun --nproc_per_node=4 -m audio_tokenization.tokenize dataset=peoples_speech_lhotse

    # Multi-node SLURM — Option A: srun spawns 1 task/node, torchrun forks GPU workers
    srun --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 \\
        torchrun --nproc_per_node=4 -m audio_tokenization.tokenize dataset=peoples_speech_lhotse

    # Multi-node SLURM — Option B: srun spawns all ranks directly (no torchrun)
    srun --nodes=2 --ntasks-per-node=4 --gpus-per-node=4 \\
        python -m audio_tokenization.tokenize dataset=peoples_speech_lhotse

    # Resume from checkpoint
    torchrun --nproc_per_node=4 -m audio_tokenization.tokenize dataset=peoples_speech_lhotse resume=true
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict

import torch
import torch.distributed as dist

from .checkpoint import (
    WorkerStats,
    aggregate_stats,
    finalize_shard_writer,
    is_cuda_oom,
    load_checkpoint,
    open_chunk_writer,
    save_checkpoint,
    SimpleWandbLogger,
)
from .data import build_cutset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main tokenization loop (per-rank)
# ---------------------------------------------------------------------------


def tokenize_loop(rank: int, world_size: int, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Main per-rank tokenization loop.

    Steps:
        1. Load prepared Shar CutSet — see ``data.py``
        2. Create ``DynamicBucketingSampler`` with global bucketing
        3. Optionally resume from checkpoint
        4. Wrap in ``UnsupervisedWaveformDataset`` + ``DataLoader`` (map-style,
           sampler in main process) for CPU/GPU overlap
        5. Loop over prefetched batches, tokenize on GPU, write micro-shards
        6. Periodically checkpoint (sampler state + chunk boundary)
    """
    from lhotse.dataset import UnsupervisedWaveformDataset
    from lhotse.dataset.sampling.dynamic_bucketing import DynamicBucketingSampler

    output_dir = cfg["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Clean up stale .tmp files from killed runs (e.g. OOM kill).
    for tmp in Path(output_dir).glob(f"rank_{rank:04d}_*.tmp"):
        logger.warning(f"[rank {rank}] Removing stale temp file: {tmp.name}")
        tmp.unlink()

    # ------------------------------------------------------------------
    # 1. Build CutSet (prepared Shar load + filters/resample safety-net)
    # ------------------------------------------------------------------
    cuts = build_cutset(cfg, rank, world_size)

    # ------------------------------------------------------------------
    # 2. Dynamic bucketing sampler — global bucketing across full dataset.
    #    Lhotse reads only metadata (.jsonl.gz manifests), no audio I/O.
    #    sync_buckets=True keeps DDP ranks in similar duration buckets
    #    to minimise tail-worker idle time.
    # ------------------------------------------------------------------
    max_duration = cfg.get("max_batch_duration", 1500.0)
    max_cuts = cfg.get("max_batch_cuts")
    num_buckets = cfg.get("num_buckets", 20)
    buffer_size = cfg.get("bucket_buffer_size", 20000)
    shuffle = cfg.get("sampler_shuffle", True)
    seed = cfg.get("sampler_seed", 42)
    quadratic_duration = cfg.get("quadratic_duration")

    sampler_kwargs = dict(
        max_duration=max_duration,
        num_buckets=num_buckets,
        buffer_size=buffer_size,
        shuffle=shuffle,
        seed=seed,
        world_size=world_size,
        rank=rank,
        sync_buckets=True,
        drop_last=False,
    )
    if max_cuts is not None:
        sampler_kwargs["max_cuts"] = max_cuts
    if quadratic_duration is not None:
        sampler_kwargs["quadratic_duration"] = quadratic_duration

    sampler = DynamicBucketingSampler(cuts, **sampler_kwargs)

    # ------------------------------------------------------------------
    # 3. Resume from checkpoint — sampler.load_state_dict() restores sampler
    #    state via metadata bookkeeping (no audio decoding), so recovery is
    #    typically fast.
    # ------------------------------------------------------------------
    resume = cfg.get("resume", False)
    start_chunk_id = 0
    cumulative_stats = WorkerStats()

    if resume:
        ckpt = load_checkpoint(output_dir, rank)
        if ckpt is not None:
            sampler.load_state_dict(ckpt["sampler_state"])
            start_chunk_id = ckpt["chunk_id"] + 1
            prev = ckpt.get("stats", {})
            cumulative_stats.samples_processed = prev.get("samples_processed", 0)
            cumulative_stats.tokens_generated = prev.get("tokens_generated", 0)
            cumulative_stats.errors = prev.get("errors", 0)
            cumulative_stats.samples_skipped = prev.get("samples_skipped", 0)
            logger.info(
                f"[rank {rank}] Resumed from chunk {start_chunk_id}, "
                f"samples={cumulative_stats.samples_processed}"
            )

    # ------------------------------------------------------------------
    # 4. DataLoader with prefetching — worker subprocesses decode audio
    #    in parallel while the main thread runs GPU tokenization.
    # ------------------------------------------------------------------
    num_workers = cfg.get("num_workers", 4)
    prefetch_factor = cfg.get("prefetch_factor", 4)
    dataloader_timeout = cfg.get("dataloader_timeout", 300)  # 5 min default

    dataset = UnsupervisedWaveformDataset(collate=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        timeout=dataloader_timeout if num_workers > 0 else 0,
    )

    # ------------------------------------------------------------------
    # 5. Create tokenizer on GPU
    # ------------------------------------------------------------------
    from audio_tokenization.vokenizers import create_tokenizer

    device = f"cuda:{cfg.get('local_rank', 0)}"
    tokenizer_path = cfg["tokenizer_path"]
    mode = cfg.get("mode", "audio_only")
    torch_compile = cfg.get("torch_compile", True)
    target_sr = int(cfg.get("target_sample_rate", 24000))
    trim_last_tokens = cfg.get("trim_last_tokens", 5)

    tokenizer = create_tokenizer(
        omni_tokenizer_path=tokenizer_path,
        mode=mode,
        device=device,
        torch_compile=torch_compile,
        trim_last_tokens=trim_last_tokens,
    )

    # Vocab size determines the integer dtype for micro-shard .bin files.
    from transformers import AutoTokenizer

    omni_tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=True)
    vocab_size = len(omni_tok)
    del omni_tok

    silence_threshold = cfg.get("silence_unique_threshold")

    # ------------------------------------------------------------------
    # 6. W&B logger (rank 0 only)
    # ------------------------------------------------------------------
    wandb_logger = None
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("enabled", False) and rank == 0:
        wandb_logger = SimpleWandbLogger(
            project=wandb_cfg.get("project", "audio-tokenization"),
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("name"),
            tags=wandb_cfg.get("tags", []),
            config={
                "rank": rank,
                "world_size": world_size,
                "max_batch_duration": max_duration,
                "num_buckets": num_buckets,
                "buffer_size": buffer_size,
                "target_sample_rate": target_sr,
                **{k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))},
            },
            log_interval_seconds=wandb_cfg.get("log_interval_seconds", 10.0),
        )

    # ------------------------------------------------------------------
    # 7. Main loop — tokenize batches, write micro-shards, checkpoint
    # ------------------------------------------------------------------
    checkpoint_interval = cfg.get("checkpoint_interval_batches", 500)
    chunk_id = start_chunk_id
    batch_count = 0

    builder, tmp_bin, tmp_idx, bin_path, idx_path = open_chunk_writer(
        output_dir, rank, chunk_id, vocab_size,
    )
    chunk_samples = 0
    stats = cumulative_stats
    total_audio_seconds = 0.0

    logger.info(
        f"[rank {rank}] Starting tokenization loop "
        f"(chunk_id={chunk_id}, checkpoint_interval={checkpoint_interval})"
    )

    consecutive_errors = 0
    max_consecutive_errors = cfg.get("max_consecutive_errors", 50)
    _loop_error = None

    try:
        for batch in dataloader:
            try:
                # UnsupervisedWaveformDataset(collate=True) returns:
                #   {"audio": tensor (B, T), "audio_lens": tensor (B,)}
                audios = batch["audio"]           # (B, T) float
                audio_lens = batch["audio_lens"]  # (B,) int — original lengths

                batch_audio_secs = audio_lens.sum().item() / target_sr
                total_audio_seconds += batch_audio_secs

                audios_gpu = audios.to(device, non_blocking=True)

                with torch.inference_mode():
                    token_list = tokenizer.tokenize_batch(
                        audios_gpu,
                        target_sr,
                        orig_audio_samples=audio_lens.tolist(),
                        pad_audio_samples=audios.shape[1],
                    )

                # Write each sample's tokens to the current micro-shard
                for tokens in token_list:
                    if tokens is None:
                        stats.errors += 1
                        continue

                    # Optional silence filter (skip near-constant token sequences)
                    if silence_threshold and tokens.numel() >= 4:
                        audio_tok = tokens[2:-2] - getattr(tokenizer, "audio_token_offset", 0)
                        if audio_tok.numel() > 0 and torch.unique(audio_tok.cpu()).numel() <= silence_threshold:
                            stats.samples_skipped += 1
                            continue

                    t = torch.as_tensor(tokens, dtype=torch.int64).detach().cpu()
                    builder.add_item(t)
                    builder.end_document()
                    stats.samples_processed += 1
                    stats.tokens_generated += len(t)
                    chunk_samples += 1

                consecutive_errors = 0  # reset on batch success

            except Exception as batch_err:
                stats.errors += 1
                consecutive_errors += 1

                # CUDA OOM: free the failed allocation so the next batch can succeed.
                if is_cuda_oom(batch_err):
                    torch.cuda.empty_cache()
                    logger.warning(
                        f"[rank {rank}] CUDA OOM on batch {batch_count}, freed cache "
                        f"({consecutive_errors}/{max_consecutive_errors})"
                    )
                else:
                    logger.warning(
                        f"[rank {rank}] Batch error ({consecutive_errors}/{max_consecutive_errors}): "
                        f"{batch_err}"
                    )

                if consecutive_errors >= max_consecutive_errors:
                    raise RuntimeError(
                        f"[rank {rank}] {max_consecutive_errors} consecutive batch errors, aborting"
                    ) from batch_err
                continue

            batch_count += 1

            # W&B log (rate-limited by interval inside logger)
            if wandb_logger is not None:
                wandb_logger.log(
                    samples=stats.samples_processed,
                    tokens=stats.tokens_generated,
                    errors=stats.errors,
                    skipped=stats.samples_skipped,
                    batch_audio_seconds=total_audio_seconds,
                )

            # Periodic checkpoint: finalize current chunk, save state, open next
            if batch_count % checkpoint_interval == 0 and chunk_samples > 0:
                finalize_shard_writer(builder, tmp_bin, tmp_idx, bin_path, idx_path)
                logger.info(
                    f"[rank {rank}] Finalized chunk {chunk_id} "
                    f"({chunk_samples} samples, {stats.tokens_generated} total tokens)"
                )

                save_checkpoint(
                    output_dir,
                    rank,
                    sampler_state=sampler.state_dict(),
                    chunk_id=chunk_id,
                    stats=stats.to_dict(),
                )

                chunk_id += 1
                builder, tmp_bin, tmp_idx, bin_path, idx_path = open_chunk_writer(
                    output_dir, rank, chunk_id, vocab_size,
                )
                chunk_samples = 0

    except Exception as e:
        logger.error(f"[rank {rank}] Fatal error in tokenization loop: {e}", exc_info=True)
        stats.errors += 1
        _loop_error = e

    # ------------------------------------------------------------------
    # 8. Finalize last chunk (always save progress, even on failure)
    # ------------------------------------------------------------------
    if chunk_samples > 0:
        finalize_shard_writer(builder, tmp_bin, tmp_idx, bin_path, idx_path)
        logger.info(
            f"[rank {rank}] Finalized final chunk {chunk_id} ({chunk_samples} samples)"
        )
    else:
        # Empty chunk — clean up temp files
        for p in (tmp_bin, tmp_idx):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

    save_checkpoint(
        output_dir,
        rank,
        sampler_state=sampler.state_dict(),
        chunk_id=chunk_id,
        stats=stats.to_dict(),
    )

    result = stats.finalize()
    result["rank"] = rank
    result["chunks_written"] = chunk_id - start_chunk_id + (1 if chunk_samples > 0 else 0)

    # ------------------------------------------------------------------
    # 9. Aggregate stats across ranks, rank 0 saves metadata
    # ------------------------------------------------------------------
    global_result = aggregate_stats(result, rank, world_size)

    if rank == 0:
        _save_metadata(output_dir, cfg, global_result, world_size)

    if wandb_logger is not None:
        wandb_logger.log_final(global_result)
        wandb_logger.finish()

    logger.info(
        f"[rank {rank}] Done: {result['samples_processed']} samples, "
        f"{result['tokens_generated']} tokens, {result['errors']} errors, "
        f"{result['elapsed_time']:.1f}s"
    )

    result["output_dir"] = output_dir

    # Re-raise after saving progress so the exit code signals failure.
    # This MUST come after all cleanup (checkpoint, metadata, wandb) so
    # partial work is never silently lost.
    if _loop_error is not None:
        raise RuntimeError(
            f"[rank {rank}] Tokenization loop failed after processing "
            f"{result['samples_processed']} samples"
        ) from _loop_error

    return result


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def _save_metadata(
    output_dir: str,
    cfg: Dict[str, Any],
    global_result: Dict[str, Any],
    world_size: int,
) -> None:
    """Save dataset metadata JSON on rank 0.

    ``global_result`` contains all-reduced stats (summed across all ranks
    via ``aggregate_stats``), not just rank-0's local stats.
    """
    metadata = {
        "dataset_type": "lhotse",
        "dataset_name": cfg.get("dataset_name", ""),
        "shar_dir": cfg.get("shar_dir", ""),
        "source_type": cfg.get("source_type", "shar"),
        "mode": cfg.get("mode", "audio_only"),
        "tokenizer_path": cfg.get("tokenizer_path", ""),
        "target_sample_rate": cfg.get("target_sample_rate"),
        "world_size": world_size,
        "max_batch_duration": cfg.get("max_batch_duration"),
        "num_buckets": cfg.get("num_buckets"),
        "quadratic_duration": cfg.get("quadratic_duration"),
        "min_duration": cfg.get("min_duration"),
        "max_duration": cfg.get("max_duration"),
        # All-reduced global stats (summed across all ranks)
        "global_stats": global_result,
    }
    metadata_path = Path(output_dir) / "dataset_info.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _infer_shar_label(cfg: Dict[str, Any]) -> str:
    """Derive a human-readable label from ``shar_dir``.

    E.g. ``/data/audioset_unbal_train_shar`` → ``audioset_unbal_train_shar``.
    For multiple dirs, joins their names with ``+``.
    Falls back to sanitised ``dataset_name`` if ``shar_dir`` is not set.
    """
    shar_dir = cfg.get("shar_dir")
    if shar_dir:
        dirs = shar_dir if isinstance(shar_dir, (list, tuple)) else [shar_dir]
        return "+".join(Path(d).name for d in dirs)
    return re.sub(r"[^\w.-]+", "-", cfg.get("dataset_name", "unknown")).strip("-")


def _build_output_subdir(cfg: Dict[str, Any]) -> str:
    """Build a dataset-specific subdirectory name to avoid checkpoint collisions.

    Format: ``{shar_label}_lhotse_{mode}[_dur{min}-{max}]``
    """
    shar_label = _infer_shar_label(cfg)
    mode = cfg.get("mode", "audio_only")
    parts = [shar_label, "lhotse", mode]

    min_dur = cfg.get("min_duration")
    max_dur = cfg.get("max_duration")
    if min_dur is not None or max_dur is not None:
        def _fmt(v):
            if v is None:
                return ""
            return str(int(v)) if float(v).is_integer() else str(v).replace(".", "p")
        parts.append(f"dur{_fmt(min_dur) or 'min'}-{_fmt(max_dur) or 'max'}")

    return "_".join(p for p in parts if p)


def run_lhotse_pipeline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Entry point for the Lhotse tokenization pipeline.

    Expects pre-built Shar data (via ``prepare_hf_to_shar`` or
    ``prepare_wds_to_shar``).  Loads the Shar CutSet, tokenizes on GPU,
    and writes micro-shards with DDP checkpointing.
    """
    # torchrun sets RANK/WORLD_SIZE/LOCAL_RANK.
    # srun (without torchrun) sets SLURM_PROCID/SLURM_NTASKS/SLURM_LOCALID.
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))

    # Safety: infer LOCAL_RANK from global rank + GPUs per node if env vars
    # are missing (e.g. bare srun without torchrun on multi-GPU nodes).
    if "LOCAL_RANK" not in os.environ and "SLURM_LOCALID" not in os.environ:
        gpus_per_node = torch.cuda.device_count()
        if gpus_per_node > 0:
            local_rank = rank % gpus_per_node
            logger.warning(
                f"[rank {rank}] LOCAL_RANK not set, inferred {local_rank} "
                f"from rank % {gpus_per_node} GPUs"
            )

    # Export for dist.init_process_group("nccl") which uses env:// by default
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))

    cfg["rank"] = rank
    cfg["world_size"] = world_size
    cfg["local_rank"] = local_rank

    # Namespace tokenization output to avoid checkpoint collisions.
    cfg["output_dir"] = str(Path(cfg["output_dir"]) / _build_output_subdir(cfg))
    torch.cuda.set_device(local_rank)

    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )
        logger.info(
            f"[rank {rank}/{world_size}] process group initialized "
            f"(backend=nccl, local_rank={local_rank})"
        )

    try:
        result = tokenize_loop(rank, world_size, cfg)
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()

    return result
