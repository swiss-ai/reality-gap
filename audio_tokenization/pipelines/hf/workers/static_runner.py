"""Static shard runner with a single DataLoader per worker."""

from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset

from audio_tokenization.pipelines.base import WorkerStats
from .batching import process_one_batch
from .shard_io import open_shard_writer, finalize_shard_writer


class ShardIndexDataset(Dataset):
    """Dataset wrapper that returns (shard_id, sample) for a list of indices."""

    def __init__(self, dataset, index_pairs: List[tuple]):
        self.dataset = dataset
        self.index_pairs = index_pairs

    def __len__(self) -> int:
        return len(self.index_pairs)

    def __getitem__(self, idx: int):
        shard_id, sample_idx = self.index_pairs[idx]
        return shard_id, self.dataset[sample_idx]


def run_shards_static(worker, shard_ids: List[int], dataset, num_shards: int, progress_actor=None):
    total_stats = WorkerStats()
    shard_stats_list = []

    shard_ids = sorted(shard_ids)
    if not shard_ids:
        final_stats = total_stats.finalize()
        final_stats["worker_id"] = worker.worker_id
        final_stats["shard_stats"] = shard_stats_list
        return final_stats

    total_len = len(dataset)
    div = total_len // num_shards
    mod = total_len % num_shards
    shard_ranges = []
    start = 0
    for i in range(num_shards):
        shard_len = div + (1 if i < mod else 0)
        end = start + shard_len
        shard_ranges.append((start, end))
        start = end

    index_pairs: List[tuple] = []
    for shard_id in shard_ids:
        start, end = shard_ranges[shard_id]
        if start >= end:
            continue
        for idx in range(start, end):
            index_pairs.append((shard_id, idx))

    if not index_pairs:
        final_stats = total_stats.finalize()
        final_stats["worker_id"] = worker.worker_id
        final_stats["shard_stats"] = shard_stats_list
        return final_stats

    shard_dataset = ShardIndexDataset(dataset, index_pairs)

    loader_kwargs = dict(
        batch_size=worker.batch_size,
        num_workers=worker.dataloader_workers,
        collate_fn=lambda x: x,
        drop_last=False,
    )
    if worker.dataloader_workers and worker.dataloader_workers > 0:
        loader_kwargs["persistent_workers"] = worker.dataloader_persistent_workers
        loader_kwargs["prefetch_factor"] = worker.dataloader_prefetch_factor

    loader = DataLoader(shard_dataset, **loader_kwargs)

    current_shard_id = None
    current_builder = None
    current_tmp_bin = None
    current_tmp_idx = None
    current_bin = None
    current_idx = None
    current_stats = None
    current_batch: List[Dict] = []

    def _start_shard(sid: int):
        return (*open_shard_writer(worker.output_dir, worker.worker_id, sid, num_shards, worker.vocab_size), WorkerStats())

    def _finalize_shard(sid: int):
        nonlocal current_builder, current_tmp_bin, current_tmp_idx, current_bin, current_idx, current_stats, current_batch
        if current_builder is None or current_stats is None:
            return

        if current_batch:
            if worker.batch_size > 1 and worker.mode == "audio_only":
                process_one_batch(worker, current_batch, current_builder, current_stats)
            else:
                for sample in current_batch:
                    prev_samples = current_stats.samples_processed
                    prev_tokens = current_stats.tokens_generated
                    prev_errors = current_stats.errors
                    prev_skipped = current_stats.samples_skipped
                    prev_duration_skipped = current_stats.duration_skipped
                    try:
                        with torch.inference_mode():
                            tokens = worker._process_sample(sample, current_stats)
                        if tokens is not None:
                            tokens_tensor = tokens.cpu() if isinstance(tokens, torch.Tensor) else torch.tensor(tokens, dtype=torch.int64)
                            current_builder.add_item(tokens_tensor)
                            current_builder.end_document()
                            current_stats.samples_processed += 1
                            current_stats.tokens_generated += len(tokens)
                    except Exception as e:
                        worker.logger.warning(f"Error tokenizing sample: {e}")
                        current_stats.errors += 1
                    worker._wandb_accumulate(
                        samples=current_stats.samples_processed - prev_samples,
                        tokens=current_stats.tokens_generated - prev_tokens,
                        errors=current_stats.errors - prev_errors,
                        skipped=current_stats.samples_skipped - prev_skipped,
                        duration_skipped=current_stats.duration_skipped - prev_duration_skipped,
                    )
            current_batch = []

        finalize_shard_writer(current_builder, current_tmp_bin, current_tmp_idx, current_bin, current_idx)
        worker._wandb_flush_if_due(force=True)

        shard_stats = current_stats.finalize()
        shard_stats["shard_id"] = sid
        shard_stats["worker_id"] = worker.worker_id
        shard_stats_list.append(shard_stats)

        total_stats.samples_processed += shard_stats["samples_processed"]
        total_stats.tokens_generated += shard_stats["tokens_generated"]
        total_stats.errors += shard_stats["errors"]
        total_stats.samples_skipped += shard_stats["samples_skipped"]
        total_stats.duration_skipped += shard_stats["duration_skipped"]

        if progress_actor is not None:
            progress_actor.update.remote(shard_stats["samples_processed"])

        worker.logger.info(
            f"Worker {worker.worker_id}: Completed shard {sid}. "
            f"Processed: {total_stats.samples_processed}, Tokens: {total_stats.tokens_generated}"
        )

    for batch in loader:
        for sid, sample in batch:
            if current_shard_id is None:
                current_shard_id = sid
                (current_builder, current_tmp_bin, current_tmp_idx, current_bin, current_idx, current_stats) = _start_shard(sid)
                start, end = shard_ranges[sid]
                worker.logger.info(
                    f"Worker {worker.worker_id}: Processing shard {sid}/{num_shards} "
                    f"({end - start} samples)"
                )

            if sid != current_shard_id:
                _finalize_shard(current_shard_id)
                current_shard_id = sid
                (current_builder, current_tmp_bin, current_tmp_idx, current_bin, current_idx, current_stats) = _start_shard(sid)
                start, end = shard_ranges[sid]
                worker.logger.info(
                    f"Worker {worker.worker_id}: Processing shard {sid}/{num_shards} "
                    f"({end - start} samples)"
                )

            if worker.batch_size > 1 and worker.mode == "audio_only":
                current_batch.append(sample)
                if len(current_batch) >= worker.batch_size:
                    process_one_batch(worker, current_batch, current_builder, current_stats)
                    current_batch = []
            else:
                prev_samples = current_stats.samples_processed
                prev_tokens = current_stats.tokens_generated
                prev_errors = current_stats.errors
                prev_skipped = current_stats.samples_skipped
                prev_duration_skipped = current_stats.duration_skipped
                try:
                    with torch.inference_mode():
                        tokens = worker._process_sample(sample, current_stats)
                    if tokens is not None:
                        tokens_tensor = tokens.cpu() if isinstance(tokens, torch.Tensor) else torch.tensor(tokens, dtype=torch.int64)
                        current_builder.add_item(tokens_tensor)
                        current_builder.end_document()
                        current_stats.samples_processed += 1
                        current_stats.tokens_generated += len(tokens)
                except Exception as e:
                    worker.logger.warning(f"Error tokenizing sample: {e}")
                    current_stats.errors += 1
                worker._wandb_accumulate(
                    samples=current_stats.samples_processed - prev_samples,
                    tokens=current_stats.tokens_generated - prev_tokens,
                    errors=current_stats.errors - prev_errors,
                    skipped=current_stats.samples_skipped - prev_skipped,
                    duration_skipped=current_stats.duration_skipped - prev_duration_skipped,
                )

    if current_shard_id is not None:
        _finalize_shard(current_shard_id)

    worker._wandb_flush_if_due(force=True)

    final_stats = total_stats.finalize()
    final_stats["worker_id"] = worker.worker_id
    final_stats["shard_stats"] = shard_stats_list
    return final_stats
