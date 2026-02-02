"""Shared-queue shard runner."""

import ray

from audio_tokenization.pipelines.base import WorkerStats


def run_shards_shared(worker, work_queue, dataset, num_shards: int, progress_actor=None):
    total_stats = WorkerStats()
    shard_stats_list = []

    while True:
        shard_id = ray.get(work_queue.get_next_shard.remote())
        if shard_id is None:
            break

        try:
            shard_data = dataset.shard(
                num_shards=num_shards,
                index=shard_id,
                contiguous=True,
            )

            worker.logger.info(
                f"Worker {worker.worker_id}: Processing shard {shard_id}/{num_shards} "
                f"({len(shard_data)} samples)"
            )

            shard_stats = worker.process_shard(shard_id, shard_data, num_shards)
            shard_stats["shard_id"] = shard_id
            shard_stats["worker_id"] = worker.worker_id
            shard_stats_list.append(shard_stats)

            total_stats.samples_processed += shard_stats["samples_processed"]
            total_stats.tokens_generated += shard_stats["tokens_generated"]
            total_stats.errors += shard_stats["errors"]
            total_stats.samples_skipped += shard_stats["samples_skipped"]
            total_stats.duration_skipped += shard_stats["duration_skipped"]
            total_stats.frequency_skipped += shard_stats.get("frequency_skipped", 0)

            if progress_actor is not None:
                progress_actor.update.remote(shard_stats["samples_processed"])

            ray.get(work_queue.mark_completed.remote(shard_id))

            worker.logger.info(
                f"Worker {worker.worker_id}: Completed shard {shard_id}. "
                f"Processed: {total_stats.samples_processed}, Tokens: {total_stats.tokens_generated}"
            )

        except Exception as e:
            worker.logger.error(f"Worker {worker.worker_id}: Error processing shard {shard_id}: {e}")
            ray.get(work_queue.mark_failed.remote(shard_id))
            total_stats.errors += 1

    worker._wandb_flush_if_due(force=True)

    final_stats = total_stats.finalize()
    final_stats["worker_id"] = worker.worker_id
    final_stats["shard_stats"] = shard_stats_list
    return final_stats
