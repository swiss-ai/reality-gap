"""Batch tokenization helpers."""

from typing import Dict, List

import torch

from audio_tokenization.pipelines.base import WorkerStats


def process_one_batch(worker, batch: List[Dict], builder, stats: WorkerStats) -> None:
    """Process a single batch of samples."""
    prev_samples = stats.samples_processed
    prev_tokens = stats.tokens_generated
    prev_errors = stats.errors
    prev_skipped = stats.samples_skipped
    prev_duration_skipped = stats.duration_skipped

    audios = []
    sample_rates = []

    for sample in batch:
        audio, sample_rate = worker._extract_audio(sample)

        if audio is None:
            stats.samples_skipped += 1
            continue

        if not worker._check_duration(audio, sample_rate):
            stats.duration_skipped += 1
            continue

        audios.append(audio)
        sample_rates.append(sample_rate)

    if not audios:
        worker._wandb_accumulate(
            samples=stats.samples_processed - prev_samples,
            tokens=stats.tokens_generated - prev_tokens,
            errors=stats.errors - prev_errors,
            skipped=stats.samples_skipped - prev_skipped,
            duration_skipped=stats.duration_skipped - prev_duration_skipped,
        )
        return

    if worker.target_bucket:
        audios = [a[:worker.target_bucket] for a in audios]

    if len(set(sample_rates)) > 1:
        worker.logger.warning(
            f"Mixed sample rates in batch (worker {worker.worker_id}); "
            "falling back to per-sample tokenization."
        )
        for audio, sample_rate in zip(audios, sample_rates):
            try:
                with torch.inference_mode():
                    tokens = worker.tokenizer.tokenize(audio, sample_rate)
                tokens_tensor = tokens.cpu() if isinstance(tokens, torch.Tensor) else torch.tensor(tokens, dtype=torch.int64)
                builder.add_item(tokens_tensor)
                builder.end_document()
                stats.samples_processed += 1
                stats.tokens_generated += len(tokens)
            except Exception as e:
                worker.logger.warning(f"Error tokenizing sample: {e}")
                stats.errors += 1
        worker._wandb_accumulate(
            samples=stats.samples_processed - prev_samples,
            tokens=stats.tokens_generated - prev_tokens,
            errors=stats.errors - prev_errors,
            skipped=stats.samples_skipped - prev_skipped,
            duration_skipped=stats.duration_skipped - prev_duration_skipped,
        )
        return

    batch_sample_rate = sample_rates[0]
    try:
        with torch.inference_mode():
            tokens_list = worker.tokenizer.tokenize_batch(audios, batch_sample_rate)
        for tokens in tokens_list:
            tokens_tensor = tokens.cpu() if isinstance(tokens, torch.Tensor) else torch.tensor(tokens, dtype=torch.int64)
            builder.add_item(tokens_tensor)
            builder.end_document()
            stats.samples_processed += 1
            stats.tokens_generated += len(tokens)
    except Exception as e:
        worker.logger.warning(f"Error tokenizing batch of {len(audios)}: {e}")
        stats.errors += len(audios)

    worker._wandb_accumulate(
        samples=stats.samples_processed - prev_samples,
        tokens=stats.tokens_generated - prev_tokens,
        errors=stats.errors - prev_errors,
        skipped=stats.samples_skipped - prev_skipped,
        duration_skipped=stats.duration_skipped - prev_duration_skipped,
    )
