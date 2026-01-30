"""Ray worker implementation for audio tokenization."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import ray
import torch
from torch.utils.data import DataLoader

from audio_tokenization.pipelines.base import WorkerStats
from .batching import process_one_batch
from .dataset import load_dataset_for_worker
from .shared_runner import run_shards_shared
from .static_runner import run_shards_static
from .shard_io import open_shard_writer, finalize_shard_writer


@ray.remote(num_gpus=1)
class Worker:
    """Ray worker for audio tokenization."""

    VALID_MODES = ["audio_only", "audio2text", "text2audio", "sft"]

    def __init__(
        self,
        tokenizer_path: str,
        output_dir: str,
        worker_id: int,
        mode: str,
        audio_field: str = "audio",
        text_field: str = "text",
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        batch_size: int = 1,
        dataloader_workers: int = 0,
        dataloader_prefetch_factor: int = 2,
        dataloader_persistent_workers: bool = True,
        target_sample_rate: Optional[int] = None,
        target_bucket: Optional[int] = None,
        wandb_logger=None,
        wandb_log_interval_seconds: Optional[int] = None,
    ):
        self.tokenizer_path = tokenizer_path
        self.output_dir = Path(output_dir)
        self.worker_id = worker_id
        self.mode = mode
        self.audio_field = audio_field
        self.text_field = text_field
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.batch_size = batch_size
        self.dataloader_workers = dataloader_workers
        self.dataloader_prefetch_factor = dataloader_prefetch_factor
        self.dataloader_persistent_workers = dataloader_persistent_workers
        self.target_sample_rate = target_sample_rate
        self.target_bucket = target_bucket
        self.wandb_logger = wandb_logger
        self._wandb_flush_interval = (
            max(1.0, float(wandb_log_interval_seconds))
            if wandb_log_interval_seconds
            else 1.0
        )
        self._wandb_last_flush = time.time()
        self._wandb_pending = {
            "samples": 0,
            "tokens": 0,
            "errors": 0,
            "skipped": 0,
            "duration_skipped": 0,
        }

        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {self.VALID_MODES}")

        self.logger = logging.getLogger(f"Worker{worker_id:02d}")
        self.logger.setLevel(logging.INFO)

        self._tokenizer = None
        self._vocab_size = None

        self.logger.info(
            f"Worker {worker_id} initialized in {mode} mode "
            f"(batch_size={batch_size}, dataloader_workers={dataloader_workers}, "
            f"prefetch_factor={dataloader_prefetch_factor}, target_sample_rate={target_sample_rate})"
        )

    def _wandb_accumulate(
        self,
        samples: int = 0,
        tokens: int = 0,
        errors: int = 0,
        skipped: int = 0,
        duration_skipped: int = 0,
    ) -> None:
        if self.wandb_logger is None:
            return
        self._wandb_pending["samples"] += samples
        self._wandb_pending["tokens"] += tokens
        self._wandb_pending["errors"] += errors
        self._wandb_pending["skipped"] += skipped
        self._wandb_pending["duration_skipped"] += duration_skipped
        self._wandb_flush_if_due()

    def _wandb_flush_if_due(self, force: bool = False) -> None:
        if self.wandb_logger is None:
            return
        now = time.time()
        if not force and (now - self._wandb_last_flush) < self._wandb_flush_interval:
            return
        if all(v == 0 for v in self._wandb_pending.values()):
            self._wandb_last_flush = now
            return

        self.wandb_logger.update.remote(
            samples=self._wandb_pending["samples"],
            tokens=self._wandb_pending["tokens"],
            errors=self._wandb_pending["errors"],
            skipped=self._wandb_pending["skipped"],
            duration_skipped=self._wandb_pending["duration_skipped"],
        )
        for key in self._wandb_pending:
            self._wandb_pending[key] = 0
        self._wandb_last_flush = now

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from audio_tokenization.vokenizers import create_tokenizer
            self._tokenizer = create_tokenizer(
                omni_tokenizer_path=self.tokenizer_path,
                device="cuda",
            )
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        if self._vocab_size is None:
            from transformers import AutoTokenizer
            omni_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            self._vocab_size = len(omni_tokenizer)
        return self._vocab_size

    def _get_audio_duration(self, audio, sample_rate: int) -> float:
        if isinstance(audio, torch.Tensor):
            return audio.shape[-1] / sample_rate
        if hasattr(audio, "__len__"):
            return len(audio) / sample_rate
        return 0.0

    def _process_sample(self, sample: Dict, stats: WorkerStats) -> Optional[List[int]]:
        if self.mode == "audio_only":
            return self._process_audio_only(sample, stats)
        if self.mode == "audio2text":
            return self._process_audio2text(sample, stats)
        if self.mode == "text2audio":
            return self._process_text2audio(sample, stats)
        if self.mode == "sft":
            return self._process_sft(sample, stats)
        raise ValueError(f"Unknown mode: {self.mode}")

    def _process_audio_only(self, sample: Dict, stats: WorkerStats) -> Optional[List[int]]:
        audio_data = sample.get(self.audio_field)
        if audio_data is None:
            stats.samples_skipped += 1
            return None

        if isinstance(audio_data, dict):
            audio = audio_data.get("array")
            sample_rate = audio_data.get("sampling_rate", 16000)
        else:
            audio = audio_data
            sample_rate = sample.get("sample_rate", 16000)

        if audio is None:
            stats.samples_skipped += 1
            return None

        duration = self._get_audio_duration(audio, sample_rate)
        if self.min_duration and duration < self.min_duration:
            stats.duration_skipped += 1
            return None
        if self.max_duration and duration > self.max_duration:
            stats.duration_skipped += 1
            return None

        return self.tokenizer.tokenize(audio, sample_rate)

    def _process_audio2text(self, sample: Dict, stats: WorkerStats) -> Optional[List[int]]:
        raise NotImplementedError("audio2text mode not yet implemented")

    def _process_text2audio(self, sample: Dict, stats: WorkerStats) -> Optional[List[int]]:
        raise NotImplementedError("text2audio mode not yet implemented")

    def _process_sft(self, sample: Dict, stats: WorkerStats) -> Optional[List[int]]:
        raise NotImplementedError("sft mode not yet implemented")

    def _extract_audio(self, sample: Dict) -> tuple:
        audio_data = sample.get(self.audio_field)
        if audio_data is None:
            return None, None

        if isinstance(audio_data, dict):
            audio = audio_data.get("array")
            sample_rate = audio_data.get("sampling_rate", 16000)
        else:
            audio = audio_data
            sample_rate = sample.get("sample_rate", 16000)

        return audio, sample_rate

    def _check_duration(self, audio, sample_rate: int) -> bool:
        duration = self._get_audio_duration(audio, sample_rate)
        if self.min_duration and duration < self.min_duration:
            return False
        if self.max_duration and duration > self.max_duration:
            return False
        return True

    def process_shard(self, shard_id: int, shard_data, total_shards: int) -> Dict[str, Any]:
        stats = WorkerStats()
        builder, tmp_bin, tmp_idx, bin_path, idx_path = open_shard_writer(
            self.output_dir, self.worker_id, shard_id, total_shards, self.vocab_size
        )

        if self.batch_size > 1 and self.mode == "audio_only":
            self._process_shard_batched(shard_data, builder, stats)
        else:
            for sample in shard_data:
                prev_samples = stats.samples_processed
                prev_tokens = stats.tokens_generated
                prev_errors = stats.errors
                prev_skipped = stats.samples_skipped
                prev_duration_skipped = stats.duration_skipped
                try:
                    with torch.inference_mode():
                        tokens = self._process_sample(sample, stats)

                    if tokens is not None:
                        tokens_tensor = tokens.cpu() if isinstance(tokens, torch.Tensor) else torch.tensor(tokens, dtype=torch.int64)
                        builder.add_item(tokens_tensor)
                        builder.end_document()
                        stats.samples_processed += 1
                        stats.tokens_generated += len(tokens)

                except Exception as e:
                    self.logger.warning(f"Error tokenizing sample: {e}")
                    stats.errors += 1

                self._wandb_accumulate(
                    samples=stats.samples_processed - prev_samples,
                    tokens=stats.tokens_generated - prev_tokens,
                    errors=stats.errors - prev_errors,
                    skipped=stats.samples_skipped - prev_skipped,
                    duration_skipped=stats.duration_skipped - prev_duration_skipped,
                )

        finalize_shard_writer(builder, tmp_bin, tmp_idx, bin_path, idx_path)
        self._wandb_flush_if_due(force=True)
        return stats.finalize()

    def _process_shard_batched(self, shard_data, builder, stats: WorkerStats) -> None:
        def iter_batches():
            if self.dataloader_workers and self.dataloader_workers > 0:
                loader = DataLoader(
                    shard_data,
                    batch_size=self.batch_size,
                    num_workers=self.dataloader_workers,
                    collate_fn=lambda x: x,
                    drop_last=False,
                    persistent_workers=self.dataloader_persistent_workers,
                    prefetch_factor=self.dataloader_prefetch_factor,
                )
                for batch in loader:
                    yield batch
            else:
                batch = []
                for sample in shard_data:
                    batch.append(sample)
                    if len(batch) >= self.batch_size:
                        yield batch
                        batch = []
                if batch:
                    yield batch

        for batch in iter_batches():
            process_one_batch(self, batch, builder, stats)

    def run_shards(
        self,
        shard_source,
        dataset_info: Dict[str, Any],
        num_shards: int,
        progress_actor=None,
    ) -> Dict[str, Any]:
        dataset = load_dataset_for_worker(
            dataset_info,
            self.audio_field,
            self.target_sample_rate,
            logger=self.logger,
        )

        if isinstance(shard_source, list):
            return run_shards_static(self, shard_source, dataset, num_shards, progress_actor)

        return run_shards_shared(self, shard_source, dataset, num_shards, progress_actor)
