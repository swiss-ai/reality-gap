#!/usr/bin/env python3
"""Ray workers for distributed audio tokenization."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import ray
import torch
import numpy as np

from audio_tokenization.pipelines.base import WorkerStats
from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder, DType

logger = logging.getLogger(__name__)


@ray.remote
class ShardQueue:
    """Ray actor for work queue distribution."""

    def __init__(self, total_shards: int, initial_shards: List[int] = None):
        """
        Initialize shard queue.

        Args:
            total_shards: Total number of shards
            initial_shards: Optional list of specific shards to process (for resume)
        """
        self.total_shards = total_shards
        if initial_shards is not None:
            self.pending = list(initial_shards)
        else:
            self.pending = list(range(total_shards))
        self.completed = []
        self.failed = []

    def get_next_shard(self) -> Optional[int]:
        """Get next shard to process."""
        if self.pending:
            return self.pending.pop(0)
        return None

    def mark_completed(self, shard_id: int):
        """Mark a shard as completed."""
        self.completed.append(shard_id)

    def mark_failed(self, shard_id: int):
        """Mark a shard as failed."""
        self.failed.append(shard_id)

    def get_status(self) -> Dict[str, Any]:
        """Get queue status."""
        return {
            "total": self.total_shards,
            "pending": len(self.pending),
            "completed": len(self.completed),
            "failed": len(self.failed),
        }


@ray.remote(num_gpus=1)
class Worker:
    """Ray worker for audio tokenization."""

    # Valid modes for audio tokenization
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

        # Validate mode
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {self.VALID_MODES}")

        # Setup logging
        self.logger = logging.getLogger(f"Worker{worker_id:02d}")
        self.logger.setLevel(logging.INFO)

        # Lazy loading
        self._tokenizer = None
        self._vocab_size = None

        self.logger.info(f"Worker {worker_id} initialized in {mode} mode (batch_size={batch_size})")

    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            from audio_tokenization.vokenizers import create_tokenizer
            self._tokenizer = create_tokenizer(
                omni_tokenizer_path=self.tokenizer_path,
                device="cuda",
            )
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        """Auto-detect vocab size from omni_tokenizer."""
        if self._vocab_size is None:
            from transformers import AutoTokenizer
            omni_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            self._vocab_size = len(omni_tokenizer)
        return self._vocab_size

    def _get_audio_duration(self, audio, sample_rate: int) -> float:
        """Calculate audio duration in seconds."""
        if isinstance(audio, torch.Tensor):
            return audio.shape[-1] / sample_rate
        elif hasattr(audio, "__len__"):
            return len(audio) / sample_rate
        return 0.0

    def _process_sample(self, sample: Dict, stats: WorkerStats) -> Optional[List[int]]:
        """Process a single sample based on mode."""
        if self.mode == "audio_only":
            return self._process_audio_only(sample, stats)
        elif self.mode == "audio2text":
            return self._process_audio2text(sample, stats)
        elif self.mode == "text2audio":
            return self._process_text2audio(sample, stats)
        elif self.mode == "sft":
            return self._process_sft(sample, stats)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _process_audio_only(self, sample: Dict, stats: WorkerStats) -> Optional[List[int]]:
        """Process audio-only mode: just tokenize the audio."""
        audio_data = sample.get(self.audio_field)
        if audio_data is None:
            stats.samples_skipped += 1
            return None

        # Handle HuggingFace audio format
        if isinstance(audio_data, dict):
            audio = audio_data.get("array")
            sample_rate = audio_data.get("sampling_rate", 16000)
        else:
            audio = audio_data
            sample_rate = sample.get("sample_rate", 16000)

        if audio is None:
            stats.samples_skipped += 1
            return None

        # Check duration
        duration = self._get_audio_duration(audio, sample_rate)
        if self.min_duration and duration < self.min_duration:
            stats.duration_skipped += 1
            return None
        if self.max_duration and duration > self.max_duration:
            stats.duration_skipped += 1
            return None

        # Tokenize
        return self.tokenizer.tokenize(audio, sample_rate)

    def _process_audio2text(self, sample: Dict, stats: WorkerStats) -> Optional[List[int]]:
        """Process audio2text mode: audio input -> text output (e.g., ASR)."""
        # TODO: Implement for future extension
        raise NotImplementedError("audio2text mode not yet implemented")

    def _process_text2audio(self, sample: Dict, stats: WorkerStats) -> Optional[List[int]]:
        """Process text2audio mode: text input -> audio output (e.g., TTS)."""
        # TODO: Implement for future extension
        raise NotImplementedError("text2audio mode not yet implemented")

    def _process_sft(self, sample: Dict, stats: WorkerStats) -> Optional[List[int]]:
        """Process SFT mode: supervised fine-tuning with text and audio."""
        # TODO: Implement for future extension
        raise NotImplementedError("sft mode not yet implemented")

    def _extract_audio(self, sample: Dict) -> tuple:
        """Extract audio array and sample rate from a sample.

        Returns:
            (audio_array, sample_rate) or (None, None) if invalid
        """
        audio_data = sample.get(self.audio_field)
        if audio_data is None:
            return None, None

        # Handle HuggingFace audio format
        if isinstance(audio_data, dict):
            audio = audio_data.get("array")
            sample_rate = audio_data.get("sampling_rate", 16000)
        else:
            audio = audio_data
            sample_rate = sample.get("sample_rate", 16000)

        return audio, sample_rate

    def _check_duration(self, audio, sample_rate: int) -> bool:
        """Check if audio meets duration criteria."""
        duration = self._get_audio_duration(audio, sample_rate)
        if self.min_duration and duration < self.min_duration:
            return False
        if self.max_duration and duration > self.max_duration:
            return False
        return True

    def process_shard(self, shard_id: int, shard_data, total_shards: int) -> Dict[str, Any]:
        """Process a single shard with optional batching."""
        stats = WorkerStats()
        output_prefix = self.output_dir / f"rank_{self.worker_id}_shard_{shard_id}_{total_shards}"

        # Megatron's builder takes .bin path, select optimal dtype based on vocab_size
        bin_path = str(output_prefix) + ".bin"
        idx_path = str(output_prefix) + ".idx"
        dtype = DType.optimal_dtype(self.vocab_size)
        builder = IndexedDatasetBuilder(bin_path, dtype=dtype)

        if self.batch_size > 1 and self.mode == "audio_only":
            # Batched processing for audio_only mode
            self._process_shard_batched(shard_data, builder, stats)
        else:
            # Single-sample processing (original behavior)
            for sample in shard_data:
                try:
                    tokens = self._process_sample(sample, stats)

                    if tokens is not None:
                        # Convert to torch tensor on CPU for Megatron's add_item
                        if isinstance(tokens, torch.Tensor):
                            tokens_tensor = tokens.cpu()
                        else:
                            tokens_tensor = torch.tensor(tokens, dtype=torch.int64)

                        builder.add_item(tokens_tensor)
                        builder.end_document()  # Each audio sample is its own document
                        stats.samples_processed += 1
                        stats.tokens_generated += len(tokens)

                except Exception as e:
                    self.logger.warning(f"Error tokenizing sample: {e}")
                    stats.errors += 1

        builder.finalize(idx_path)
        return stats.finalize()

    def _process_shard_batched(self, shard_data, builder, stats: WorkerStats):
        """Process shard using batched tokenization for efficiency.

        Groups samples into batches and tokenizes them together on GPU.
        Requires all samples in a batch to have the same length (use bucket filtering).
        """
        batch_audios = []
        batch_sample_rate = None

        def flush_batch():
            """Process accumulated batch."""
            nonlocal batch_audios, batch_sample_rate

            if not batch_audios:
                return

            try:
                # Tokenize entire batch at once
                tokens_list = self.tokenizer.tokenize_batch(batch_audios, batch_sample_rate)

                for tokens in tokens_list:
                    if isinstance(tokens, torch.Tensor):
                        tokens_tensor = tokens.cpu()
                    else:
                        tokens_tensor = torch.tensor(tokens, dtype=torch.int64)

                    builder.add_item(tokens_tensor)
                    builder.end_document()
                    stats.samples_processed += 1
                    stats.tokens_generated += len(tokens)

            except Exception as e:
                self.logger.warning(f"Error tokenizing batch of {len(batch_audios)}: {e}")
                stats.errors += len(batch_audios)

            # Reset batch
            batch_audios = []
            batch_sample_rate = None

        for sample in shard_data:
            audio, sample_rate = self._extract_audio(sample)

            if audio is None:
                stats.samples_skipped += 1
                continue

            if not self._check_duration(audio, sample_rate):
                stats.duration_skipped += 1
                continue

            # Check if sample rate matches current batch
            if batch_sample_rate is not None and sample_rate != batch_sample_rate:
                # Different sample rate, flush current batch first
                flush_batch()

            batch_sample_rate = sample_rate
            batch_audios.append(audio)

            # Flush when batch is full
            if len(batch_audios) >= self.batch_size:
                flush_batch()

        # Flush remaining samples
        flush_batch()

    def run_shards(
        self,
        work_queue: ShardQueue,
        dataset_info: Dict[str, Any],
        num_shards: int,
        progress_actor=None,
    ) -> Dict[str, Any]:
        """Run all assigned shards.

        Args:
            work_queue: Ray actor for shard distribution
            dataset_info: Dict containing dataset loading parameters:
                - name: Dataset name
                - config: Dataset config name
                - split: Dataset split
                - cache_dir: Cache directory
                - max_samples: Optional sample limit (applied via split slicing)
                - filtered_indices: Optional list of indices for bucket filtering
            num_shards: Total number of shards
            progress_actor: Optional Ray actor for progress tracking

        Returns:
            Dict with worker statistics
        """
        from datasets import load_dataset

        # Load dataset (with optional max_samples limit via split slicing)
        split = dataset_info["split"]
        max_samples = dataset_info.get("max_samples")
        if max_samples:
            split = f"{split}[:{max_samples}]"  # e.g., "unbal_train[:400]"

        dataset = load_dataset(
            dataset_info["name"],
            name=dataset_info.get("config"),
            split=split,
            cache_dir=dataset_info.get("cache_dir"),
        )

        # Apply index filtering if provided (for bucket-based filtering)
        filtered_indices = dataset_info.get("filtered_indices")
        if filtered_indices is not None:
            self.logger.info(
                f"Worker {self.worker_id}: Applying index filter ({len(filtered_indices)} samples)"
            )
            dataset = dataset.select(filtered_indices)

        total_stats = WorkerStats()

        while True:
            # Get next shard
            shard_id = ray.get(work_queue.get_next_shard.remote())
            if shard_id is None:
                break

            try:
                # Get shard data
                shard_data = dataset.shard(
                    num_shards=num_shards,
                    index=shard_id,
                    contiguous=True,
                )

                self.logger.info(
                    f"Worker {self.worker_id}: Processing shard {shard_id}/{num_shards} "
                    f"({len(shard_data)} samples)"
                )

                # Process shard
                shard_stats = self.process_shard(shard_id, shard_data, num_shards)

                # Update totals
                total_stats.samples_processed += shard_stats["samples_processed"]
                total_stats.tokens_generated += shard_stats["tokens_generated"]
                total_stats.errors += shard_stats["errors"]
                total_stats.samples_skipped += shard_stats["samples_skipped"]
                total_stats.duration_skipped += shard_stats["duration_skipped"]

                # Update progress
                if progress_actor is not None:
                    ray.get(progress_actor.update.remote(shard_stats["samples_processed"]))

                # Mark completed
                ray.get(work_queue.mark_completed.remote(shard_id))

                self.logger.info(
                    f"Worker {self.worker_id}: Completed shard {shard_id}. "
                    f"Processed: {total_stats.samples_processed}, Tokens: {total_stats.tokens_generated}"
                )

            except Exception as e:
                self.logger.error(f"Worker {self.worker_id}: Error processing shard {shard_id}: {e}")
                ray.get(work_queue.mark_failed.remote(shard_id))
                total_stats.errors += 1

        # Finalize stats
        final_stats = total_stats.finalize()
        final_stats["worker_id"] = self.worker_id

        return final_stats
