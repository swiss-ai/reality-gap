#!/usr/bin/env python3
"""
Base pipeline class for audio tokenization and shared utilities.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import ray
import torch
from tqdm import tqdm


class BasePipeline(ABC):
    """Abstract base class for audio tokenization pipelines."""

    def __init__(
        self,
        tokenizer_path: str,
        output_dir: str,
        num_gpus: int,
        device: str,
        target_sample_rate: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize base pipeline.

        Args:
            tokenizer_path: Path to omni-tokenizer
            output_dir: Output directory for tokenized data
            num_gpus: Number of parallel workers
            device: Device for tokenization (cuda/cpu)
            **kwargs: Additional arguments
        """
        self.tokenizer_path = tokenizer_path
        self.output_dir = output_dir
        self.num_gpus = num_gpus
        self.device = device
        self.target_sample_rate = target_sample_rate

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)

        # Store additional kwargs
        self.kwargs = kwargs

    @abstractmethod
    def setup(self):
        """Setup the pipeline (initialize workers, etc.)."""
        pass

    @abstractmethod
    def process(self):
        """Main processing logic."""
        pass

    @abstractmethod
    def cleanup(self):
        """Cleanup resources."""
        pass

    def generate_metadata(self, results: list, processing_time: float, **kwargs) -> Dict[str, Any]:
        """
        Generate standardized metadata from worker results.

        Args:
            results: List of worker statistics
            processing_time: Total processing time in seconds
            **kwargs: Additional metadata fields (dataset_name, config_name, etc.)

        Returns:
            Metadata dictionary
        """
        # Calculate aggregate statistics
        total_samples = sum(r["samples_processed"] for r in results)
        total_tokens = sum(r["tokens_generated"] for r in results)
        total_errors = sum(r["errors"] for r in results)
        total_skipped = sum(r.get("samples_skipped", 0) for r in results)
        total_duration_skipped = sum(r.get("duration_skipped", 0) for r in results)
        total_audio_tokens = sum(r.get("audio_tokens", 0) for r in results)
        total_text_tokens = sum(r.get("text_tokens", 0) for r in results)

        metadata = {
            "statistics": {
                "total_samples_processed": total_samples,
                "samples_skipped": total_skipped,
                "duration_skipped": total_duration_skipped,
                "total_tokens": total_tokens,
                "audio_tokens": total_audio_tokens,
                "text_tokens": total_text_tokens,
                "errors": total_errors,
            },
            "averages": {
                "tokens_per_sample": total_tokens / total_samples if total_samples > 0 else 0,
                "audio_tokens_per_sample": total_audio_tokens / total_samples if total_samples > 0 else 0,
                "text_tokens_per_sample": total_text_tokens / total_samples if total_samples > 0 else 0,
            },
            "token_distribution": {
                "audio_percentage": total_audio_tokens / total_tokens * 100 if total_tokens > 0 else 0,
                "text_percentage": total_text_tokens / total_tokens * 100 if total_tokens > 0 else 0,
            },
            "processing": {
                "num_gpus": len(results),
                "processing_time_seconds": processing_time,
                "samples_per_second": total_samples / processing_time if processing_time > 0 else 0,
                "tokens_per_second": total_tokens / processing_time if processing_time > 0 else 0,
            },
            "worker_details": [
                {
                    "worker_id": i,
                    "samples_processed": r["samples_processed"],
                    "tokens_generated": r["tokens_generated"],
                    "audio_tokens": r.get("audio_tokens", 0),
                    "text_tokens": r.get("text_tokens", 0),
                    "errors": r["errors"],
                    "samples_skipped": r.get("samples_skipped", 0),
                    "duration_skipped": r.get("duration_skipped", 0),
                    "throughput": r.get("throughput", 0),
                }
                for i, r in enumerate(results)
            ],
        }

        # Add any additional kwargs to metadata
        metadata.update(kwargs)

        return metadata

    def run(self) -> Dict[str, Any]:
        """
        Run the complete pipeline.

        Returns:
            Dictionary with processing results
        """
        try:
            self.logger.info(f"Starting {self.__class__.__name__}")

            # Setup
            self.setup()

            # Process
            result = self.process()

            # Cleanup
            self.cleanup()

            self.logger.info(f"Completed {self.__class__.__name__}")
            return result

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            self.cleanup()
            raise


@ray.remote
class ProgressActor:
    """Actor for collecting progress updates and W&B logging."""

    def __init__(
        self,
        total_samples: int,
        total_shards: int = 0,
        wandb_enabled: bool = False,
        desc: str = "Samples processed",
    ):
        self.total_samples = total_samples
        self.total_shards = total_shards
        self.processed = 0
        self.shards_completed = 0
        self.pbar = tqdm(total=total_samples, desc=desc)

        # W&B state
        self.wandb_enabled = wandb_enabled
        self._wandb = None
        if wandb_enabled:
            import wandb
            self._wandb = wandb

        # Tracking
        self.start_time = time.time()
        self.shard_history: List[Dict[str, Any]] = []

    def update(self, samples: int):
        """Update progress bar with completed samples."""
        self.processed += samples
        self.pbar.update(samples)

    def log_shard(
        self,
        shard_id: int,
        worker_id: int,
        samples_processed: int,
        tokens_generated: int,
        errors: int,
        duration_seconds: float,
    ):
        """Log shard completion with metrics to W&B."""
        self.shards_completed += 1
        elapsed_total = time.time() - self.start_time

        shard_metrics = {
            "shard_id": shard_id,
            "worker_id": worker_id,
            "samples_processed": samples_processed,
            "tokens_generated": tokens_generated,
            "errors": errors,
            "duration_seconds": duration_seconds,
            "samples_per_second": samples_processed / duration_seconds if duration_seconds > 0 else 0,
            "tokens_per_second": tokens_generated / duration_seconds if duration_seconds > 0 else 0,
        }
        self.shard_history.append(shard_metrics)

        if self.wandb_enabled and self._wandb:
            self._wandb.log({
                # Per-shard metrics
                "shard/id": shard_id,
                "shard/worker_id": worker_id,
                "shard/samples_processed": samples_processed,
                "shard/tokens_generated": tokens_generated,
                "shard/errors": errors,
                "shard/duration_seconds": duration_seconds,
                "shard/samples_per_second": shard_metrics["samples_per_second"],
                "shard/tokens_per_second": shard_metrics["tokens_per_second"],
                # Cumulative progress
                "progress/shards_completed": self.shards_completed,
                "progress/shards_total": self.total_shards,
                "progress/shards_percent": self.shards_completed / self.total_shards * 100 if self.total_shards > 0 else 0,
                "progress/samples_processed": self.processed,
                "progress/samples_total": self.total_samples,
                "progress/samples_percent": self.processed / self.total_samples * 100 if self.total_samples > 0 else 0,
                "progress/elapsed_seconds": elapsed_total,
                "progress/avg_samples_per_second": self.processed / elapsed_total if elapsed_total > 0 else 0,
            })

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        elapsed = time.time() - self.start_time
        return {
            "total_processed": self.processed,
            "shards_completed": self.shards_completed,
            "elapsed_seconds": elapsed,
            "avg_samples_per_second": self.processed / elapsed if elapsed > 0 else 0,
            "shard_history": self.shard_history,
        }

    def close(self):
        """Close the progress bar and return final count."""
        self.pbar.close()
        return self.processed


@dataclass
class WorkerStats:
    """Statistics tracked by workers."""
    samples_processed: int = 0
    tokens_generated: int = 0
    audio_tokens: int = 0
    text_tokens: int = 0
    errors: int = 0
    samples_skipped: int = 0
    duration_skipped: int = 0
    start_time: float = field(default_factory=time.time)
    elapsed_time: float = 0.0
    throughput: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "samples_processed": self.samples_processed,
            "tokens_generated": self.tokens_generated,
            "audio_tokens": self.audio_tokens,
            "text_tokens": self.text_tokens,
            "errors": self.errors,
            "samples_skipped": self.samples_skipped,
            "duration_skipped": self.duration_skipped,
            "elapsed_time": self.elapsed_time,
            "throughput": self.throughput,
        }

    def finalize(self) -> Dict[str, Any]:
        """Finalize stats and return as dict."""
        self.elapsed_time = time.time() - self.start_time
        self.throughput = self.tokens_generated / self.elapsed_time if self.elapsed_time > 0 else 0
        return self.to_dict()


class BaseAudioTokenizerWorker:
    """
    Base class for audio tokenization workers.
    Contains shared tokenizer initialization and processing logic.
    Subclasses handle output file creation and work distribution.
    """

    def __init__(
        self,
        tokenizer_path: str,
        worker_id: int,
        mode: str,  # "audio_only", "audio2text", "text2audio", or "sft"
        audio_field: str = "audio",
        text_field: str = "text",
        device: str = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
    ):
        self.worker_id = worker_id
        self.mode = mode
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.audio_field = audio_field
        self.text_field = text_field
        self.min_duration = min_duration
        self.max_duration = max_duration

        # Setup logging
        self.logger = logging.getLogger(f"Worker{worker_id:02d}")
        self.logger.setLevel(logging.INFO)

        # Import here to avoid circular imports
        from audio_tokenization.vokenizers import create_tokenizer

        # Initialize tokenizer using factory function
        self.tokenizer = create_tokenizer(
            omni_tokenizer_path=tokenizer_path,
            device=self.device,
        )

        # Initialize stats
        self.stats = WorkerStats()

        self.logger.info(f"Worker {worker_id} initialized on {self.device} in {mode} mode")

    def get_audio_duration(self, audio, sample_rate: int) -> float:
        """Calculate audio duration in seconds."""
        if isinstance(audio, torch.Tensor):
            return audio.shape[-1] / sample_rate
        elif hasattr(audio, "__len__"):
            return len(audio) / sample_rate
        return 0.0

    def should_process_duration(self, duration: float) -> bool:
        """
        Check if audio meets duration criteria for filtering.

        Args:
            duration: Audio duration in seconds

        Returns:
            True if audio should be processed, False if it should be skipped
        """
        if self.min_duration and duration < self.min_duration:
            return False
        if self.max_duration and duration > self.max_duration:
            return False
        return True

    def get_sample_status(self, audio, text) -> str:
        """
        Determine the processing status of a sample.

        Args:
            audio: Audio data (may be None)
            text: Text data (may be None)

        Returns:
            Status string: 'ok', 'data_skip', or 'duration_skip'
        """
        # Check for missing audio
        if audio is None:
            return "data_skip"

        # Check if mode requires text and it's missing
        if self.mode in ["audio2text", "text2audio", "sft"] and not text:
            return "data_skip"

        return "ok"

    def tokenize_sample(self, audio, sample_rate: int, text: str = None) -> Optional[Any]:
        """
        Tokenize a single sample (shared logic).

        Args:
            audio: Input audio
            sample_rate: Audio sample rate
            text: Input text (may be None for audio-only mode)

        Returns:
            Tokens (as list), or None if error
        """
        try:
            # Use unified tokenize method
            tokens = self.tokenizer.tokenize(audio, sample_rate)
            return tokens

        except Exception as e:
            self.logger.warning(f"Error processing sample: {e}")
            return None

    def _extract_data(self, sample: Dict) -> tuple:
        """
        Extract audio and/or text from sample based on mode.
        Can be overridden by specific implementations.
        """
        try:
            # Extract audio
            audio_data = sample[self.audio_field]

            # Handle HuggingFace audio format
            if isinstance(audio_data, dict):
                audio = audio_data.get("array")
                sample_rate = audio_data.get("sampling_rate", 16000)
            else:
                audio = audio_data
                sample_rate = sample.get("sample_rate", 16000)

            # Extract text (only for modes that need it)
            text = sample[self.text_field] if self.mode in ["audio2text", "text2audio", "sft"] else None

            return audio, sample_rate, text

        except KeyError as e:
            field = e.args[0]
            raise KeyError(
                f"Required field '{field}' not found in sample. "
                f"Available fields: {', '.join(sample.keys())}"
            ) from None

    def update_stats(
        self,
        samples: int = 0,
        tokens: int = 0,
        errors: int = 0,
        skipped: int = 0,
        duration_skipped: int = 0,
        audio_tokens: int = 0,
        text_tokens: int = 0,
    ):
        """Update worker statistics."""
        self.stats.samples_processed += samples
        self.stats.tokens_generated += tokens
        self.stats.audio_tokens += audio_tokens
        self.stats.text_tokens += text_tokens
        self.stats.errors += errors
        self.stats.samples_skipped += skipped
        self.stats.duration_skipped += duration_skipped

    def format_stats_message(self, prefix: str, stats: Dict, elapsed: float = None) -> str:
        """Format statistics into a clean log message."""
        samples = stats.get("samples", stats.get("samples_processed", 0))
        tokens = stats.get("tokens", stats.get("tokens_generated", 0))
        avg_tokens = tokens / samples if samples > 0 else 0

        # Base message
        parts = [f"{samples} samples", f"{tokens} tokens ({avg_tokens:.1f} avg)"]

        # Throughput
        if elapsed and elapsed > 0:
            parts.append(f"{samples / elapsed:.1f} samples/s")

        # Token breakdown (always show for non-audio_only modes)
        if self.mode != "audio_only":
            aud_tok = stats.get("audio_tokens", 0)
            txt_tok = stats.get("text_tokens", 0)
            if aud_tok > 0 or txt_tok > 0:
                parts.append(f"[audio: {aud_tok}, txt: {txt_tok}]")

        # Skip counts
        skips = []
        if stats.get("skipped", stats.get("samples_skipped", 0)) > 0:
            skips.append(f"{stats.get('skipped', stats.get('samples_skipped', 0))} data_skip")
        if stats.get("duration_skipped", 0) > 0:
            skips.append(f"{stats['duration_skipped']} dur_skip")
        if skips:
            parts.append(f"({', '.join(skips)})")

        return f"{prefix}: {', '.join(parts)}"

    def get_final_stats(self) -> Dict:
        """Get final statistics for the worker."""
        stats_dict = self.stats.finalize()

        msg = self.format_stats_message(f"Worker {self.worker_id} finished", stats_dict, self.stats.elapsed_time)
        self.logger.info(msg)

        return stats_dict
