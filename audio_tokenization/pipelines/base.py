#!/usr/bin/env python3
"""
Base pipeline class for tokenization and shared utilities.
Adapted from vision_tokenization for audio modality.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
import ray
import torch
from tqdm import tqdm


class BasePipeline(ABC):
    """Abstract base class for tokenization pipelines."""

    def __init__(
        self,
        tokenizer_path: str,
        output_dir: str,
        num_gpus: int,
        device: str,
        **kwargs
    ):
        """
        Initialize base pipeline.

        Args:
            tokenizer_path: Path to tokenizer
            output_dir: Output directory for tokenized data
            num_gpus: Number of parallel workers
            device: Device for tokenization (cuda/cpu)
            **kwargs: Additional arguments
        """
        self.tokenizer_path = tokenizer_path
        self.output_dir = output_dir
        self.num_gpus = num_gpus
        self.device = device

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
        total_samples = sum(r['samples_processed'] for r in results)
        total_tokens = sum(r['tokens_generated'] for r in results)
        total_errors = sum(r['errors'] for r in results)
        total_skipped = sum(r.get('samples_skipped', 0) for r in results)

        metadata = {
            'statistics': {
                'total_samples_processed': total_samples,
                'samples_skipped': total_skipped,
                'total_tokens': total_tokens,
                'errors': total_errors
            },
            'averages': {
                'tokens_per_sample': total_tokens / total_samples if total_samples > 0 else 0,
            },
            'processing': {
                'num_gpus': len(results),
                'processing_time_seconds': processing_time,
                'samples_per_second': total_samples / processing_time if processing_time > 0 else 0,
                'tokens_per_second': total_tokens / processing_time if processing_time > 0 else 0
            },
            'worker_details': [
                {
                    'worker_id': i,
                    'samples_processed': r['samples_processed'],
                    'tokens_generated': r['tokens_generated'],
                    'errors': r['errors'],
                    'samples_skipped': r.get('samples_skipped', 0),
                    'throughput': r.get('throughput', 0)
                } for i, r in enumerate(results)
            ]
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
    """Lightweight actor for collecting progress updates without polling."""

    def __init__(self, total_samples: int, desc: str = "Samples processed"):
        self.total_samples = total_samples
        self.processed = 0
        self.pbar = tqdm(total=total_samples, desc=desc)

    def update(self, samples: int):
        """Update progress bar with completed samples."""
        self.processed += samples
        self.pbar.update(samples)

    def close(self):
        """Close the progress bar."""
        self.pbar.close()
        return self.processed


class BaseTokenizerWorker:
    """
    Base class for tokenization workers.
    Contains shared tokenizer initialization and processing logic.
    Subclasses handle output file creation and work distribution.
    """

    def __init__(
        self,
        tokenizer_path: str,
        worker_id: int,
        mode: str,  # "audio_only"
        audio_field: str = "audio",
        device: str = None,
    ):
        self.worker_id = worker_id
        self.mode = mode
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.audio_field = audio_field

        # Setup logging
        self.logger = logging.getLogger(f"Worker{worker_id:02d}")
        self.logger.setLevel(logging.INFO)

        # Import here to avoid circular imports
        from audio_tokenization.vokenizers.audio import create_tokenizer

        # Initialize tokenizer using factory function
        self.tokenizer = create_tokenizer(
            mode=mode,
            text_tokenizer_path=tokenizer_path,
            device=self.device,
        )

        # Initialize stats
        import time
        self.stats = {
            'batches_processed': 0,
            'samples_processed': 0,
            'tokens_generated': 0,
            'errors': 0,
            'samples_skipped': 0,
            'start_time': time.time()
        }

        self.logger.info(f"Worker {worker_id} initialized on {self.device} in {mode} mode")

    def get_sample_status(self, audio) -> str:
        """
        Determine the processing status of a sample.

        Args:
            audio: Audio data (may be None)

        Returns:
            Status string: 'ok' or 'data_skip'
        """
        # Check for missing audio
        if audio is None:
            return 'data_skip'

        return 'ok'

    def tokenize_sample(self, audio, sampling_rate: int = None) -> Optional[Any]:
        """
        Tokenize a single sample (shared logic).

        Args:
            audio: Input audio (numpy array or torch tensor)
            sampling_rate: Sampling rate of the audio (optional, defaults to 16000)

        Returns:
            Tokens (as tensor or numpy array), or None if error
        """
        try:
            import torch

            # Use unified tokenize method
            if sampling_rate is None:
                sampling_rate = 16000  # Default for audio tokenization

            tokens = self.tokenizer.tokenize(audio=audio, sampling_rate=sampling_rate)

            # Convert to numpy if needed
            tokens_np = tokens.cpu().numpy() if torch.is_tensor(tokens) else tokens

            return tokens_np

        except Exception as e:
            self.logger.warning(f"Error processing sample: {e}")
            return None

    def _extract_data(self, sample: Dict) -> tuple:
        """
        Extract audio from sample based on mode.
        Can be overridden by specific implementations.

        Returns:
            Tuple of (audio, sampling_rate)
        """
        try:
            # Extract audio
            audio = sample[self.audio_field]
            # Unwrap single-item lists
            if isinstance(audio, list) and len(audio) == 1:
                audio = audio[0]
            # HF Audio object: {"array": ..., "sampling_rate": ...}
            if isinstance(audio, dict) and "array" in audio:
                audio = audio["array"]

            # Extract sampling_rate if available
            sampling_rate = sample.get("sampling_rate", None)
            if sampling_rate is None and isinstance(sample.get(self.audio_field), dict):
                sampling_rate = sample[self.audio_field].get("sampling_rate", None)

            return audio, sampling_rate

        except KeyError as e:
            field = e.args[0]
            raise KeyError(
                f"Required field '{field}' not found in sample. "
                f"Available fields: {', '.join(sample.keys())}"
            ) from None

    def update_stats(self, samples: int = 0, tokens: int = 0, errors: int = 0,
                     skipped: int = 0):
        """Update worker statistics."""
        self.stats['samples_processed'] += samples
        self.stats['tokens_generated'] += tokens
        self.stats['errors'] += errors
        self.stats['samples_skipped'] += skipped

    def format_stats_message(self, prefix: str, stats: Dict, elapsed: float = None) -> str:
        """Format statistics into a clean log message."""
        samples = stats.get('samples', stats.get('samples_processed', 0))
        tokens = stats.get('tokens', stats.get('tokens_generated', 0))
        avg_tokens = tokens / samples if samples > 0 else 0

        # Base message
        parts = [f"{samples} samples", f"{tokens} tokens ({avg_tokens:.1f} avg)"]

        # Throughput
        if elapsed and elapsed > 0:
            parts.append(f"{samples / elapsed:.1f} samples/s")

        # Skip counts
        skips = []
        if stats.get('skipped', stats.get('samples_skipped', 0)) > 0:
            skips.append(f"{stats.get('skipped', stats.get('samples_skipped', 0))} data_skip")
        if skips:
            parts.append(f"({', '.join(skips)})")

        return f"{prefix}: {', '.join(parts)}"

    def get_final_stats(self) -> Dict:
        """Get final statistics for the worker."""
        import time

        elapsed = time.time() - self.stats['start_time']
        self.stats['elapsed_time'] = elapsed
        self.stats['throughput'] = self.stats['tokens_generated'] / elapsed if elapsed > 0 else 0

        msg = self.format_stats_message(f"Worker {self.worker_id} finished", self.stats, elapsed)
        self.logger.info(msg)

        return self.stats

