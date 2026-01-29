#!/usr/bin/env python3
"""
HuggingFace datasets tokenization pipeline.
Handles both audio-only and SFT tokenization modes.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import ray

from audio_tokenization.pipelines.base import BasePipeline, ProgressActor
from audio_tokenization.pipelines.hf.workers import ShardQueue, Worker


class HFDatasetPipeline(BasePipeline):
    """Pipeline for tokenizing HuggingFace datasets."""

    def __init__(
        self,
        tokenizer_path: str,
        output_dir: str,
        dataset_name: str,
        dataset_split: str,
        mode: str,  # "audio_only", "audio2text", "text2audio", or "sft"
        num_gpus: int,
        device: str,
        num_shards: int,  # Required for checkpointing
        config_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        num_proc: int = 32,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        max_samples: Optional[int] = None,
        audio_field: str = "audio",
        text_field: str = "text",
        resume: bool = False,
        **kwargs,
    ):
        super().__init__(tokenizer_path, output_dir, num_gpus, device, **kwargs)

        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.config_name = config_name
        self.cache_dir = cache_dir
        self.num_proc = num_proc
        self.mode = mode
        self.num_shards = num_shards
        self.total_samples = 0

        self.min_duration = min_duration
        self.max_duration = max_duration
        self.max_samples = max_samples
        self.audio_field = audio_field
        self.text_field = text_field
        self.resume = resume

        # Validate mode
        valid_modes = ["audio_only", "audio2text", "text2audio", "sft"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")

        # Validate num_shards is sufficient for workers
        if self.num_shards < self.num_gpus:
            self.logger.warning(
                f"num_shards ({self.num_shards}) < num_gpus ({self.num_gpus}). "
                f"Adjusting num_shards to {self.num_gpus} to ensure all workers have work."
            )
            self.num_shards = self.num_gpus

    def _get_completed_shards(self) -> set:
        """Get list of already completed shards by checking for .idx files."""
        completed = set()
        output_path = Path(self.output_dir)

        if not output_path.exists():
            return completed

        # Pattern: rank_X_shard_Y_Z.idx where Y is shard_id and Z is total_shards
        pattern = re.compile(r"rank_\d+_shard_(\d+)_(\d+)\.idx")

        # Collect all shard counts found
        shard_counts_found = set()
        files_by_shard_count = {}

        for idx_file in output_path.glob("*.idx"):
            match = pattern.match(idx_file.name)
            if match:
                shard_id = int(match.group(1))
                total_shards = int(match.group(2))

                shard_counts_found.add(total_shards)
                if total_shards not in files_by_shard_count:
                    files_by_shard_count[total_shards] = []
                files_by_shard_count[total_shards].append(idx_file.name)

                if total_shards == self.num_shards:
                    completed.add(shard_id)

        # Check for inconsistency
        if shard_counts_found and self.num_shards not in shard_counts_found:
            # No files match the expected shard count
            self.logger.error(
                f"ERROR: No existing shards match expected count ({self.num_shards}). "
                f"Found shard counts: {sorted(shard_counts_found)}"
            )
            for count in sorted(shard_counts_found):
                self.logger.error(f"  {count} total shards: {len(files_by_shard_count[count])} files")
            self.logger.error(
                f"To resume, use --num-shards {sorted(shard_counts_found)[0]} or start fresh without --resume"
            )
            sys.exit(1)

        if len(shard_counts_found) > 1:
            # Multiple different shard counts found
            self.logger.error(f"ERROR: Inconsistent total shard counts found: {sorted(shard_counts_found)}")
            for count in sorted(shard_counts_found):
                self.logger.error(f"  {count} total shards: {len(files_by_shard_count[count])} files")
            self.logger.error("Clean the output directory or use a different output path")
            sys.exit(1)

        return completed

    def setup(self):
        """Setup Ray and load dataset metadata."""
        self.logger.info(f"Initializing Ray with {self.num_gpus} workers")

        # Initialize Ray with GPU support
        if not ray.is_initialized():
            ray.init(num_cpus=self.num_gpus + 2, num_gpus=self.num_gpus)

        # Load dataset metadata for sample counts
        config_info = f" (config: {self.config_name})" if self.config_name else ""
        self.logger.info(f"Loading dataset info: {self.dataset_name}{config_info}/{self.dataset_split}")
        from datasets import load_dataset_builder

        builder = load_dataset_builder(
            self.dataset_name,
            name=self.config_name,
            cache_dir=self.cache_dir,
        )
        split_name = self.dataset_split.split("[", 1)[0]
        split_info = builder.info.splits.get(split_name)
        if split_info is None:
            self.logger.warning(
                f"Split '{split_name}' not found in dataset info; progress totals may be off."
            )
            total_samples = 0
        else:
            total_samples = split_info.num_examples or 0
            if split_info.num_examples is None:
                self.logger.warning(
                    f"Split '{split_name}' has no num_examples; progress totals may be off."
                )

        if self.max_samples:
            total_samples = min(self.max_samples, total_samples) if total_samples else self.max_samples

        self.total_samples = total_samples
        self.logger.info(f"Processing {self.total_samples} samples")

        # Auto-create output subdirectory based on config_name and mode if config_name is provided
        if self.config_name:
            self.output_dir = str(Path(self.output_dir) / f"{self.config_name}_{self.mode}")

        self.logger.info(f"Output directory: {self.output_dir}")

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Setup workers
        self._setup_workers()

    def _setup_workers(self):
        """Setup workers for shard-based tokenization.

        Creates a work queue that distributes shards to workers. Each shard
        will be processed completely by a worker and saved as a separate output file.
        """
        # Create progress tracker
        self.progress_actor = ProgressActor.remote(self.total_samples)

        # Check for existing completed shards if resuming
        if self.resume:
            completed_shards = self._get_completed_shards()
            if completed_shards:
                self.logger.info(
                    f"Resume mode: Found {len(completed_shards)} completed shards: {sorted(completed_shards)}"
                )
                # Create work queue with only uncompleted shards
                uncompleted = [i for i in range(self.num_shards) if i not in completed_shards]
                self.logger.info(
                    f"Will process {len(uncompleted)} remaining shards: {uncompleted[:10]}{'...' if len(uncompleted) > 10 else ''}"
                )
                self.work_queue = ShardQueue.remote(self.num_shards, initial_shards=uncompleted)
            else:
                self.logger.info("Resume mode: No completed shards found, starting from beginning")
                self.work_queue = ShardQueue.remote(self.num_shards)
        else:
            # Create work queue for shard distribution
            self.work_queue = ShardQueue.remote(self.num_shards)

        # Start unified workers
        self.workers = []
        for i in range(self.num_gpus):
            worker = Worker.remote(
                tokenizer_path=self.tokenizer_path,
                output_dir=self.output_dir,
                worker_id=i,
                mode=self.mode,
                audio_field=self.audio_field,
                text_field=self.text_field,
                min_duration=self.min_duration,
                max_duration=self.max_duration,
            )
            self.workers.append(worker)

        self.logger.info(f"Setup {self.num_gpus} workers for {self.mode} mode")

    def process(self) -> Dict[str, Any]:
        """Process the dataset using shard-based approach."""
        # Create dataset info dict to pass to workers
        dataset_info = {
            "name": self.dataset_name,
            "config": self.config_name,
            "split": self.dataset_split,
            "cache_dir": self.cache_dir,
            "total_samples": self.total_samples,
            "max_samples": self.max_samples,
        }

        # Start workers processing shards
        worker_futures = [
            worker.run_shards.remote(self.work_queue, dataset_info, self.num_shards, self.progress_actor)
            for worker in self.workers
        ]

        # Wait for completion
        results = ray.get(worker_futures)

        # Get final progress
        total_processed = ray.get(self.progress_actor.close.remote())

        # Calculate summary statistics
        total_samples_processed = sum(r["samples_processed"] for r in results)
        total_tokens = sum(r["tokens_generated"] for r in results)
        total_errors = sum(r["errors"] for r in results)
        max_time = max(r["elapsed_time"] for r in results)

        # Save metadata
        self._save_metadata(results, max_time)

        # Merge index files if needed
        self._merge_results()

        return {
            "total_processed": total_processed,
            "total_samples": total_samples_processed,
            "total_tokens": total_tokens,
            "total_errors": total_errors,
            "processing_time": max_time,
            "workers": len(self.workers),
            "mode": self.mode,
            "output_dir": self.output_dir,
            "metadata_file": str(Path(self.output_dir) / "dataset_info.json"),
        }

    def _merge_results(self):
        """Merge results from all workers."""
        # Implementation depends on the specific format
        # This would merge the individual worker outputs
        self.logger.info(f"Merging results in {self.output_dir}")

    def _save_metadata(self, results: list, processing_time: float):
        """Save dataset processing metadata to JSON file."""
        # Generate metadata using base pipeline method
        metadata = self.generate_metadata(
            results=results,
            processing_time=processing_time,
            dataset_type=f"{self.mode} tokenization",
            dataset_name=self.dataset_name,
            config_name=self.config_name,
            split=self.dataset_split,
            mode=self.mode,
            audio_field=self.audio_field,
            text_field=self.text_field,
            tokenizer={
                "path": self.tokenizer_path,
            },
            audio_filtering={
                "min_duration": self.min_duration,
                "max_duration": self.max_duration,
            },
        )

        # Save to file
        metadata_path = Path(self.output_dir) / "dataset_info.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Saved metadata to {metadata_path}")

    def cleanup(self):
        """Cleanup Ray resources."""
        if ray.is_initialized():
            ray.shutdown()


def run_hf_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run HF dataset pipeline with configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Processing results
    """
    pipeline = HFDatasetPipeline(**config)
    return pipeline.run()
