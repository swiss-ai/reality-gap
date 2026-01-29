#!/usr/bin/env python3
"""
Bucketed HuggingFace datasets tokenization pipeline.
Pre-filters samples by length bucket before distribution to workers.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import ray

from audio_tokenization.pipelines.hf.pipeline import HFDatasetPipeline
from audio_tokenization.pipelines.hf.bucket_index import BucketIndex


class BucketedHFDatasetPipeline(HFDatasetPipeline):
    """Pipeline with bucket-based pre-filtering for efficient batch tokenization.

    This pipeline pre-filters samples by length bucket before distributing to Ray workers.
    The bucket metadata file contains GLOBAL_ID which maps directly to HuggingFace
    dataset indices, enabling O(1) filtering via dataset.select(indices).

    Example:
        >>> pipeline = BucketedHFDatasetPipeline(
        ...     tokenizer_path="/path/to/tokenizer",
        ...     output_dir="./output",
        ...     dataset_name="agkphysics/AudioSet",
        ...     dataset_split="bal_train",
        ...     mode="audio_only",
        ...     num_gpus=4,
        ...     num_shards=20,
        ...     bucket_metadata_dir="/path/to/buckets",
        ...     target_buckets=240000,  # 10-second clips
        ... )
        >>> result = pipeline.run()
    """

    def __init__(
        self,
        bucket_metadata_dir: str,
        target_buckets: Optional[Union[int, List[int]]] = None,
        **kwargs,
    ):
        """Initialize BucketedHFDatasetPipeline.

        Args:
            bucket_metadata_dir: Directory containing bucket TSV files
            target_buckets: Target bucket length(s) to filter by.
                Can be a single int (e.g., 240000) or list of ints.
                If None, all samples are processed (no filtering).
            **kwargs: Arguments passed to HFDatasetPipeline
        """
        super().__init__(**kwargs)

        self.bucket_metadata_dir = bucket_metadata_dir
        self.target_buckets = target_buckets
        self._filtered_indices: Optional[List[int]] = None
        self.bucket_index: Optional[BucketIndex] = None

    def _get_split_name(self) -> str:
        """Extract split name from dataset_split (handles slicing notation)."""
        return self.dataset_split.split("[", 1)[0]

    def setup(self):
        """Setup with bucket filtering before Ray distribution."""
        self.logger.info(f"Initializing Ray with {self.num_gpus} workers")

        # Initialize Ray with GPU support
        if not ray.is_initialized():
            ray.init(num_cpus=self.num_gpus + 2, num_gpus=self.num_gpus)

        # Load bucket index
        split_name = self._get_split_name()
        self.bucket_index = BucketIndex(self.bucket_metadata_dir, split_name)
        self.bucket_index.load()

        # Log available buckets
        bucket_counts = self.bucket_index.get_bucket_counts()
        self.logger.info(f"Loaded bucket index with {len(bucket_counts)} buckets, {len(self.bucket_index)} total samples")

        # Get filtered indices based on target buckets
        if self.target_buckets is not None:
            self._filtered_indices = self.bucket_index.get_indices(self.target_buckets)
            self.logger.info(
                f"Bucket filter: target={self.target_buckets} -> {len(self._filtered_indices)} samples"
            )

            # Apply max_samples limit if specified
            if self.max_samples and len(self._filtered_indices) > self.max_samples:
                self._filtered_indices = self._filtered_indices[:self.max_samples]
                self.logger.info(f"Applied max_samples limit: {self.max_samples} samples")

            self.total_samples = len(self._filtered_indices)
        else:
            # No bucket filtering, use all samples
            self._filtered_indices = None
            self.logger.info("No bucket filter specified, processing all samples")

            # Fall back to dataset metadata for total_samples
            from datasets import load_dataset_builder
            builder = load_dataset_builder(
                self.dataset_name,
                name=self.config_name,
                cache_dir=self.cache_dir,
            )
            split_info = builder.info.splits.get(split_name)
            if split_info and split_info.num_examples:
                total_samples = split_info.num_examples
            else:
                total_samples = 0

            if self.max_samples:
                total_samples = min(self.max_samples, total_samples) if total_samples else self.max_samples

            self.total_samples = total_samples

        self.logger.info(f"Processing {self.total_samples} samples")

        # Auto-create output subdirectory
        if self.config_name:
            bucket_suffix = f"_bucket_{self.target_buckets}" if self.target_buckets else ""
            self.output_dir = str(Path(self.output_dir) / f"{self.config_name}_{self.mode}{bucket_suffix}")

        self.logger.info(f"Output directory: {self.output_dir}")

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Setup workers
        self._setup_workers()

    def process(self) -> Dict[str, Any]:
        """Process the dataset using shard-based approach with bucket filtering."""
        # Create dataset info dict to pass to workers
        dataset_info = {
            "name": self.dataset_name,
            "config": self.config_name,
            "split": self.dataset_split,
            "cache_dir": self.cache_dir,
            "total_samples": self.total_samples,
            "max_samples": self.max_samples if self._filtered_indices is None else None,
            # NEW: Pass filtered indices for bucket-based filtering
            "filtered_indices": self._filtered_indices,
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
            "bucket_filter": self.target_buckets,
        }

    def _save_metadata(self, results: list, processing_time: float):
        """Save dataset processing metadata to JSON file."""
        # Generate metadata using base pipeline method
        metadata = self.generate_metadata(
            results=results,
            processing_time=processing_time,
            dataset_type=f"{self.mode} tokenization (bucketed)",
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
            bucket_filtering={
                "metadata_dir": self.bucket_metadata_dir,
                "target_buckets": self.target_buckets,
                "filtered_sample_count": len(self._filtered_indices) if self._filtered_indices else None,
            },
        )

        # Save to file
        metadata_path = Path(self.output_dir) / "dataset_info.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Saved metadata to {metadata_path}")
