#!/usr/bin/env python3
"""
Bucketed HuggingFace datasets tokenization pipeline.
Pre-filters samples by length bucket before distribution to workers.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import ray

from audio_tokenization.pipelines.hf.pipeline import HFDatasetPipeline
from audio_tokenization.utils.ray_utils import init_ray
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
        target_bucket: int,
        shuffle_seed: int = 42,
        **kwargs,
    ):
        """Initialize BucketedHFDatasetPipeline.

        Args:
            bucket_metadata_dir: Directory containing bucket TSV files
            target_bucket: Target bucket length to filter by (e.g., 240000 for 10-sec at 24kHz).
                All samples will have the same length, enabling efficient batch processing.
            shuffle_seed: Random seed for shuffling filtered indices (ensures even shard
                distribution and reproducibility for resume). Default: 42.
            **kwargs: Arguments passed to HFDatasetPipeline
        """
        super().__init__(**kwargs)

        self.bucket_metadata_dir = bucket_metadata_dir
        self.target_bucket = target_bucket
        self.shuffle_seed = shuffle_seed
        self._filtered_indices: Optional[np.ndarray] = None
        self._filtered_indices_ref = None
        self.bucket_index: Optional[BucketIndex] = None

    def _get_split_name(self) -> str:
        """Extract split name from dataset_split (handles slicing notation)."""
        return self.dataset_split.split("[", 1)[0]

    def _verify_resume_compatibility(self, output_dir: str) -> None:
        """Verify that resume parameters match the existing run.

        Checks dataset_info.json for critical parameters that must match
        for resume to work correctly (same samples in same order).

        Args:
            output_dir: Directory containing dataset_info.json

        Raises:
            SystemExit: If parameters don't match
        """
        metadata_path = Path(output_dir) / "dataset_info.json"
        if not metadata_path.exists():
            self.logger.info("Resume: No existing metadata found, starting fresh")
            return

        try:
            with open(metadata_path, "r") as f:
                existing_metadata = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self.logger.warning(f"Resume: Could not read metadata file: {e}")
            return

        # Extract existing values from metadata structure
        existing_bucket = existing_metadata.get("bucket_filtering", {})
        existing_processing = existing_metadata.get("processing", {})

        # Critical parameters that must match for reproducibility
        checks = [
            ("shuffle_seed", self.shuffle_seed, existing_bucket.get("shuffle_seed")),
            ("target_bucket", self.target_bucket, existing_bucket.get("target_bucket")),
            ("num_shards", self.num_shards, existing_metadata.get("num_shards")),
            ("dataset_name", self.dataset_name, existing_metadata.get("dataset_name")),
            ("config_name", self.config_name, existing_metadata.get("config_name")),
            ("dataset_split", self.dataset_split, existing_metadata.get("split")),
        ]

        mismatches = []
        for param_name, current_value, existing_value in checks:
            if existing_value is not None and current_value != existing_value:
                mismatches.append(f"  - {param_name}: current={current_value}, existing={existing_value}")

        if mismatches:
            self.logger.error("Resume ERROR: Configuration mismatch detected!")
            self.logger.error("The following parameters don't match the existing run:")
            for mismatch in mismatches:
                self.logger.error(mismatch)
            self.logger.error("")
            self.logger.error("To fix this, either:")
            self.logger.error("  1. Use matching parameters for resume")
            self.logger.error("  2. Delete the output directory and start fresh")
            self.logger.error("  3. Use a different output directory")
            sys.exit(1)

        self.logger.info("Resume: Configuration verified - parameters match existing run")

    def setup(self):
        """Setup with bucket filtering before Ray distribution."""
        self.logger.info(f"Initializing Ray with {self.num_gpus} workers")

        # Initialize Ray with GPU support
        init_ray(self.ray_config, self.num_gpus)

        # Load bucket index
        split_name = self._get_split_name()
        self.bucket_index = BucketIndex(self.bucket_metadata_dir, split_name)
        self.bucket_index.load()

        # Log available buckets
        bucket_counts = self.bucket_index.get_bucket_counts()
        self.logger.info(f"Loaded bucket index with {len(bucket_counts)} buckets, {len(self.bucket_index)} total samples")

        # Get filtered indices for target bucket (single bucket only)
        self._filtered_indices = np.array(
            self.bucket_index.get_indices(self.target_bucket),
            dtype=np.int64,
        )
        self.logger.info(
            f"Bucket filter: target={self.target_bucket} -> {len(self._filtered_indices)} samples"
        )

        # Shuffle indices with fixed seed for even shard distribution and reproducibility
        rng = np.random.default_rng(seed=self.shuffle_seed)
        self._filtered_indices = rng.permutation(self._filtered_indices)
        self.logger.info(f"Shuffled indices with seed={self.shuffle_seed} for even shard distribution")

        # Apply max_samples limit if specified
        if self.max_samples and len(self._filtered_indices) > self.max_samples:
            self._filtered_indices = self._filtered_indices[:self.max_samples]
            self.logger.info(f"Applied max_samples limit: {self.max_samples} samples")

        self.total_samples = len(self._filtered_indices)
        self._filtered_indices_ref = ray.put(self._filtered_indices)

        self.logger.info(f"Processing {self.total_samples} samples")

        self._validate_shard_counts()

        # Auto-create output subdirectory from dataset/config/split/mode/bucket
        self.output_dir = str(
            Path(self.output_dir) / self._build_output_subdir(extra_suffix=f"bucket_{self.target_bucket}")
        )

        self.logger.info(f"Output directory: {self.output_dir}")

        # Verify resume compatibility before proceeding
        if self.resume:
            self._verify_resume_compatibility(self.output_dir)

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize W&B if enabled (driver-only or live actor)
        if self.wandb_enabled:
            wandb_name = self.wandb_config.get("name")
            if wandb_name is None:
                # Auto-generate name from dataset/config/split/mode/bucket
                wandb_name = self._append_node_suffix(
                    self._build_output_subdir(extra_suffix=f"bucket_{self.target_bucket}")
                )

            wandb_run_config = self._build_wandb_run_config()
            wandb_run_config.update({
                "bucket_metadata_dir": self.bucket_metadata_dir,
                "target_bucket": self.target_bucket,
                "shuffle_seed": self.shuffle_seed,
            })

            if self.wandb_live:
                from audio_tokenization.utils.wandb_logger import WandbLoggerActor

                self._wandb_logger = WandbLoggerActor.remote(
                    project=self.wandb_config.get("project", "audio-tokenization"),
                    entity=self.wandb_config.get("entity"),
                    name=wandb_name,
                    tags=self.wandb_config.get("tags", []),
                    config=wandb_run_config,
                    total_samples=self.total_samples,
                    log_interval_seconds=self.wandb_log_interval,
                )
                self.logger.info("W&B live logger actor initialized")
            else:
                import wandb

                self._wandb_run = wandb.init(
                    project=self.wandb_config.get("project", "audio-tokenization"),
                    entity=self.wandb_config.get("entity"),
                    name=wandb_name,
                    tags=self.wandb_config.get("tags", []),
                    config=wandb_run_config,
                    resume="allow",
                )
                self.logger.info(f"W&B run initialized: {self._wandb_run.name}")

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
            # NEW: Pass filtered indices via object store to share across workers
            "filtered_indices_ref": self._filtered_indices_ref,
        }

        # Start workers processing shards
        worker_futures = []
        if self.shard_assignment == "shared":
            worker_futures = [
                worker.run_shards.remote(self.work_queue, dataset_info, self.num_shards, self.progress_actor)
                for worker in self.workers
            ]
        else:
            for i, worker in enumerate(self.workers):
                shard_ids = self.shards_by_worker[i] if self.shards_by_worker else []
                worker_futures.append(
                    worker.run_shards.remote(shard_ids, dataset_info, self.num_shards, self.progress_actor)
                )

        # Wait for completion
        results = ray.get(worker_futures)

        # Get final progress
        ray.get(self.progress_actor.close.remote())

        # Calculate summary statistics
        total_samples_processed = sum(r["samples_processed"] for r in results)
        total_tokens = sum(r["tokens_generated"] for r in results)
        total_errors = sum(r["errors"] for r in results)
        max_time = max(r["elapsed_time"] for r in results)

        # Log per-shard metrics to W&B (driver-only)
        if self.wandb_enabled and self._wandb_run is not None:
            import wandb

            step = 0
            for worker_result in results:
                for shard_stats in worker_result.get("shard_stats", []):
                    samples = shard_stats.get("samples_processed", 0)
                    tokens = shard_stats.get("tokens_generated", 0)
                    elapsed = shard_stats.get("elapsed_time", 0)
                    wandb.log(
                        {
                            "shard/worker_id": shard_stats.get("worker_id"),
                            "shard/shard_id": shard_stats.get("shard_id"),
                            "shard/samples_processed": samples,
                            "shard/tokens_generated": tokens,
                            "shard/errors": shard_stats.get("errors", 0),
                            "shard/samples_skipped": shard_stats.get("samples_skipped", 0),
                            "shard/duration_skipped": shard_stats.get("duration_skipped", 0),
                            "shard/frequency_skipped": shard_stats.get("frequency_skipped", 0),
                            "shard/elapsed_time": elapsed,
                            "shard/samples_per_second": samples / elapsed if elapsed > 0 else 0,
                            "shard/tokens_per_second": tokens / elapsed if elapsed > 0 else 0,
                        },
                        step=step,
                    )
                    step += 1

        # Log final metrics to W&B
        if self.wandb_enabled and self._wandb_run is not None:
            import wandb

            wandb.log({
                "final/total_samples_processed": total_samples_processed,
                "final/total_tokens": total_tokens,
                "final/total_errors": total_errors,
                "final/processing_time_seconds": max_time,
                "final/samples_per_second": total_samples_processed / max_time if max_time > 0 else 0,
                "final/tokens_per_second": total_tokens / max_time if max_time > 0 else 0,
                "final/target_bucket": self.target_bucket,
            })

        # Log final metrics to live W&B logger
        if self.wandb_enabled and self.wandb_live and self._wandb_logger is not None:
            ray.get(self._wandb_logger.log_final.remote({
                "total_samples_processed": total_samples_processed,
                "total_tokens": total_tokens,
                "total_errors": total_errors,
                "processing_time_seconds": max_time,
                "samples_per_second": total_samples_processed / max_time if max_time > 0 else 0,
                "tokens_per_second": total_tokens / max_time if max_time > 0 else 0,
                "target_bucket": self.target_bucket,
            }))

        # Save metadata
        self._save_metadata(results, max_time)

        # Merge index files if needed
        self._merge_results()

        return {
            "total_samples": total_samples_processed,
            "total_tokens": total_tokens,
            "total_errors": total_errors,
            "processing_time": max_time,
            "workers": len(self.workers),
            "mode": self.mode,
            "output_dir": self.output_dir,
            "metadata_file": str(Path(self.output_dir) / "dataset_info.json"),
            "bucket_filter": self.target_bucket,
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
            num_shards=self.num_shards,  # Critical for resume verification
            audio_field=self.audio_field,
            text_field=self.text_field,
            tokenizer={
                "path": self.tokenizer_path,
                "sampling_rate": self.target_sample_rate,
            },
            audio_filtering={
                "min_duration": self.min_duration,
                "max_duration": self.max_duration,
                "min_sample_rate": self.min_sample_rate,
            },
            bucket_filtering={
                "metadata_dir": self.bucket_metadata_dir,
                "target_bucket": self.target_bucket,
                "shuffle_seed": self.shuffle_seed,
                "filtered_sample_count": len(self._filtered_indices),
            },
        )

        # Save to file
        metadata_path = Path(self.output_dir) / "dataset_info.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Saved metadata to {metadata_path}")
