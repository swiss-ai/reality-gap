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
from audio_tokenization.utils.ray_utils import init_ray
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
        shard_assignment: str = "shared",
        config_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        max_samples: Optional[int] = None,
        audio_field: str = "audio",
        text_field: str = "text",
        resume: bool = False,
        batch_size: int = 1,
        dataloader_workers: int = 0,
        dataloader_prefetch_factor: int = 2,
        dataloader_persistent_workers: bool = True,
        target_sample_rate: Optional[int] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        ray_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            tokenizer_path,
            output_dir,
            num_gpus,
            device,
            target_sample_rate=target_sample_rate,
            **kwargs,
        )

        # W&B config
        self.wandb_config = wandb_config or {}
        self.wandb_enabled = self.wandb_config.get("enabled", False)
        self.wandb_live = self.wandb_config.get("live", False)
        log_interval = self.wandb_config.get("log_interval_seconds")
        self.wandb_log_interval = 10 if log_interval is None else log_interval
        self._wandb_run = None
        self._wandb_logger = None

        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.config_name = config_name
        self.cache_dir = cache_dir
        self.mode = mode
        self.num_shards = num_shards
        self.shard_assignment = shard_assignment
        self.total_samples = 0

        self.min_duration = min_duration
        self.max_duration = max_duration
        self.max_samples = max_samples
        self.audio_field = audio_field
        self.text_field = text_field
        self.resume = resume
        self.batch_size = batch_size
        self.dataloader_workers = dataloader_workers
        self.dataloader_prefetch_factor = dataloader_prefetch_factor
        self.dataloader_persistent_workers = dataloader_persistent_workers

        # Ray config
        self.ray_config = ray_config or {}

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

        # Validate shard assignment mode
        if self.shard_assignment not in {"shared", "static"}:
            raise ValueError("shard_assignment must be 'shared' or 'static'")

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

    def _validate_shard_counts(self) -> None:
        """Validate shard counts against total samples."""
        if self.total_samples and self.num_shards > self.total_samples:
            raise ValueError(
                f"num_shards ({self.num_shards}) > total_samples ({self.total_samples}). "
                "Reduce num_shards or increase the dataset size."
            )

    def _build_wandb_run_config(self) -> Dict[str, Any]:
        """Build W&B config shared by HF pipelines."""
        return {
            "dataset_name": self.dataset_name,
            "config_name": self.config_name,
            "dataset_split": self.dataset_split,
            "mode": self.mode,
            "num_gpus": self.num_gpus,
            "num_shards": self.num_shards,
            "shard_assignment": self.shard_assignment,
            "batch_size": self.batch_size,
            "dataloader_workers": self.dataloader_workers,
            "dataloader_prefetch_factor": self.dataloader_prefetch_factor,
            "dataloader_persistent_workers": self.dataloader_persistent_workers,
            "target_sample_rate": self.target_sample_rate,
            "min_duration": self.min_duration,
            "max_duration": self.max_duration,
            "tokenizer_path": self.tokenizer_path,
            "ray_address": self.ray_config.get("address"),
            "total_samples": self.total_samples,
        }

    def setup(self):
        """Setup Ray and load dataset metadata."""
        self.logger.info(f"Initializing Ray with {self.num_gpus} workers")

        # Initialize Ray with GPU support
        init_ray(self.ray_config, self.num_gpus)

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

        self._validate_shard_counts()

        # Auto-create output subdirectory based on config_name and mode if config_name is provided
        if self.config_name:
            self.output_dir = str(Path(self.output_dir) / f"{self.config_name}_{self.mode}")

        self.logger.info(f"Output directory: {self.output_dir}")

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize W&B if enabled (driver-only or live actor)
        if self.wandb_enabled:
            wandb_name = self.wandb_config.get("name")
            if wandb_name is None:
                # Auto-generate name from dataset and mode
                config_suffix = f"_{self.config_name}" if self.config_name else ""
                wandb_name = f"{self.dataset_name.split('/')[-1]}{config_suffix}_{self.mode}"

            wandb_run_config = self._build_wandb_run_config()

            if self.wandb_live:
                from audio_tokenization.utils.wandb_logger import WandbLoggerActor

                self._wandb_logger = WandbLoggerActor.remote(
                    project=self.wandb_config.get("project", "audio-tokenization"),
                    entity=self.wandb_config.get("entity"),
                    name=wandb_name,
                    tags=self.wandb_config.get("tags", []),
                    config=wandb_run_config,
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

    def _setup_workers(self):
        """Setup workers for shard-based tokenization.

        Creates a work queue that distributes shards to workers. Each shard
        will be processed completely by a worker and saved as a separate output file.
        """
        # Create progress tracker with W&B support for live monitoring
        self.progress_actor = ProgressActor.remote(
            total_samples=self.total_samples,
            total_shards=self.num_shards,
            wandb_enabled=self.wandb_enabled,
        )

        self.work_queue = None
        self.shards_by_worker = None
        if self.shard_assignment == "shared":
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
        else:
            # Static shard assignment for per-worker DataLoader reuse
            if self.resume:
                completed_shards = self._get_completed_shards()
                if completed_shards:
                    self.logger.info(
                        f"Resume mode: Found {len(completed_shards)} completed shards: {sorted(completed_shards)}"
                    )
                else:
                    self.logger.info("Resume mode: No completed shards found, starting from beginning")
                pending_shards = [i for i in range(self.num_shards) if i not in completed_shards]
            else:
                pending_shards = list(range(self.num_shards))

            self.shards_by_worker = [
                pending_shards[i::self.num_gpus] for i in range(self.num_gpus)
            ]

            for i, shard_list in enumerate(self.shards_by_worker):
                self.logger.info(f"Worker {i} assigned {len(shard_list)} shards")

        # Start unified workers
        self.workers = []
        for i in range(self.num_gpus):
            worker_cpus = self.dataloader_workers + 2 if self.dataloader_workers > 0 else 1
            worker = Worker.options(num_cpus=worker_cpus).remote(
                tokenizer_path=self.tokenizer_path,
                output_dir=self.output_dir,
                worker_id=i,
                mode=self.mode,
                audio_field=self.audio_field,
                text_field=self.text_field,
                min_duration=self.min_duration,
                max_duration=self.max_duration,
                batch_size=self.batch_size,
                dataloader_workers=self.dataloader_workers,
                dataloader_prefetch_factor=self.dataloader_prefetch_factor,
                dataloader_persistent_workers=self.dataloader_persistent_workers,
                target_sample_rate=self.target_sample_rate,
                target_bucket=getattr(self, 'target_bucket', None),
                wandb_logger=self._wandb_logger if self.wandb_live else None,
                wandb_log_interval_seconds=self.wandb_log_interval if self.wandb_live else None,
            )
            self.workers.append(worker)

        self.logger.info(f"Setup {self.num_gpus} workers for {self.mode} mode (batch_size={self.batch_size})")

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
        total_processed = ray.get(self.progress_actor.close.remote())

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
            }))

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
                "sampling_rate": self.target_sample_rate,
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
        """Cleanup Ray and W&B resources."""
        # Finish W&B run
        if self.wandb_enabled:
            if self.wandb_live and self._wandb_logger is not None:
                ray.get(self._wandb_logger.finish.remote())
                self._wandb_logger = None
            elif self._wandb_run is not None:
                import wandb

                wandb.finish()
                self._wandb_run = None

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
