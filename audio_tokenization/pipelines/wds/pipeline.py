#!/usr/bin/env python3
"""
WebDataset (tar shard) tokenization pipeline.
"""

import json
import re
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import ray

from audio_tokenization.pipelines.base import BasePipeline, ProgressActor
from audio_tokenization.utils.ray_utils import init_ray
from audio_tokenization.pipelines.hf.workers.shard_assignment import ShardQueue
from .workers import WDSWorker


class WDSDatasetPipeline(BasePipeline):
    """Pipeline for tokenizing WebDataset tar shards."""

    def __init__(
        self,
        tokenizer_path: str,
        output_dir: str,
        dataset_name: str,
        dataset_split: str,
        shards: Sequence[str],
        audio_extensions: Sequence[str],
        mode: str,
        num_gpus: int,
        device: str,
        shard_assignment: str = "shared",
        num_shards: Optional[int] = None,
        buffer_size: Optional[int] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        max_samples: Optional[int] = None,
        target_sample_rate: Optional[int] = None,
        min_sample_rate: Optional[int] = None,
        target_bucket: Optional[int] = None,
        silence_unique_threshold: Optional[int] = None,
        torch_compile: bool = True,
        decode_workers_per_gpu: int = 0,
        dataloader_prefetch_factor: int = 2,
        resume: bool = False,
        batch_size: int = 1,
        metadata_path: Optional[str] = None,
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

        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.shards = list(shards or [])
        self.audio_extensions = list(audio_extensions or [])
        self.mode = mode
        self.shard_assignment = shard_assignment
        self.num_shards = num_shards or 0
        self.buffer_size = buffer_size
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.max_samples = max_samples
        self.target_bucket = target_bucket
        self.silence_unique_threshold = silence_unique_threshold
        self.torch_compile = torch_compile
        self.min_sample_rate = min_sample_rate
        self.decode_workers_per_gpu = int(decode_workers_per_gpu)
        self.dataloader_prefetch_factor = int(dataloader_prefetch_factor)
        self.resume = resume
        self.batch_size = batch_size
        self.metadata_path = metadata_path

        self.ray_config = ray_config or {}

        self.wandb_config = wandb_config or {}
        self.wandb_enabled = self.wandb_config.get("enabled", False)
        self.wandb_live = self.wandb_config.get("live", False)
        log_interval = self.wandb_config.get("log_interval_seconds")
        self.wandb_log_interval = 10 if log_interval is None else log_interval
        self._wandb_run = None
        self._wandb_logger = None

        # Resolved shard paths
        self.shard_paths: List[str] = []
        self.total_samples = 0

        # Validate shard assignment mode
        if self.shard_assignment not in {"shared", "static"}:
            raise ValueError("shard_assignment must be 'shared' or 'static'")

    def _sanitize_component(self, value: str) -> str:
        return re.sub(r"[^\w.-]+", "-", value).strip("-")

    def _format_duration(self, value: Optional[float]) -> Optional[str]:
        if value is None:
            return None
        if float(value).is_integer():
            return str(int(value))
        return str(value).replace(".", "p")

    def _build_output_subdir(self, extra_suffix: Optional[str] = None) -> str:
        dataset_label = self._sanitize_component(self.dataset_name)
        split_label = self._sanitize_component(self.dataset_split)
        parts = [dataset_label, split_label, "wds", self.mode]
        if self.min_duration is not None or self.max_duration is not None:
            min_dur = self._format_duration(self.min_duration) or "min"
            max_dur = self._format_duration(self.max_duration) or "max"
            parts.append(f"dur{min_dur}-{max_dur}")
        if extra_suffix:
            parts.append(self._sanitize_component(extra_suffix))
        return "_".join([p for p in parts if p])

    def _resolve_shards(self) -> List[str]:
        shard_paths: List[str] = []
        for pattern in self.shards:
            expanded = glob(pattern)
            if expanded:
                shard_paths.extend(sorted(expanded))
            else:
                shard_paths.append(pattern)
        # Deduplicate while preserving order
        seen = set()
        unique_paths = []
        for path in shard_paths:
            if path in seen:
                continue
            seen.add(path)
            unique_paths.append(path)
        return unique_paths

    def _get_completed_shards(self) -> set:
        completed = set()
        output_path = Path(self.output_dir)
        if not output_path.exists():
            return completed
        pattern = re.compile(r"rank_\d+_shard_(\d+)_(\d+)\.idx")
        shard_counts_found = set()
        files_by_shard_count = {}
        for idx_file in output_path.glob("*.idx"):
            match = pattern.match(idx_file.name)
            if match:
                shard_id = int(match.group(1))
                total_shards = int(match.group(2))
                shard_counts_found.add(total_shards)
                files_by_shard_count.setdefault(total_shards, []).append(idx_file.name)
                if total_shards == self.num_shards:
                    completed.add(shard_id)

        if shard_counts_found and self.num_shards not in shard_counts_found:
            self.logger.error(
                f"ERROR: No existing shards match expected count ({self.num_shards}). "
                f"Found shard counts: {sorted(shard_counts_found)}"
            )
            for count in sorted(shard_counts_found):
                self.logger.error(f"  {count} total shards: {len(files_by_shard_count[count])} files")
            self.logger.error("Clean the output directory or use a different output path")
            raise RuntimeError("Shard count mismatch")

        if len(shard_counts_found) > 1:
            self.logger.error(f"ERROR: Inconsistent total shard counts found: {sorted(shard_counts_found)}")
            for count in sorted(shard_counts_found):
                self.logger.error(f"  {count} total shards: {len(files_by_shard_count[count])} files")
            raise RuntimeError("Inconsistent shard counts")

        return completed

    def setup(self):
        self.logger.info(f"Initializing Ray with {self.num_gpus} workers")
        init_ray(self.ray_config, self.num_gpus)

        # Resolve shard list
        self.shard_paths = self._resolve_shards()
        if not self.shard_paths:
            raise ValueError("No shards found for WDS pipeline")

        self.num_shards = len(self.shard_paths)
        self.logger.info(f"Resolved {self.num_shards} shards")
        if self.max_samples:
            self.logger.warning("max_samples is not enforced for WDS pipelines (streaming shards).")

        # Auto-create output subdirectory
        self.output_dir = str(Path(self.output_dir) / self._build_output_subdir())
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize W&B if enabled
        if self.wandb_enabled:
            wandb_name = self.wandb_config.get("name")
            if wandb_name is None:
                wandb_name = self._build_output_subdir()

            wandb_run_config = {
                "dataset_name": self.dataset_name,
                "dataset_split": self.dataset_split,
                "mode": self.mode,
                "num_gpus": self.num_gpus,
                "num_shards": self.num_shards,
                "shard_assignment": self.shard_assignment,
                "batch_size": self.batch_size,
                "buffer_size": self.buffer_size,
                "target_sample_rate": self.target_sample_rate,
                "min_sample_rate": self.min_sample_rate,
                "decode_workers_per_gpu": self.decode_workers_per_gpu,
                "dataloader_prefetch_factor": self.dataloader_prefetch_factor,
                "min_duration": self.min_duration,
                "max_duration": self.max_duration,
                "target_bucket": self.target_bucket,
                "silence_unique_threshold": self.silence_unique_threshold,
                "tokenizer_path": self.tokenizer_path,
                "ray_address": self.ray_config.get("address"),
            }

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

        self._setup_workers()

    def _setup_workers(self):
        self.progress_actor = ProgressActor.remote(
            total_samples=self.total_samples,
            total_shards=self.num_shards,
            wandb_enabled=self.wandb_enabled,
            desc="WDS samples processed",
        )

        self.work_queue = None
        self.shards_by_worker = None
        if self.shard_assignment == "shared":
            if self.resume:
                completed = self._get_completed_shards()
                if completed:
                    pending = [i for i in range(self.num_shards) if i not in completed]
                    self.logger.info(f"Resume mode: {len(pending)} shards remaining")
                    self.work_queue = ShardQueue.remote(self.num_shards, initial_shards=pending)
                else:
                    self.work_queue = ShardQueue.remote(self.num_shards)
            else:
                self.work_queue = ShardQueue.remote(self.num_shards)
        else:
            if self.resume:
                completed = self._get_completed_shards()
                pending = [i for i in range(self.num_shards) if i not in completed]
            else:
                pending = list(range(self.num_shards))
            self.shards_by_worker = [pending[i::self.num_gpus] for i in range(self.num_gpus)]

        # Start workers
        self.workers = []
        for i in range(self.num_gpus):
            worker_cpus = self.decode_workers_per_gpu + 2 if self.decode_workers_per_gpu > 0 else 1
            worker = WDSWorker.options(num_cpus=worker_cpus).remote(
                tokenizer_path=self.tokenizer_path,
                output_dir=self.output_dir,
                worker_id=i,
                mode=self.mode,
                audio_extensions=self.audio_extensions,
                batch_size=self.batch_size,
                buffer_size=self.buffer_size,
                min_duration=self.min_duration,
                max_duration=self.max_duration,
                target_sample_rate=self.target_sample_rate,
                min_sample_rate=self.min_sample_rate,
                target_bucket=self.target_bucket,
                silence_unique_threshold=self.silence_unique_threshold,
                torch_compile=self.torch_compile,
                decode_workers_per_gpu=self.decode_workers_per_gpu,
                dataloader_prefetch_factor=self.dataloader_prefetch_factor,
                metadata_path=self.metadata_path,
                wandb_logger=self._wandb_logger if self.wandb_live else None,
                wandb_log_interval_seconds=self.wandb_log_interval if self.wandb_live else None,
            )
            self.workers.append(worker)

    def process(self) -> Dict[str, Any]:
        worker_futures = []
        if self.shard_assignment == "shared":
            worker_futures = [
                worker.run_shards.remote(self.work_queue, self.shard_paths, self.num_shards, self.progress_actor)
                for worker in self.workers
            ]
        else:
            for i, worker in enumerate(self.workers):
                shard_ids = self.shards_by_worker[i] if self.shards_by_worker else []
                worker_futures.append(
                    worker.run_shards.remote(shard_ids, self.shard_paths, self.num_shards, self.progress_actor)
                )

        results = ray.get(worker_futures)
        ray.get(self.progress_actor.close.remote())

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

        if self.wandb_enabled and self.wandb_live and self._wandb_logger is not None:
            ray.get(self._wandb_logger.log_final.remote({
                "total_samples_processed": total_samples_processed,
                "total_tokens": total_tokens,
                "total_errors": total_errors,
                "processing_time_seconds": max_time,
                "samples_per_second": total_samples_processed / max_time if max_time > 0 else 0,
                "tokens_per_second": total_tokens / max_time if max_time > 0 else 0,
            }))

        self._save_metadata(results, max_time)

        return {
            "total_samples": total_samples_processed,
            "total_tokens": total_tokens,
            "total_errors": total_errors,
            "processing_time": max_time,
            "workers": len(self.workers),
            "mode": self.mode,
            "output_dir": self.output_dir,
            "metadata_file": str(Path(self.output_dir) / "dataset_info.json"),
        }

    def _save_metadata(self, results: list, processing_time: float):
        metadata = self.generate_metadata(
            results=results,
            processing_time=processing_time,
            dataset_type=f"{self.mode} tokenization (wds)",
            dataset_name=self.dataset_name,
            split=self.dataset_split,
            mode=self.mode,
            tokenizer={
                "path": self.tokenizer_path,
                "sampling_rate": self.target_sample_rate,
            },
            audio_filtering={
                "min_duration": self.min_duration,
                "max_duration": self.max_duration,
                "target_bucket": self.target_bucket,
                "silence_unique_threshold": self.silence_unique_threshold,
                "min_sample_rate": self.min_sample_rate,
            },
            wds={
                "shards": self.shards,
                "resolved_shards": self.num_shards,
                "audio_extensions": self.audio_extensions,
                "buffer_size": self.buffer_size,
                "decode_workers_per_gpu": self.decode_workers_per_gpu,
                "dataloader_prefetch_factor": self.dataloader_prefetch_factor,
                "metadata_path": self.metadata_path,
            },
        )

        metadata_path = Path(self.output_dir) / "dataset_info.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Saved metadata to {metadata_path}")

    def cleanup(self):
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


def run_wds_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    pipeline = WDSDatasetPipeline(**config)
    return pipeline.run()
