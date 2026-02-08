#!/usr/bin/env python3
"""Main entry point for audio tokenization pipeline.

Usage:
    python -m audio_tokenization.tokenize
    python -m audio_tokenization.tokenize num_gpus=8
    python -m audio_tokenization.tokenize dataset=peoples_speech_wds
    python -m audio_tokenization.tokenize dataset=audioset_lhotse
"""

# Avoid thread oversubscription with many dataloader workers
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Print config only on rank 0
    if int(os.environ.get("RANK", 0)) == 0:
        logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    tokenizer_cfg = cfg.get("tokenizer", {})
    tokenizer_path = tokenizer_cfg.get("path") if tokenizer_cfg else None
    tokenizer_sampling_rate = tokenizer_cfg.get("sampling_rate") if tokenizer_cfg else None
    torch_compile = tokenizer_cfg.get("torch_compile", True) if tokenizer_cfg else True
    trim_last_tokens = tokenizer_cfg.get("trim_last_tokens", 5) if tokenizer_cfg else 5
    if tokenizer_path is None:
        tokenizer_path = cfg.get("tokenizer_path")

    dataset_type = cfg.dataset.get("dataset_type", "lhotse")
    decode_workers_per_gpu = cfg.get("decode_workers_per_gpu", 0)
    dataloader_prefetch_factor = cfg.get("dataloader_prefetch_factor", 2)
    min_sample_rate = cfg.get("min_sample_rate")

    if dataset_type == "wds":
        from audio_tokenization.pipelines.wds import WDSDatasetPipeline as Pipeline

        pipeline = Pipeline(
            tokenizer_path=tokenizer_path,
            target_sample_rate=tokenizer_sampling_rate,
            torch_compile=torch_compile,
            trim_last_tokens=trim_last_tokens,
            output_dir=cfg.output_dir,
            dataset_name=cfg.dataset.dataset_name,
            dataset_split=cfg.dataset.get("dataset_split", "train"),
            shards=cfg.dataset.get("shards", []),
            audio_extensions=cfg.dataset.get("audio_extensions", []),
            mode=cfg.get("mode", "audio_only"),
            num_gpus=cfg.num_gpus,
            device=cfg.get("device", "cuda"),
            shard_assignment=cfg.get("shard_assignment", "shared"),
            num_shards=0,
            buffer_size=cfg.dataset.get("buffer_size"),
            min_duration=cfg.get("min_duration"),
            max_duration=cfg.get("max_duration"),
            min_sample_rate=min_sample_rate,
            max_samples=cfg.dataset.get("max_samples"),
            target_bucket=cfg.dataset.get("target_bucket"),
            silence_unique_threshold=cfg.dataset.get("silence_unique_threshold"),
            decode_workers_per_gpu=decode_workers_per_gpu,
            dataloader_prefetch_factor=dataloader_prefetch_factor,
            resume=cfg.get("resume", False),
            batch_size=cfg.get("batch_size", 1),
            metadata_path=cfg.dataset.get("metadata_dir") or cfg.dataset.get("metadata_path"),
            wandb_config=OmegaConf.to_container(cfg.get("wandb", {}), resolve=True),
            ray_config=OmegaConf.to_container(cfg.get("ray", {}), resolve=True),
        )
    elif dataset_type == "lhotse":
        from audio_tokenization.pipelines.lhotse import run_lhotse_pipeline

        # Build flat config dict for the unified Lhotse pipeline (DDP, no Ray)
        pipeline_cfg = {
            "tokenizer_path": tokenizer_path,
            "target_sample_rate": tokenizer_sampling_rate,
            "torch_compile": torch_compile,
            "trim_last_tokens": trim_last_tokens,
            "output_dir": cfg.output_dir,
            "dataset_name": cfg.dataset.dataset_name,
            "dataset_split": cfg.dataset.get("dataset_split", "train"),
            "mode": cfg.get("mode", "audio_only"),
            "resume": cfg.get("resume", False),
            "min_duration": cfg.dataset.get("min_duration", cfg.get("min_duration")),
            "max_duration": cfg.dataset.get("max_duration", cfg.get("max_duration")),
            "min_sample_rate": cfg.dataset.get("min_sample_rate", min_sample_rate),
            # Shar data (pre-built by prepare_hf_to_shar / prepare_wds_to_shar)
            "shar_dir": cfg.dataset.get("shar_dir"),
            "shar_index_filename": cfg.dataset.get("shar_index_filename", "shar_index.json"),
            # Dynamic bucketing sampler
            "max_batch_duration": cfg.dataset.get("max_batch_duration", 1500.0),
            "max_batch_cuts": cfg.dataset.get("max_batch_cuts"),
            "num_buckets": cfg.dataset.get("num_buckets", 20),
            "bucket_buffer_size": cfg.dataset.get("bucket_buffer_size", 20000),
            "sampler_shuffle": cfg.dataset.get("sampler_shuffle", True),
            "sampler_seed": cfg.dataset.get("sampler_seed", 42),
            "quadratic_duration": cfg.dataset.get("quadratic_duration"),
            # DataLoader prefetching
            "num_workers": cfg.dataset.get("num_workers", 4),
            "prefetch_factor": cfg.dataset.get("prefetch_factor", 4),
            # Checkpointing
            "checkpoint_interval_batches": cfg.dataset.get("checkpoint_interval_batches", 500),
            # Filters
            "silence_unique_threshold": cfg.dataset.get("silence_unique_threshold"),
            # W&B
            "wandb": OmegaConf.to_container(cfg.get("wandb", {}), resolve=True),
        }

        result = run_lhotse_pipeline(pipeline_cfg)

        logger.info("Pipeline completed!")
        logger.info(f"Total processed: {result.get('samples_processed', 0)}")
        logger.info(f"Total tokens: {result.get('tokens_generated', 0)}")
        logger.info(f"Output directory: {result.get('output_dir', cfg.output_dir)}")

        return result
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}. Expected one of: wds, lhotse")

    result = pipeline.run()

    logger.info("Pipeline completed!")
    logger.info(f"Total processed: {result.get('total_samples', result.get('total_processed', 0))}")
    logger.info(f"Total tokens: {result.get('total_tokens', 0)}")
    logger.info(f"Output directory: {result.get('output_dir', '')}")

    return result


if __name__ == "__main__":
    main()
