#!/usr/bin/env python3
"""Main entry point for audio tokenization pipeline.

Usage:
    python -m audio_tokenization.tokenize
    python -m audio_tokenization.tokenize num_gpus=8
    python -m audio_tokenization.tokenize dataset=librispeech dataset.config_name=clean
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
    # Print config
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    tokenizer_cfg = cfg.get("tokenizer", {})
    tokenizer_path = tokenizer_cfg.get("path") if tokenizer_cfg else None
    tokenizer_sampling_rate = tokenizer_cfg.get("sampling_rate") if tokenizer_cfg else None
    torch_compile = tokenizer_cfg.get("torch_compile", True) if tokenizer_cfg else True
    if tokenizer_path is None:
        tokenizer_path = cfg.get("tokenizer_path")

    dataset_type = cfg.dataset.get("dataset_type", "hf")
    decode_workers_per_gpu = cfg.get("decode_workers_per_gpu", 0)
    dataloader_prefetch_factor = cfg.get("dataloader_prefetch_factor", 2)
    min_sample_rate = cfg.get("min_sample_rate")

    if dataset_type == "wds":
        from audio_tokenization.pipelines.wds import WDSDatasetPipeline as Pipeline

        pipeline = Pipeline(
            tokenizer_path=tokenizer_path,
            target_sample_rate=tokenizer_sampling_rate,
            torch_compile=torch_compile,
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
    else:
        # Compute num_shards from num_gpus * shards_per_gpu
        num_shards = cfg.num_gpus * cfg.get("shards_per_gpu", 1)
        logger.info(
            f"Sharding: {cfg.num_gpus} GPUs × {cfg.get('shards_per_gpu', 1)} shards/GPU = {num_shards} total shards"
        )

        # Common pipeline arguments
        common_args = dict(
            tokenizer_path=tokenizer_path,
            target_sample_rate=tokenizer_sampling_rate,
            torch_compile=torch_compile,
            output_dir=cfg.output_dir,
            dataset_name=cfg.dataset.dataset_name,
            dataset_split=cfg.dataset.get("dataset_split", "train"),
            mode=cfg.get("mode", "audio_only"),
            num_gpus=cfg.num_gpus,
            device=cfg.get("device", "cuda"),
            num_shards=num_shards,
            shard_assignment=cfg.get("shard_assignment", "shared"),
            config_name=cfg.dataset.get("config_name"),
            cache_dir=cfg.dataset.get("cache_dir"),
            decode_workers_per_gpu=decode_workers_per_gpu,
            dataloader_prefetch_factor=dataloader_prefetch_factor,
            min_duration=cfg.get("min_duration"),
            max_duration=cfg.get("max_duration"),
            min_sample_rate=min_sample_rate,
            max_samples=cfg.dataset.get("max_samples"),
            audio_field=cfg.dataset.get("audio_field", "audio"),
            text_field=cfg.dataset.get("text_field", "text"),
            resume=cfg.get("resume", False),
            batch_size=cfg.get("batch_size", 1),
            wandb_config=OmegaConf.to_container(cfg.get("wandb", {}), resolve=True),
            ray_config=OmegaConf.to_container(cfg.get("ray", {}), resolve=True),
        )

        # Check if bucket filtering is enabled
        bucket_cfg = cfg.dataset.get("bucket", {})
        if bucket_cfg.get("enabled", False):
            # Use bucketed pipeline with pre-filtering (single bucket only)
            from audio_tokenization.pipelines.hf import BucketedHFDatasetPipeline as Pipeline

            target_bucket = bucket_cfg.target_bucket
            shuffle_seed = bucket_cfg.get("shuffle_seed", 42)
            logger.info(
                f"Using BucketedHFDatasetPipeline with target_bucket={target_bucket}, shuffle_seed={shuffle_seed}"
            )

            pipeline = Pipeline(
                bucket_metadata_dir=bucket_cfg.metadata_dir,
                target_bucket=target_bucket,
                shuffle_seed=shuffle_seed,
                **common_args,
            )
        else:
            # Use standard pipeline
            from audio_tokenization.pipelines.hf import HFDatasetPipeline as Pipeline

            pipeline = Pipeline(**common_args)

    result = pipeline.run()

    logger.info("Pipeline completed!")
    logger.info(f"Total processed: {result.get('total_samples', result.get('total_processed', 0))}")
    logger.info(f"Total tokens: {result.get('total_tokens', 0)}")
    logger.info(f"Output directory: {result.get('output_dir', '')}")

    return result


if __name__ == "__main__":
    main()
