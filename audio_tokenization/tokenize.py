#!/usr/bin/env python3
"""Main entry point for audio tokenization pipeline.

Usage:
    python -m audio_tokenization.tokenize
    python -m audio_tokenization.tokenize num_gpus=8
    python -m audio_tokenization.tokenize dataset=librispeech dataset.config_name=clean
"""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Print config
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Common pipeline arguments
    common_args = dict(
        tokenizer_path=cfg.tokenizer_path,
        output_dir=cfg.output_dir,
        dataset_name=cfg.dataset.dataset_name,
        dataset_split=cfg.dataset.get("dataset_split", "train"),
        mode=cfg.get("mode", "audio_only"),
        num_gpus=cfg.num_gpus,
        device=cfg.get("device", "cuda"),
        num_shards=cfg.num_shards,
        config_name=cfg.dataset.get("config_name"),
        cache_dir=cfg.dataset.get("cache_dir"),
        num_proc=cfg.get("num_proc", 32),
        min_duration=cfg.get("min_duration"),
        max_duration=cfg.get("max_duration"),
        max_samples=cfg.dataset.get("max_samples"),
        audio_field=cfg.dataset.get("audio_field", "audio"),
        text_field=cfg.dataset.get("text_field", "text"),
        resume=cfg.get("resume", False),
        batch_size=cfg.get("batch_size", 1),
    )

    # Check if bucket filtering is enabled
    bucket_cfg = cfg.dataset.get("bucket", {})
    if bucket_cfg.get("enabled", False):
        # Use bucketed pipeline with pre-filtering
        from audio_tokenization.pipelines.hf import BucketedHFDatasetPipeline

        # Get target bucket(s) - support both singular and plural config keys
        target_buckets = bucket_cfg.get("target_bucket") or bucket_cfg.get("target_buckets")

        logger.info(f"Using BucketedHFDatasetPipeline with target_buckets={target_buckets}")

        pipeline = BucketedHFDatasetPipeline(
            bucket_metadata_dir=bucket_cfg.metadata_dir,
            target_buckets=target_buckets,
            **common_args,
        )
    else:
        # Use standard pipeline
        from audio_tokenization.pipelines.hf import HFDatasetPipeline

        pipeline = HFDatasetPipeline(**common_args)

    result = pipeline.run()

    logger.info("Pipeline completed!")
    logger.info(f"Total processed: {result.get('total_processed', 0)}")
    logger.info(f"Total tokens: {result.get('total_tokens', 0)}")
    logger.info(f"Output directory: {result.get('output_dir', '')}")

    return result


if __name__ == "__main__":
    main()
