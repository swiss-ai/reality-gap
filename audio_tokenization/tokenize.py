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

    # Import here to avoid slow imports before config is loaded
    from audio_tokenization.pipelines.hf import HFDatasetPipeline

    pipeline = HFDatasetPipeline(
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
    )

    result = pipeline.run()

    logger.info("Pipeline completed!")
    logger.info(f"Total processed: {result.get('total_processed', 0)}")
    logger.info(f"Total tokens: {result.get('total_tokens', 0)}")
    logger.info(f"Output directory: {result.get('output_dir', '')}")

    return result


if __name__ == "__main__":
    main()
