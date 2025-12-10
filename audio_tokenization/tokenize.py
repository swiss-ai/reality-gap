#!/usr/bin/env python3
"""
Main entry point for audio tokenization.

This script provides a unified interface for different tokenization pipelines.

Usage:
    # Using command line arguments
    python tokenize.py hf \
        --tokenizer-path $SCRATCH/test_audio_omni_tokenizer \
        --dataset-name librispeech_asr \
        --dataset-split train.100 \
        --mode audio_only \
        --output-dir $SCRATCH/test_output \
        --num-gpus 1 \
        --num-shards 1 \
        --device cuda \
        --audio-field audio \
        --max-samples 10

    # Resume from previous checkpoint (skips completed shards)
    python tokenize.py hf --config path/to/config.json --resume

    # View all available options
    python tokenize.py hf --help
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from audio_tokenization.pipelines.hf.pipeline import HFDatasetPipeline


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def create_hf_parser(subparsers):
    """Create parser for HuggingFace datasets."""
    parser = subparsers.add_parser(
        'hf',
        help='Tokenize HuggingFace audio datasets'
    )

    # Common arguments (also available in subparser)
    parser.add_argument(
        '--tokenizer-path',
        type=str,
        required=False,
        help='Path to the audio omni-tokenizer'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=False,
        help='Output directory for tokenized data'
    )
    parser.add_argument(
        '--num-gpus',
        type=int,
        required=False,
        help='Number of GPUs for tokenization'
    )
    parser.add_argument(
        '--device',
        type=str,
        required=False,
        choices=['cuda', 'cpu'],
        help='Device for tokenization'
    )

    # HF-specific arguments
    parser.add_argument(
        '--config',
        type=str,
        help='Path to JSON configuration file'
    )
    parser.add_argument(
        '--dataset-name',
        type=str,
        required=False,
        help='HuggingFace dataset name'
    )
    parser.add_argument(
        '--dataset-split',
        type=str,
        required=False,
        help='Dataset split to process'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['audio_only'],
        required=False,
        default='audio_only',
        help='Tokenization mode (currently only audio_only supported)'
    )
    parser.add_argument(
        '--config-name',
        type=str,
        help='Dataset configuration/subset name'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        help='Cache directory for downloaded datasets'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        help='Pfad zu lokal standardisierten Daten (HF Dataset-Verzeichnis oder .parquet)'
    )
    parser.add_argument(
        '--dataset-format',
        type=str,
        choices=['auto', 'hf', 'parquet'],
        default='auto',
        help='Format der lokalen Daten (auto-detect, HF Dataset oder Parquet)'
    )
    parser.add_argument(
        '--num-proc',
        type=int,
        help='Number of processes for dataset loading'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        help='Maximum number of samples to process'
    )
    parser.add_argument(
        '--num-shards',
        type=int,
        help='Number of shards for distributed processing and checkpointing (required)'
    )
    parser.add_argument(
        '--audio-field',
        type=str,
        default='audio',
        help='Name of audio field in dataset'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing checkpoint by skipping completed shards'
    )

    return parser


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Audio Tokenization Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Global arguments (only verbose here, common args are in subparser)
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    # Subparsers for different data formats
    subparsers = parser.add_subparsers(
        dest='data_format',
        help='Data format to process',
        required=True
    )

    # Add format-specific parsers
    create_hf_parser(subparsers)

    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Load config file or use CLI args
    if args.config:
        logger.info(f"Loading config from {args.config}")
        with open(args.config) as f:
            config = json.load(f)

        logger.info(f"Config loaded: {list(config.keys())}")

        # CLI args override config file
        cli_overrides = {k: v for k, v in vars(args).items() if v is not None}
        logger.info(f"CLI overrides: {list(cli_overrides.keys())}")
        config.update(cli_overrides)
        logger.info(f"Final config keys: {list(config.keys())}")
    else:
        config = vars(args)

    # Extract data format and remove non-pipeline keys
    data_format = args.data_format
    for key in ['config', 'verbose', 'data_format']:
        config.pop(key, None)

    # Validate required parameters
    required = ['tokenizer_path', 'output_dir', 'num_gpus', 'device']
    if data_format == 'hf':
        if config.get('dataset_path'):
            required += ['mode', 'num_shards', 'dataset_path']
        else:
            required += ['dataset_name', 'dataset_split', 'mode', 'num_shards']

    missing = [k for k in required if not config.get(k)]
    if missing:
        logger.error(f"Missing required parameters: {', '.join(missing)}")
        logger.error("Provide them via config file or command line arguments")
        sys.exit(1)

    try:
        if data_format == 'hf':
            logger.info("Running HuggingFace dataset pipeline")
            # Remove None values and pass to pipeline
            pipeline_config = {k: v for k, v in config.items() if v is not None}
            pipeline = HFDatasetPipeline(**pipeline_config)
            result = pipeline.run()

        else:
            raise ValueError(f"Unknown data format: {data_format}")

        # Report results
        logger.info("Pipeline completed successfully")
        logger.info(f"Total samples processed: {result['total_samples']}")
        logger.info(f"Total tokens generated: {result['total_tokens']}")
        logger.info(f"Processing time: {result['processing_time']:.2f}s")
        logger.info(f"Output directory: {result['output_dir']}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

