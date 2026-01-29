#!/usr/bin/env python3
"""
Add Audio Tokens to Text Tokenizer

Extends a base text tokenizer (e.g., LLaMA) by adding audio tokens,
creating an omni-tokenizer. Vocab size is auto-detected from the
audio tokenizer class.

Usage:
    python add_audio_tokens.py \
        --input-tokenizer swiss-ai/Apertus-8B-2509 \
        --output-tokenizer /capstor/store/cscs/swissai/infra01/MLLM/apertus_wavtokenizer \
        --audio-tokenizer-type wavtokenizer
"""

import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from .core import add_audio_tokens
except ImportError:
    # Fallback for direct execution
    from core import add_audio_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Add audio tokens to an existing omni-tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Add WavTokenizer audio tokens (vocab size auto-detected)
    python add_audio_tokens.py \\
        --input-tokenizer /path/to/llama3_emu3_tokenizer \\
        --output-tokenizer /path/to/llama3_emu3_wav_tokenizer \\
        --audio-tokenizer-type wavtokenizer

Supported audio tokenizers:
    - wavtokenizer (4096 codebook)
    - wavtokenizer-75 (4096 codebook)
        """
    )

    parser.add_argument(
        "--input-tokenizer",
        type=str,
        required=True,
        help="HuggingFace model ID or local path to base tokenizer (e.g., meta-llama/Llama-3-8B or /path/to/tokenizer)",
    )
    parser.add_argument(
        "--output-tokenizer",
        type=str,
        required=True,
        help="Path to save the new omni-tokenizer with audio tokens added",
    )
    parser.add_argument(
        "--audio-tokenizer-type",
        type=str,
        required=True,
        help="Audio tokenizer type for vocab size auto-detection (e.g., wavtokenizer)",
    )

    args = parser.parse_args()

    # Add audio tokens (vocab size auto-detected from tokenizer type)
    _, stats = add_audio_tokens(
        input_tokenizer_path=args.input_tokenizer,
        output_path=args.output_tokenizer,
        audio_tokenizer_name=args.audio_tokenizer_type,
    )

    print("\n" + "=" * 60)
    print("AUDIO TOKEN ADDITION SUMMARY")
    print("=" * 60)
    print(f"Input tokenizer:          {stats['input_tokenizer']}")
    print(f"Audio tokenizer:          {stats['audio_tokenizer']}")
    print(f"Original vocabulary size: {stats['original_vocab_size']:,}")
    print(f"Audio tokens added:       {stats['audio_tokens_added']:,}")
    print(f"Structure tokens:         {stats['structure_tokens_added']} (audio_start, audio_end)")
    print(f"Final vocabulary size:    {stats['final_vocab_size']:,}")
    print("=" * 60)
    print(f"\nOmni-tokenizer with audio tokens saved to: {args.output_tokenizer}")


if __name__ == "__main__":
    main()
