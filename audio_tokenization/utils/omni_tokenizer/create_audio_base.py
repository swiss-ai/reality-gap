#!/usr/bin/env python3
"""
Create Base Audio Omni-Tokenizer

Creates a base omnimodal tokenizer by adding audio tokens to a text tokenizer.
Auto-detects codebook size from the audio tokenizer.

Usage:
    # For WavTokenizer (audio-tokenizer-path is optional)
    python create_audio_base.py \\
        --text-tokenizer-path meta-llama/Llama-3-8B \\
        --output-path ./llama3_audio_tokenizer
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directories to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.dirname(script_dir)
audio_tokenization_dir = os.path.dirname(utils_dir)
repo_root = os.path.dirname(audio_tokenization_dir)

# Add paths for imports
sys.path.insert(0, script_dir)  # For core.py
sys.path.insert(0, audio_tokenization_dir)  # For utils.omni_tokenizer
sys.path.insert(0, repo_root)  # For src imports

# Note: The local tokenizers/ directory has been renamed to legacy_tokenizers/
# to avoid conflicts with HuggingFace's tokenizers package.

# Import core function
from core import create_audio_base_tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Create base audio omni-tokenizer with auto-detected codebook size",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create with WavTokenizer (4096 audio tokens - auto-detected)
  python create_audio_base.py \\
      --text-tokenizer-path meta-llama/Llama-3-8B \\
      --output-path ./llama3_audio_omni
  
  # With explicit audio-tokenizer-path (optional, ignored for WavTokenizer)
  python create_audio_base.py \\
      --text-tokenizer-path meta-llama/Llama-3-8B \\
      --audio-tokenizer-path wavtokenizer \\
      --output-path ./llama3_audio_omni

    Available audio tokenizers:
    - WavTokenizer (4096 codebook)
        """
    )

    parser.add_argument(
        "--text-tokenizer-path",
        type=str,
        required=True,
        help="Path to base text tokenizer (e.g., meta-llama/Llama-3-8B)"
    )
    parser.add_argument(
        "--audio-tokenizer-path",
        type=str,
        default=None,
        help="Optional path to audio tokenizer (ignored for WavTokenizer, which auto-loads from HuggingFace. Used for config documentation.)"
    )
    parser.add_argument(
        "--audio-tokenizer",
        type=str,
        default="WavTokenizer",
        choices=["WavTokenizer"],
        help="Audio tokenizer type (default: WavTokenizer)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save omni-tokenizer"
    )
    parser.add_argument(
        "--num-reserved-tokens",
        type=int,
        default=200,
        help="Number of RESERVED_OMNI tokens to add (default: 200)"
    )

    args = parser.parse_args()

    # Create omni-tokenizer
    tokenizer, stats = create_audio_base_tokenizer(
        text_tokenizer_path=args.text_tokenizer_path,
        output_path=args.output_path,
        audio_tokenizer=args.audio_tokenizer,
        audio_tokenizer_path=args.audio_tokenizer_path,
        num_reserved_tokens=args.num_reserved_tokens
    )

    print("\n" + "="*60)
    print("AUDIO OMNI-TOKENIZER CREATION SUMMARY")
    print("="*60)
    print(f"Text tokenizer:           {stats['text_tokenizer']}")
    print(f"Audio tokenizer:          {stats['audio_tokenizer']}")
    print(f"Tokenizer type:           {stats['tokenizer_type']}")
    print(f"Original vocabulary size: {stats['original_vocab_size']:,}")
    print(f"Structure tokens added:   {stats['structure_tokens_added']:,}")
    print(f"Reserved tokens added:    {stats['reserved_tokens_added']:,}")
    print(f"Audio tokens added:       {stats['audio_tokens_added']:,}")
    print(f"Final vocabulary size:     {stats['final_vocab_size']:,}")
    print(f"Total tokens added:        {stats['final_vocab_size'] - stats['original_vocab_size']:,}")
    print("="*60)
    print("\n✅ Base audio omni-tokenizer created successfully!")
    print(f"   Saved to: {args.output_path}")


if __name__ == "__main__":
    main()

