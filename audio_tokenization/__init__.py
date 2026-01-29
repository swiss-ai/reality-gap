"""Audio Tokenization Pipeline for benchmark-audio-tokenizer.

This package provides a distributed tokenization pipeline for audio datasets,
similar to vision_tokenization but optimized for audio tokenizers.
"""

__version__ = "0.1.0"

def main():
    """Entry point for CLI."""
    from .tokenize import main as _main
    return _main()
