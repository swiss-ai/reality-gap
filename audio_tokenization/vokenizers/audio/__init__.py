"""
Audio tokenizers for audio-only sequences.
"""

from .audio_only import AudioOnlyTokenizer
from typing import Union


def create_tokenizer(
    mode: str,
    text_tokenizer_path: str,
    device: str = "cuda",
    **kwargs
) -> AudioOnlyTokenizer:
    """
    Factory function to create the appropriate audio tokenizer based on mode.

    Args:
        mode: Tokenization mode ("audio_only")
        text_tokenizer_path: Path to the text tokenizer (Omni-Tokenizer)
        device: Device for tokenization (cuda or cpu)
        **kwargs: Additional tokenizer-specific arguments (currently unused)

    Returns:
        The appropriate tokenizer instance based on mode

    Raises:
        ValueError: If mode is not recognized
    """
    if mode == "audio_only":
        return AudioOnlyTokenizer(
            text_tokenizer_path=text_tokenizer_path,
            device=device
        )
    else:
        raise ValueError(
            f"Unknown tokenizer mode: {mode}. "
            f"Must be one of: ['audio_only']"
        )


__all__ = [
    'AudioOnlyTokenizer',
    'create_tokenizer',
]

