"""WavTokenizer implementations."""

from .audio_only import WavTokenizerAudioOnly


def create_tokenizer(
    omni_tokenizer_path: str,
    mode: str = "audio_only",
    device: str = "cuda",
    **kwargs,
):
    """Factory function to create WavTokenizer.

    Args:
        omni_tokenizer_path: Path to omni_tokenizer (must have audio tokens added)
        mode: Tokenization mode (only "audio_only" supported)
        device: Device for tokenization

    Note:
        The audio token offset is automatically determined from the omni_tokenizer
        by querying: convert_tokens_to_ids("<|audio token 0|>")
    """
    if mode != "audio_only":
        raise ValueError(f"Only 'audio_only' mode is supported, got: {mode}")

    return WavTokenizerAudioOnly(
        omni_tokenizer_path=omni_tokenizer_path,
        device=device,
        **kwargs,
    )
