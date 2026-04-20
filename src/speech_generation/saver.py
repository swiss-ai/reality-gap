"""Output saving abstraction for TTS-generated speech tokens.

Replace save_sample() with a Lhotse Shar writer later without
changing the generation pipeline.
"""

import json
from pathlib import Path

import torch
import torchaudio

from .base import TTSOutput


def save_sample(
    sample_id: str,
    output: TTSOutput,
    source_text: str,
    normalized_text: str,
    output_dir: Path,
    save_audio: bool = True,
) -> dict:
    """Save a single TTS output as {id}.pt + {id}.json (+ optional {id}.wav).

    Args:
        sample_id: Unique identifier for this sample.
        output: TTSOutput from a backend.
        source_text: Original input text before normalization.
        normalized_text: Text after normalization (what the TTS saw).
        output_dir: Directory to write files into.
        save_audio: Whether to also save the rendered WAV.

    Returns:
        Metadata dict for this sample.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save token tensor
    torch.save(output.speech_tokens.cpu(), output_dir / f"{sample_id}.pt")

    # Build metadata
    metadata = {
        "id": sample_id,
        "source_text": source_text,
        "normalized_text": normalized_text,
        "codebook_size": output.codebook_size,
        "token_rate_hz": output.token_rate_hz,
        "num_tokens": output.num_tokens,
        "duration_seconds": round(output.duration_seconds, 3),
        **output.metadata,
    }

    # Optional audio
    if save_audio and output.audio is not None:
        audio_path = output_dir / f"{sample_id}.wav"
        audio = output.audio.cpu()
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        torchaudio.save(str(audio_path), audio, output.audio_sample_rate)
        metadata["audio_file"] = f"{sample_id}.wav"

    # Save sidecar JSON
    with open(output_dir / f"{sample_id}.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata
