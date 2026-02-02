#!/usr/bin/env python3
"""
Core utilities for creating audio omni-tokenizers.

Shared functions for adding audio tokens to text tokenizers.
Follows the same pattern as vision_tokenization.
"""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

from transformers import AutoTokenizer


# All known RESERVED_OMNI renames across modalities
# This allows any modality tokenizer to detect slots already used by other modalities
RESERVED_TOKEN_RENAMES = {
    # Vision structure tokens (001-007)
    1: "<|img_start|>",
    2: "<|img_end|>",
    3: "<|img_token_start|>",
    4: "<|img_end_of_row|>",
    5: "<|img_end_of_frame|>",
    6: "<|img_generation_start|>",
    7: "<|image|>",
    # Audio structure tokens (008-009)
    8: "<|audio_start|>",
    9: "<|audio_end|>",
}

# Audio-specific renames (subset of RESERVED_TOKEN_RENAMES)
AUDIO_STRUCTURE_TOKEN_RENAMES = [
    ("<|RESERVED_OMNI_008|>", "<|audio_start|>"),
    ("<|RESERVED_OMNI_009|>", "<|audio_end|>"),
]

# Default number of reserved tokens for all modalities
DEFAULT_NUM_RESERVED_TOKENS = 200


def _copy_modality_mapping_files(
    existing_modalities: Dict[str, Any],
    input_tokenizer_path: str,
    output_path: str,
    *,
    announce: bool,
) -> None:
    modalities = existing_modalities.get("modalities") or {}
    if not modalities:
        return
    if announce:
        print("\nCopying existing modality mapping files...")
    for info in modalities.values():
        src = os.path.join(input_tokenizer_path, info["mapping_file"])
        dst = os.path.join(output_path, info["mapping_file"])
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy(src, dst)
            print(f"  Copied {info['mapping_file']}")


def _update_tokenizer_config(
    config_path: str,
    stats: Dict[str, Any],
    base_vocab_size: int,
    audio_vocab_size: int,
    audio_tokenizer_name: str,
) -> None:
    if not os.path.exists(config_path):
        return
    with open(config_path, "r") as f:
        config = json.load(f)
    config["vocab_size"] = stats["final_vocab_size"]
    # Only set base_vocab_size if not already present (preserve existing value)
    if "base_vocab_size" not in config:
        config["base_vocab_size"] = base_vocab_size
    config["audio_tokenizer"] = {
        "type": audio_tokenizer_name,
        "codebook_size": audio_vocab_size,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def _collect_reserved_tokens(existing_vocab: Dict[str, Any], num_reserved_tokens: int) -> list:
    reserved_tokens = []
    for i in range(num_reserved_tokens):
        original = f"<|RESERVED_OMNI_{i:03d}|>"
        renamed = RESERVED_TOKEN_RENAMES.get(i)

        # Skip if slot is already used (original or renamed exists in vocab)
        if original in existing_vocab:
            continue
        if renamed and renamed in existing_vocab:
            continue

        reserved_tokens.append(original)
    return reserved_tokens


def _collect_audio_tokens(existing_vocab: Dict[str, Any], audio_vocab_size: int) -> list:
    audio_tokens = []
    for i in range(audio_vocab_size):
        token = f"<|audio token {i}|>"
        if token not in existing_vocab:
            audio_tokens.append(token)
    return audio_tokens


def detect_existing_modalities(tokenizer_path: str) -> Dict[str, Any]:
    """
    Detect existing modalities in a tokenizer.

    Checks for vision_token_mapping.json, audio_token_mapping.json, and
    base_vocab_size in tokenizer_config.json to understand what modalities
    already exist.

    Args:
        tokenizer_path: Path to the tokenizer directory

    Returns:
        Dict with:
            - base_vocab_size: int or None (true text-only vocab)
            - modalities: dict of modality name -> info dict
    """
    result: Dict[str, Any] = {"base_vocab_size": None, "modalities": {}}

    # Check tokenizer_config.json for base_vocab_size
    config_path = os.path.join(tokenizer_path, "tokenizer_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        result["base_vocab_size"] = config.get("base_vocab_size")

    # Check for vision modality
    vision_mapping_path = os.path.join(tokenizer_path, "vision_token_mapping.json")
    if os.path.exists(vision_mapping_path):
        with open(vision_mapping_path, "r", encoding="utf-8") as f:
            vision_data = json.load(f)
        result["modalities"]["vision"] = {
            "mapping_file": "vision_token_mapping.json",
            "vocab_size": vision_data.get("visual_vocab_size"),
            "token_format": vision_data.get("vision_token_format"),
        }

    # Check for audio modality (for idempotency)
    audio_mapping_path = os.path.join(tokenizer_path, "audio_token_mapping.json")
    if os.path.exists(audio_mapping_path):
        with open(audio_mapping_path, "r", encoding="utf-8") as f:
            audio_data = json.load(f)
        result["modalities"]["audio"] = {
            "mapping_file": "audio_token_mapping.json",
            "vocab_size": audio_data.get("audio_vocab_size"),
            "tokenizer_type": audio_data.get("audio_tokenizer"),
            "token_format": audio_data.get("audio_token_format"),
        }

    return result


def rename_reserved_token(save_path: str, tokenizer, old_token: str, new_token: str) -> None:
    """
    Rename a reserved token in saved tokenizer files.

    Args:
        save_path: Path where tokenizer was saved
        tokenizer: Tokenizer instance to get token ID
        old_token: Old token name (e.g., "<|RESERVED_OMNI_001|>")
        new_token: New token name (e.g., "<|audio_start|>")
    """
    token_id = tokenizer.convert_tokens_to_ids(old_token)
    if token_id == tokenizer.unk_token_id:
        print(f"  {old_token} not found, skipping rename to {new_token}")
        return

    # Modify tokenizer.json
    tokenizer_json_path = os.path.join(save_path, "tokenizer.json")
    if os.path.exists(tokenizer_json_path):
        with open(tokenizer_json_path, "r", encoding="utf-8") as f:
            content = f.read()

        content = content.replace(f'"{old_token}"', f'"{new_token}"')
        content = content.replace(old_token, new_token)

        with open(tokenizer_json_path, "w", encoding="utf-8") as f:
            f.write(content)

    # Also modify tokenizer_config.json
    config_path = os.path.join(save_path, "tokenizer_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        def replace_in_dict(obj):
            if isinstance(obj, dict):
                return {k: replace_in_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_in_dict(item) for item in obj]
            elif isinstance(obj, str):
                return obj.replace(old_token, new_token)
            return obj

        config = replace_in_dict(config)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    print(f"  Renamed {old_token} -> {new_token} (ID {token_id})")


def get_audio_vocab_size_from_tokenizer(audio_tokenizer_name: str = "wavtokenizer") -> int:
    """
    Get audio vocab size by querying the audio tokenizer class.

    Args:
        audio_tokenizer_name: Name of the audio tokenizer

    Returns:
        int: The codebook size of the audio tokenizer
    """
    # Add src directory to path
    repo_root = Path(__file__).parent.parent.parent.parent
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    name_lower = audio_tokenizer_name.lower()

    if name_lower in ("wavtokenizer", "wavtokenizer-40"):
        from audio_tokenizers.implementations.wavtokenizer import WavTokenizer40
        return WavTokenizer40.codebook_size.fget(None)  # Get property without instantiation
    elif name_lower == "wavtokenizer-75":
        from audio_tokenizers.implementations.wavtokenizer import WavTokenizer75
        return WavTokenizer75.codebook_size.fget(None)
    else:
        raise ValueError(f"Unknown audio tokenizer: {audio_tokenizer_name}")


def add_audio_tokens(
    input_tokenizer_path: str,
    output_path: str,
    audio_tokenizer_name: str = "wavtokenizer",
    num_reserved_tokens: int = DEFAULT_NUM_RESERVED_TOKENS,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Add audio tokens to a tokenizer.

    Audio vocab size is auto-detected from the audio tokenizer class.
    This function handles both text-only tokenizers and tokenizers with
    existing modalities (e.g., vision tokens).

    This function:
    1. Detects existing modalities (vision, etc.) via mapping files
    2. Adds RESERVED_OMNI tokens if not present
    3. Renames some to audio structure tokens (audio_start, audio_end)
    4. Adds audio content tokens: <|audio token 0|> through <|audio token N|>
    5. Copies existing modality mapping files (vision_token_mapping.json, etc.)
    6. Saves audio_token_mapping.json with all token mappings
    7. Updates tokenizer_config.json preserving base_vocab_size

    Args:
        input_tokenizer_path: Path to input tokenizer
        output_path: Path to save updated tokenizer
        audio_tokenizer_name: Audio tokenizer for vocab size auto-detection (default: "wavtokenizer")
        num_reserved_tokens: Number of RESERVED_OMNI tokens (default: 200)

    Returns:
        Tuple of (tokenizer object, stats dict)

    Files created:
        - tokenizer files (tokenizer.json, tokenizer_config.json, etc.)
        - audio_token_mapping.json (mapping from audio indices to token IDs)
        - copies of existing modality mapping files (e.g., vision_token_mapping.json)
    """
    # Auto-detect audio vocab size from tokenizer class
    audio_vocab_size = get_audio_vocab_size_from_tokenizer(audio_tokenizer_name)
    print(f"Auto-detected audio vocab size: {audio_vocab_size} (from {audio_tokenizer_name})\n")

    # Detect existing modalities in the input tokenizer
    existing_modalities = detect_existing_modalities(input_tokenizer_path)
    print("=" * 60)
    print("CREATING AUDIO OMNI-TOKENIZER")
    print("=" * 60)

    print(f"\nInput tokenizer: {input_tokenizer_path}")
    print(f"Output path: {output_path}")
    print(f"Audio vocab size: {audio_vocab_size:,}")
    print(f"Reserved tokens: {num_reserved_tokens}")

    if existing_modalities["modalities"]:
        print(f"Detected existing modalities: {list(existing_modalities['modalities'].keys())}")
    if existing_modalities["base_vocab_size"]:
        print(f"Detected base_vocab_size: {existing_modalities['base_vocab_size']:,}")

    # Load existing tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(input_tokenizer_path, use_fast=True)
    current_vocab_size = len(tokenizer)
    print(f"Current vocabulary size: {current_vocab_size:,}")

    # Determine base vocab size (true text-only vocab)
    # Use existing base_vocab_size if available, otherwise use current vocab size
    base_vocab_size = existing_modalities["base_vocab_size"] or current_vocab_size
    print(f"Base vocabulary size (text-only): {base_vocab_size:,}")

    # Statistics
    stats = {
        "input_tokenizer": input_tokenizer_path,
        "audio_tokenizer": audio_tokenizer_name,
        "original_vocab_size": current_vocab_size,
        "base_vocab_size": base_vocab_size,
        "reserved_tokens_added": 0,
        "structure_tokens_added": len(AUDIO_STRUCTURE_TOKEN_RENAMES),
        "audio_tokens_added": 0,
        "final_vocab_size": 0,
        "existing_modalities": list(existing_modalities["modalities"].keys()),
    }

    # Check if audio tokens already exist (idempotency)
    if "audio" in existing_modalities["modalities"]:
        existing_audio = existing_modalities["modalities"]["audio"]
        existing_vocab_size = existing_audio.get("vocab_size")
        existing_tokenizer = existing_audio.get("tokenizer_type")

        # Validate that requested parameters match existing
        if existing_vocab_size and existing_vocab_size != audio_vocab_size:
            raise ValueError(
                f"Audio tokens already exist with vocab_size={existing_vocab_size}, "
                f"but requested audio_vocab_size={audio_vocab_size}. "
                f"To use a different audio tokenizer, start from a tokenizer without audio tokens."
            )

        print(f"\nAudio tokens already exist in tokenizer (found audio_token_mapping.json)!")
        print(f"  Existing: {existing_tokenizer} (vocab_size={existing_vocab_size})")
        print("Skipping audio token addition...")

        # Just copy everything to output
        os.makedirs(output_path, exist_ok=True)
        tokenizer.save_pretrained(output_path)

        # Copy all modality mapping files (skip if in-place update)
        _copy_modality_mapping_files(
            existing_modalities,
            input_tokenizer_path,
            output_path,
            announce=False,
        )

        stats["final_vocab_size"] = current_vocab_size
        return tokenizer, stats

    # Collect all tokens to add
    tokens_to_add = []

    # Get existing vocabulary
    existing_vocab = tokenizer.get_vocab()

    # Add RESERVED_OMNI tokens (skip slots already used by any modality)
    print(f"\nAdding RESERVED_OMNI tokens (up to {num_reserved_tokens})...")
    reserved_tokens = _collect_reserved_tokens(existing_vocab, num_reserved_tokens)

    tokens_to_add.extend(reserved_tokens)
    stats["reserved_tokens_added"] = len(reserved_tokens)

    # Add audio content tokens (no padding for flexibility)
    print(f"\nGenerating {audio_vocab_size:,} audio content tokens...")
    audio_tokens = _collect_audio_tokens(existing_vocab, audio_vocab_size)

    tokens_to_add.extend(audio_tokens)
    stats["audio_tokens_added"] = len(audio_tokens)
    print(f"Token format: <|audio token N|>")
    print(f"Range: <|audio token 0|> to <|audio token {audio_vocab_size - 1}|>")

    # Add all tokens to vocabulary
    print(f"\nAdding {len(tokens_to_add):,} tokens to vocabulary...")
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
    print(f"Added {num_added:,} new tokens")

    stats["final_vocab_size"] = len(tokenizer)
    print(f"New vocabulary size: {stats['final_vocab_size']:,}")

    # Save tokenizer
    print(f"\nSaving tokenizer to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    tokenizer.save_pretrained(output_path)

    # Copy existing modality mapping files (e.g., vision_token_mapping.json)
    _copy_modality_mapping_files(
        existing_modalities,
        input_tokenizer_path,
        output_path,
        announce=True,
    )

    # Update tokenizer_config.json
    config_path = os.path.join(output_path, "tokenizer_config.json")
    _update_tokenizer_config(config_path, stats, base_vocab_size, audio_vocab_size, audio_tokenizer_name)

    # Rename RESERVED_OMNI tokens to audio structure tokens
    print(f"\nRenaming RESERVED_OMNI tokens to audio structure tokens...")
    for old_token, new_token in AUDIO_STRUCTURE_TOKEN_RENAMES:
        rename_reserved_token(output_path, tokenizer, old_token, new_token)

    # Create audio token mapping
    print("\nCreating audio token mapping...")
    audio_mapping = {}
    for i in range(audio_vocab_size):
        token = f"<|audio token {i}|>"
        token_id = tokenizer.convert_tokens_to_ids(token)
        audio_mapping[i] = token_id

    _save_audio_mapping(output_path, audio_mapping, audio_vocab_size, stats["final_vocab_size"], audio_tokenizer_name)

    # Verification
    print("\n" + "=" * 60)
    print("VERIFICATION - Sample Token IDs")
    print("=" * 60)

    print("\nBoundary marker:")
    boundary_id = tokenizer.convert_tokens_to_ids("<|RESERVED_OMNI_000|>")
    if boundary_id != tokenizer.unk_token_id:
        print(f"  <|RESERVED_OMNI_000|>: ID {boundary_id}")

    print("\nAudio structure tokens:")
    for _, new_token in AUDIO_STRUCTURE_TOKEN_RENAMES:
        token_id = tokenizer.convert_tokens_to_ids(new_token)
        if token_id != tokenizer.unk_token_id:
            print(f"  {new_token}: ID {token_id}")

    print("\nAudio content tokens (sample):")
    sample_indices = [0, audio_vocab_size // 2, audio_vocab_size - 1]
    for idx in sample_indices:
        token = f"<|audio token {idx}|>"
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"  {token}: ID {token_id}")

    print(f"\nAudio token offset: {audio_mapping[0]}")
    print("=" * 60)

    return tokenizer, stats


def _save_audio_mapping(output_path: str, audio_mapping: dict, audio_vocab_size: int, vocab_size: int, audio_tokenizer_name: str):
    """Save audio token mapping to JSON file."""
    mapping_data = {
        "audio_tokenizer": audio_tokenizer_name,
        "audio_vocab_size": audio_vocab_size,
        "audio_token_format": "<|audio token N|>",
        "audio_token_offset": audio_mapping[0],
        "vocab_size": vocab_size,
        "audio_token_ids": audio_mapping,
    }

    mapping_path = os.path.join(output_path, "audio_token_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(mapping_data, f, indent=2)
    print(f"Saved audio token mapping to {mapping_path}")
