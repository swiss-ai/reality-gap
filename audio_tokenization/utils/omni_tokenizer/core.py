#!/usr/bin/env python3
"""
Core utilities for creating audio omni-tokenizers.

Shared functions for adding audio tokens to text tokenizers.
Imports generic functions from vision_tokenization and adds audio-specific logic.
"""

import sys
import os
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# Ensure src/ is in sys.path for audio_tokenizers package imports
_current_dir = Path(__file__).resolve().parent
_audio_tokenization_dir = _current_dir.parent.parent
_repo_root = _audio_tokenization_dir.parent
_src_dir = _repo_root / "src"

if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Note: The local tokenizers/ directory has been renamed to legacy_tokenizers/
# to avoid conflicts with HuggingFace's tokenizers package.

# Import transformers lazily (only when needed) to avoid PyTorch/Transformers version conflicts
# The import happens inside create_audio_base_tokenizer() when AutoTokenizer is actually needed
# This allows the script to at least parse arguments before hitting version conflicts

# Copy generic utility functions directly to avoid PyTorch/Transformers version conflicts
# These are simple utility functions that don't depend on vision-specific code

def deduplicate_tokens(tokens_to_add, existing_vocab, verbose=True):
    """
    Remove duplicate tokens and filter out existing tokens from vocabulary.
    Copied from vision_tokenization/utils/omni_tokenizer/core.py
    """
    unique_new_tokens = []
    seen = set()

    for token in tokens_to_add:
        if token not in existing_vocab and token not in seen:
            unique_new_tokens.append(token)
            seen.add(token)

    if verbose and len(tokens_to_add) != len(unique_new_tokens):
        filtered_count = len(tokens_to_add) - len(unique_new_tokens)
        print(f"Filtered out {filtered_count} duplicate/existing tokens")

    return unique_new_tokens


def add_tokens_with_feedback(tokenizer, tokens, token_type="special"):
    """
    Add tokens to tokenizer with user feedback.
    Copied from vision_tokenization/utils/omni_tokenizer/core.py
    """
    if not tokens:
        print(f"No new {token_type} tokens to add")
        return 0

    print(f"\nAdding {len(tokens)} new {token_type} tokens to tokenizer...")

    # Show sample of tokens being added
    if len(tokens) <= 10:
        for token in tokens:
            print(f"  - {token}")
    else:
        # Show first and last few tokens for large lists
        for token in tokens[:3]:
            print(f"  - {token}")
        print(f"  ... ({len(tokens) - 6} more tokens)")
        for token in tokens[-3:]:
            print(f"  - {token}")

    # Add tokens to tokenizer
    num_added = tokenizer.add_special_tokens({
        "additional_special_tokens": tokens
    })

    print(f"Successfully added {num_added} new tokens")
    return num_added


def rename_reserved_token(save_path: str, tokenizer, old_token: str, new_token: str) -> None:
    """
    Rename a reserved token in saved tokenizer files.
    Copied from vision_tokenization/utils/omni_tokenizer/core.py
    """
    token_id = tokenizer.convert_tokens_to_ids(old_token)
    if token_id == tokenizer.unk_token_id:
        print(f"  ⚠️  {old_token} not found, skipping rename to {new_token}")
        return

    # Modify tokenizer.json
    tokenizer_json_path = os.path.join(save_path, "tokenizer.json")
    if os.path.exists(tokenizer_json_path):
        with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Replace all occurrences
        content = content.replace(f'"{old_token}"', f'"{new_token}"')
        content = content.replace(old_token, new_token)

        with open(tokenizer_json_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"  ✅ Renamed {old_token} to {new_token} (ID {token_id}) in tokenizer.json")

    # Also modify tokenizer_config.json
    config_path = os.path.join(save_path, "tokenizer_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Replace in all config values
        def replace_in_dict(obj):
            if isinstance(obj, dict):
                return {k: replace_in_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_in_dict(item) for item in obj]
            elif isinstance(obj, str):
                return obj.replace(old_token, new_token)
            return obj

        config = replace_in_dict(config)

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        print(f"  ✅ Renamed {old_token} to {new_token} in tokenizer_config.json")

# Add paths for importing audio tokenizers
REPO_ROOT = Path(__file__).resolve().parents[3]
AUDIO_TOKENIZERS_SRC = REPO_ROOT / "src" / "audio_tokenizers"
if str(AUDIO_TOKENIZERS_SRC) not in sys.path:
    sys.path.insert(0, str(AUDIO_TOKENIZERS_SRC))


def save_tokenizer_with_correct_vocab_size(
    tokenizer,
    save_path: str,
    original_vocab_size: int,
    audio_tokenizer_type: str,
    audio_tokenizer_path: str,
    codebook_size: int
) -> None:
    """
    Save tokenizer with correct vocab_size and audio tokenizer info in config.

    Args:
        tokenizer: The tokenizer instance with potentially updated vocabulary
        save_path: Path where tokenizer will be saved
        original_vocab_size: The original vocab size before adding tokens
        audio_tokenizer_type: Type of audio tokenizer (e.g., 'WavTokenizer') - REQUIRED
        audio_tokenizer_path: Path to audio tokenizer model - REQUIRED
        codebook_size: Audio tokenizer codebook size - REQUIRED
    """
    # First save the tokenizer
    tokenizer.save_pretrained(save_path)

    # Get actual vocabulary size after adding tokens
    actual_vocab_size = len(tokenizer.get_vocab())
    base_vocab_size = original_vocab_size

    # Update tokenizer_config.json with correct vocab_size
    config_path = os.path.join(save_path, "tokenizer_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Set vocab_size to actual size for model.resize_token_embeddings() etc.
    config['vocab_size'] = actual_vocab_size
    config['base_vocab_size'] = base_vocab_size
    config['added_tokens_count'] = actual_vocab_size - base_vocab_size

    # Add audio tokenizer configuration (instead of vision_tokenizer)
    config['audio_tokenizer'] = {
        'type': audio_tokenizer_type,
        'path': audio_tokenizer_path or 'auto-loaded-from-huggingface',
        'codebook_size': codebook_size
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✓ Updated tokenizer_config.json:")
    print(f"  Total vocabulary: {actual_vocab_size:,}")
    print(f"  Base vocabulary: {base_vocab_size:,}")
    print(f"  Added tokens: {actual_vocab_size - base_vocab_size:,}")
    print(f"  Audio tokenizer: {audio_tokenizer_type}")
    print(f"  Audio tokenizer path: {audio_tokenizer_path}")
    print(f"  Codebook size: {codebook_size:,}")


def load_audio_tokenizer(audio_tokenizer: str = "WavTokenizer", audio_tokenizer_path: Optional[str] = None):
    """
    Load an audio tokenizer and extract its metadata.

    Args:
        audio_tokenizer: User-friendly tokenizer name (default: "WavTokenizer")
        audio_tokenizer_path: Optional path to audio tokenizer model (ignored for WavTokenizer,
                            which auto-loads from HuggingFace. May be used for other tokenizers.)

    Returns:
        Tuple of (tokenizer instance, codebook_size, tokenizer_name)
    """
    print(f"Loading audio tokenizer metadata ({audio_tokenizer})...")
    
    try:
        if audio_tokenizer == "WavTokenizer":
            # Import WavTokenizerWrapper
            # src/ should already be in sys.path (set at module level)
            from audio_tokenizers.implementations.wavtokenizer import WavTokenizerWrapper
            
            # Create instance to get metadata (device doesn't matter for codebook_size)
            # Note: audio_tokenizer_path is ignored for WavTokenizer - it auto-loads from HuggingFace
            audio_tok = WavTokenizerWrapper(device="cpu")
            
            codebook_size = audio_tok.codebook_size
            tokenizer_name = audio_tok.name
            
            print(f"  ✓ Audio tokenizer: {tokenizer_name}")
            print(f"  ✓ Codebook size: {codebook_size:,}")
            if audio_tokenizer_path:
                print(f"  ℹ️  Note: audio_tokenizer_path ignored for WavTokenizer (auto-loaded from HuggingFace)")
            
            return audio_tok, codebook_size, tokenizer_name
        else:
            raise ValueError(
                f"Unknown audio tokenizer: '{audio_tokenizer}'. "
                f"Available: ['WavTokenizer']"
            )

    except Exception as e:
        raise RuntimeError(f"Failed to load audio tokenizer metadata for '{audio_tokenizer}': {e}")


def create_audio_base_tokenizer(
    text_tokenizer_path: str,
    output_path: str,
    audio_tokenizer: str = "WavTokenizer",
    audio_tokenizer_path: Optional[str] = None,
    num_reserved_tokens: int = 200
) -> Tuple[Any, Dict[str, int]]:
    """
    Create an omnimodal base tokenizer by adding audio tokens to a text tokenizer.

    This function:
    1. Auto-detects codebook size from the audio tokenizer
    2. Adds structure tokens for audio formatting
    3. Adds reserved tokens (RESERVED_001 becomes <|audio_start|>)
    4. Adds audio tokens: <|audio token 000000|> through <|audio token XXXXXX|>
    5. Saves audio_token_mapping.json with all token mappings

    Uses generic functions from vision_tokenization (deduplicate_tokens, 
    add_tokens_with_feedback, rename_reserved_token) for maximum code reuse.

    Args:
        text_tokenizer_path: Path to the base text tokenizer directory
        output_path: Path to save omni-tokenizer
        audio_tokenizer: Audio tokenizer name (default: "WavTokenizer")
        audio_tokenizer_path: Optional path to audio tokenizer model (ignored for WavTokenizer,
                            which auto-loads from HuggingFace. Used for config documentation.)
        num_reserved_tokens: Number of RESERVED_OMNI tokens to add (default: 200)

    Returns:
        Tuple of (tokenizer object, stats dict)

    Files created:
        - tokenizer_config.json, special_tokens_map.json, etc. (standard tokenizer files)
        - audio_token_mapping.json (mapping from audio indices to token IDs)
    """

    print("="*60)
    print("CREATING AUDIO OMNI-TOKENIZER (BASE)")
    print("="*60)

    # Load audio tokenizer to auto-detect codebook size
    _, audio_vocab_size, audio_tokenizer_name = load_audio_tokenizer(
        audio_tokenizer, audio_tokenizer_path
    )

    print(f"\nText tokenizer: {text_tokenizer_path}")
    print(f"Audio vocab size: {audio_vocab_size:,} (auto-detected)")
    print("="*60 + "\n")

    # Load the text tokenizer
    # Import AutoTokenizer here (lazy import) to delay PyTorch/Transformers version conflicts
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        if "TransformGetItemToIndex" in str(e) or "torch._dynamo" in str(e):
            # Try to get version info for better error message
            import torch
            torch_version = getattr(torch, '__version__', 'unknown')
            try:
                import transformers
                transformers_version = getattr(transformers, '__version__', 'unknown')
            except:
                transformers_version = 'unknown'
            
            raise RuntimeError(
                f"PyTorch/Transformers version conflict detected!\n\n"
                f"Installed versions:\n"
                f"  - torch: {torch_version}\n"
                f"  - transformers: {transformers_version}\n\n"
                f"This is a known compatibility issue. Recommended versions:\n"
                f"  - torch==2.6.0 (as in requirements.txt)\n"
                f"  - transformers>=4.44.0,<4.54.0 (compatible with torch 2.6.0)\n\n"
                f"To fix, try:\n"
                f"  pip install transformers==4.53.1\n"
                f"  # or\n"
                f"  pip install 'transformers>=4.44.0,<4.54.0'\n\n"
                f"Original error: {e}"
            ) from e
        raise
    
    print(f"Loading text tokenizer from {text_tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_path, use_fast=True)

    # Store original info - capture base vocab BEFORE any modifications
    original_vocab_size = len(tokenizer.get_vocab())
    print(f"Original vocabulary size: {original_vocab_size:,}")

    # Statistics
    stats = {
        "text_tokenizer": text_tokenizer_path,
        "audio_tokenizer": audio_tokenizer_name,
        "tokenizer_type": "base",
        "original_vocab_size": original_vocab_size,
        "audio_tokens_added": 0,
        "structure_tokens_added": 0,
        "reserved_tokens_added": 0,
        "final_vocab_size": 0
    }

    # Collect all special tokens to add
    special_tokens_to_add = []

    # Get existing vocabulary
    existing_vocab = tokenizer.get_vocab()

    # Add RESERVED_OMNI tokens (200 tokens for all modalities and control)
    print(f"\nAdding {num_reserved_tokens} RESERVED_OMNI tokens...")
    reserved_tokens = []
    for i in range(num_reserved_tokens):
        reserved_tokens.append(f"<|RESERVED_OMNI_{i:03d}|>")

    special_tokens_to_add.extend(reserved_tokens)
    stats["reserved_tokens_added"] = num_reserved_tokens
    stats["structure_tokens_added"] = 3  # Will be renamed from RESERVED_OMNI_001-003

    # Add audio tokens
    print(f"\nGenerating {audio_vocab_size:,} audio tokens for {audio_tokenizer_name}...")
    audio_tokens = []
    # Calculate padding based on vocab size
    padding = len(str(audio_vocab_size - 1))
    for i in range(audio_vocab_size):
        audio_tokens.append(f"<|audio token {i:0{padding}d}|>")

    special_tokens_to_add.extend(audio_tokens)
    stats["audio_tokens_added"] = len(audio_tokens)
    print(f"Token format: <|audio token {{:0{padding}d}}|>")
    print(f"Range: <|audio token {0:0{padding}d}|> to <|audio token {audio_vocab_size-1:0{padding}d}|>")

    # Use generic function from vision_tokenization
    unique_new_tokens = deduplicate_tokens(special_tokens_to_add, existing_vocab)

    # Use generic function from vision_tokenization
    add_tokens_with_feedback(tokenizer, unique_new_tokens)

    # Update final vocab size
    stats["final_vocab_size"] = len(tokenizer)
    print(f"\nNew vocabulary size: {stats['final_vocab_size']:,}")

    # Save the updated tokenizer and update vocab_size in config
    print(f"\nSaving omni-tokenizer to {output_path}...")
    save_tokenizer_with_correct_vocab_size(
        tokenizer,
        output_path,
        original_vocab_size,
        audio_tokenizer_type=audio_tokenizer,
        audio_tokenizer_path=audio_tokenizer_path,
        codebook_size=audio_vocab_size
    )

    # Rename RESERVED_OMNI tokens to audio structure tokens
    # Use generic function from vision_tokenization
    print(f"\nRenaming RESERVED_OMNI tokens to audio structure tokens...")
    token_renames = [
        ("<|RESERVED_OMNI_001|>", "<|audio_start|>"),
        ("<|RESERVED_OMNI_002|>", "<|audio_token_start|>"),
        ("<|RESERVED_OMNI_003|>", "<|audio_end|>"),
    ]

    for old_token, new_token in token_renames:
        rename_reserved_token(output_path, tokenizer, old_token, new_token)

    # Save audio token mapping
    print("\nCreating audio token mapping...")
    audio_mapping = {}
    for i in range(audio_vocab_size):
        token = f"<|audio token {i:0{padding}d}|>"
        token_id = tokenizer.convert_tokens_to_ids(token)
        audio_mapping[i] = token_id

    # Save public mapping file
    mapping_path = os.path.join(output_path, "audio_token_mapping.json")
    with open(mapping_path, 'w') as f:
        json.dump({
            "tokenizer_type": "base",
            "audio_tokenizer": audio_tokenizer_name,
            "audio_tokenizer_type": audio_tokenizer,
            "audio_tokenizer_path": audio_tokenizer_path or "auto-loaded-from-huggingface",
            "audio_token_ids": audio_mapping,
            "audio_vocab_size": audio_vocab_size,
            "audio_token_format": f"<|audio token {{:0{padding}d}}|>",
            "num_reserved_tokens": num_reserved_tokens,
            "original_vocab_size": original_vocab_size,
            "structure_tokens_added": stats["structure_tokens_added"],
            "reserved_tokens_added": stats["reserved_tokens_added"],
            "final_vocab_size": stats["final_vocab_size"]
        }, f, indent=2)
    print(f"Saved audio token mapping to {mapping_path}")

    # Verification - show some token IDs
    print("\n" + "="*60)
    print("VERIFICATION - Sample Token IDs")
    print("="*60)

    # Check boundary marker
    print("\nBoundary marker:")
    boundary_id = tokenizer.convert_tokens_to_ids("<|RESERVED_OMNI_000|>")
    if boundary_id != tokenizer.unk_token_id:
        print(f"  <|RESERVED_OMNI_000|>: ID {boundary_id}")

    # Check audio structure tokens
    print("\nAudio structure tokens:")
    for token in ["<|audio_start|>", "<|audio_token_start|>", "<|audio_end|>"]:
        if token in tokenizer.get_vocab():
            token_id = tokenizer.convert_tokens_to_ids(token)
            print(f"  {token}: ID {token_id}")

    # Check sample audio tokens
    print(f"\nAudio tokens (for {audio_tokenizer_name}):")
    sample_indices = [0, audio_vocab_size // 2, audio_vocab_size - 1]
    for idx in sample_indices:
        token = f"<|audio token {idx:0{padding}d}|>"
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"  {token}: ID {token_id}")

    print("\n" + "="*60)

    return tokenizer, stats


def load_audio_token_mapping(tokenizer_path: str) -> Dict:
    """
    Load audio token mapping from a tokenizer directory.

    Args:
        tokenizer_path: Path to tokenizer directory containing audio_token_mapping.json

    Returns:
        dict: Mapping information including audio_token_ids
    """
    mapping_path = os.path.join(tokenizer_path, "audio_token_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Audio token mapping not found at {mapping_path}")


def get_audio_token_id(audio_index: int, tokenizer_path: str = None, mapping: dict = None) -> int:
    """
    Convert audio index to token ID using saved mapping.

    Args:
        audio_index: Index from audio tokenizer (0 to audio_vocab_size-1)
        tokenizer_path: Path to tokenizer directory (if mapping not provided)
        mapping: Pre-loaded mapping dict (if tokenizer_path not provided)

    Returns:
        int: Token ID in the vocabulary
    """
    if mapping is None:
        mapping = load_audio_token_mapping(tokenizer_path)

    audio_token_ids = mapping["audio_token_ids"]
    # JSON stores keys as strings, so we need to convert to string for lookup
    audio_index_str = str(audio_index)
    if audio_index_str in audio_token_ids:
        return audio_token_ids[audio_index_str]
    elif audio_index in audio_token_ids:
        # Fallback: if keys are integers (shouldn't happen with JSON, but just in case)
        return audio_token_ids[audio_index]
    else:
        raise ValueError(f"Audio index {audio_index} not found. Valid range: 0-{mapping['audio_vocab_size']-1}")
