#!/usr/bin/env python3
"""
Universal audio sampling script for all tokenizers.
Generates original and reconstructed audio pairs for multiple tokenizers.
"""

import os
import random
import json
import torch
import numpy as np
import soundfile as sf
from datasets import load_from_disk
from tqdm import tqdm
import sys
import argparse
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from audio_tokenizers import get_tokenizer, list_tokenizers


# ---------------------- CONFIG ----------------------
# Use capstor for the cached datasets
CAPSTOR_CACHE = "/capstor/store/cscs/swissai/infra01/audio-datasets"
OUTPUT_ROOT = os.path.join(os.getcwd(), "samples")
DEFAULT_NUM_SAMPLES = 5
SEED = 42

DATASETS = {
    "eurospeech": {
        "cache_dir": os.path.join(CAPSTOR_CACHE, "eurospeech_cache"),
        "languages": [
            "bosnia-herzegovina", "bulgaria", "croatia", "denmark", "estonia",
            "finland", "france", "germany", "greece", "iceland", "italy", "latvia",
            "lithuania", "malta", "norway", "portugal", "serbia", "slovakia",
            "slovenia", "sweden", "uk", "ukraine"
        ],
    },
    "fleurs": {
        "cache_dir": os.path.join(CAPSTOR_CACHE, "fleurs_cache"),
        "languages": [
            "ast_es", "ca_es", "nl_nl", "en_us", "gl_es", "hu_hu", "ga_ie",
            "kea_cv", "lb_lu", "oc_fr", "es_419", "cy_gb",
            "hy_am", "be_by", "cs_cz", "ka_ge", "mk_mk", "pl_pl", "ro_ro", "ru_ru",
            "cmn_hans_cn", "yue_hant_hk", "ja_jp", "ko_kr",
            "hi_in", "bn_in", "ta_in", "te_in",
            "th_th", "vi_vn", "id_id",
            "af_za", "sw_ke", "am_et", "yo_ng",
            "ar_eg", "tr_tr", "he_il", "fa_ir",
        ],
    },
}


def sample_and_reconstruct(
    tokenizer_name: str,
    datasets: Optional[List[str]] = None,
    languages: Optional[List[str]] = None,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    seed: int = SEED,
    device: str = None
):
    """
    Sample and reconstruct audio for a specific tokenizer.

    Args:
        tokenizer_name: Name of the tokenizer to use
        datasets: List of dataset names to process (default: all)
        languages: List of language codes to process (default: all)
        num_samples: Number of samples per language
        seed: Random seed for reproducibility
        device: Device to use (cuda/cpu)
    """
    random.seed(seed)
    torch.manual_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"UNIVERSAL AUDIO SAMPLING SCRIPT")
    print(f"{'='*60}")
    print(f"Tokenizer: {tokenizer_name}")
    print(f"Device: {device}")
    print(f"Samples per language: {num_samples}")
    print(f"Random seed: {seed}")
    print(f"{'='*60}")

    # Load tokenizer
    print(f"\nLoading {tokenizer_name} tokenizer...")
    try:
        tokenizer = get_tokenizer(tokenizer_name, device=device)
        print(f"✓ Tokenizer loaded successfully")
        print(f"  Input SR: {tokenizer.sample_rate} Hz")
        print(f"  Output SR: {tokenizer.output_sample_rate} Hz")
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        return

    # Determine which datasets to process
    datasets_to_process = datasets if datasets else list(DATASETS.keys())

    # Process each dataset
    for dataset_name in datasets_to_process:
        if dataset_name not in DATASETS:
            print(f"\n⚠ Dataset '{dataset_name}' not found in config, skipping...")
            continue

        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name.upper()}")
        print(f"{'='*60}")

        cfg = DATASETS[dataset_name]
        cache_dir = cfg["cache_dir"]

        # Check if cache directory exists
        if not os.path.exists(cache_dir):
            print(f"⚠ Cache directory not found: {cache_dir}")
            print(f"  Please run the download script first.")
            continue

        # Determine which languages to process
        if languages:
            langs_to_process = [l for l in languages if l in cfg["languages"]]
        else:
            langs_to_process = cfg["languages"]

        print(f"Languages to process: {len(langs_to_process)}")

        for lang in langs_to_process:
            lang_dir = os.path.join(cache_dir, lang)

            if not os.path.exists(lang_dir):
                print(f"⚠ Skipping {lang} (not found at {lang_dir})")
                continue

            print(f"\n--- Processing {lang} ({dataset_name}) ---")

            # Load dataset
            try:
                ds = load_from_disk(lang_dir)
                n = len(ds)
                if n == 0:
                    print(f"⚠ Empty dataset for {lang}")
                    continue
                print(f"  Dataset size: {n} samples")
            except Exception as e:
                print(f"⚠ Error loading dataset: {e}")
                continue

            # Sample indices
            indices = random.sample(range(n), min(num_samples, n))

            # Create output directory
            output_dir = os.path.join(OUTPUT_ROOT, tokenizer_name, dataset_name, lang)
            os.makedirs(output_dir, exist_ok=True)

            meta = []
            successful = 0
            failed = 0

            print(f"  Sampling {len(indices)} audio clips...")

            for i, idx in enumerate(tqdm(indices, desc=f"  {lang}", leave=False)):
                sample = ds[idx]
                audio = sample["audio"]["array"]
                sr = sample["audio"]["sampling_rate"]

                # Convert to tensor
                audio_tensor = torch.from_numpy(audio).float()
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
                audio_tensor = audio_tensor.to(device)

                try:
                    # Encode and decode separately
                    # Note: GLM-4-Voice has minimal voice cloning effect, so we don't use reconstruct
                    tokens, encode_info = tokenizer.encode(audio_tensor, sr=sr)

                    # Decode
                    reconstructed, decode_info = tokenizer.decode(tokens)

                    # Get output sample rate
                    recon_sr = decode_info.get("output_sample_rate", tokenizer.output_sample_rate)
                    if recon_sr is None:
                        recon_sr = sr  # Fallback to input sample rate

                    # Convert back to numpy
                    recon_audio = reconstructed.squeeze().cpu().numpy()

                    # Normalize to avoid clipping
                    audio_normalized = audio / (np.max(np.abs(audio)) + 1e-8)
                    recon_normalized = recon_audio / (np.max(np.abs(recon_audio)) + 1e-8)

                    # Save files
                    sample_id = f"{i+1:04d}"
                    orig_path = os.path.join(output_dir, f"sample_{sample_id}_original.wav")
                    recon_path = os.path.join(output_dir, f"sample_{sample_id}_reconstructed.wav")

                    sf.write(orig_path, audio_normalized, sr)
                    sf.write(recon_path, recon_normalized, recon_sr)

                    # Store metadata
                    meta.append({
                        "dataset": dataset_name,
                        "language": lang,
                        "sample_id": sample_id,
                        "dataset_index": idx,
                        "original_file": orig_path,
                        "reconstructed_file": recon_path,
                        "original_sr": sr,
                        "reconstructed_sr": recon_sr,
                        "original_duration": len(audio) / sr,
                        "reconstructed_duration": len(recon_audio) / recon_sr,
                        "num_tokens": tokens.numel(),
                        "token_shape": list(tokens.shape),
                    })

                    successful += 1

                except Exception as e:
                    print(f"\n  ✗ Error processing sample {idx}: {e}")
                    failed += 1
                    continue

            # Save metadata
            if meta:
                meta_path = os.path.join(output_dir, "metadata.json")
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
                print(f"  ✓ Saved {successful} pairs for {lang}")
                if failed > 0:
                    print(f"  ✗ Failed: {failed} samples")
            else:
                print(f"  ✗ No samples processed successfully for {lang}")

    print(f"\n{'='*60}")
    print(f"COMPLETED!")
    print(f"Samples saved under: {OUTPUT_ROOT}/{tokenizer_name}/")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Universal audio sampling script for all tokenizers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Sample with NeuCodec tokenizer
            python generate_samples.py --tokenizer neucodec

            # Sample with CosyVoice2, only FLEURS dataset
            python generate_samples.py --tokenizer cosyvoice2 --datasets fleurs

            # Sample specific languages with more samples
            python generate_samples.py --tokenizer neucodec --languages en_us de_de --num-samples 10

            # List available tokenizers
            python generate_samples.py --list-tokenizers
        """
    )

    parser.add_argument(
        "--tokenizer", "-t",
        type=str,
        help="Tokenizer to use (e.g., neucodec, cosyvoice2, tadicodec)"
    )

    parser.add_argument(
        "--datasets", "-d",
        nargs="+",
        default=None,
        help="Datasets to process (default: all). Options: eurospeech, fleurs"
    )

    parser.add_argument(
        "--languages", "-l",
        nargs="+",
        default=None,
        help="Specific languages to process (default: all)"
    )

    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"Number of samples per language (default: {DEFAULT_NUM_SAMPLES})"
    )

    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED})"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Default: auto-detect"
    )

    parser.add_argument(
        "--list-tokenizers",
        action="store_true",
        help="List all available tokenizers and exit"
    )

    args = parser.parse_args()

    # List tokenizers if requested
    if args.list_tokenizers:
        print("\nAvailable tokenizers:")
        for name in sorted(list_tokenizers()):
            print(f"  - {name}")
        print()
        return

    # Check if tokenizer is specified
    if not args.tokenizer:
        print("Error: Please specify a tokenizer with --tokenizer")
        print("\nAvailable tokenizers:")
        for name in sorted(list_tokenizers()):
            print(f"  - {name}")
        print("\nUse --help for more information")
        return

    # Check if tokenizer exists
    available = list_tokenizers()
    if args.tokenizer not in available:
        print(f"Error: Tokenizer '{args.tokenizer}' not found")
        print("\nAvailable tokenizers:")
        for name in sorted(available):
            print(f"  - {name}")
        return

    # Run sampling
    sample_and_reconstruct(
        tokenizer_name=args.tokenizer,
        datasets=args.datasets,
        languages=args.languages,
        num_samples=args.num_samples,
        seed=args.seed,
        device=args.device
    )


if __name__ == "__main__":
    main()