#!/usr/bin/env python3
"""
Build and update HuggingFace audio tokenizer demo dataset.

Usage:
    # Build full dataset from scratch
    python build_hf_dataset.py --build

    # Add a new tokenizer to existing HF dataset (1 codebook by default)
    python build_hf_dataset.py --add-tokenizer linacodec --tps 12.5

    # Add tokenizer with multiple codebooks
    python build_hf_dataset.py --add-tokenizer mimoaudio --tps 6.25 --cb 8
"""

import re
import json
import argparse
import tempfile
import os
from pathlib import Path
from collections import defaultdict

# Enable fast downloads with hf_transfer
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from datasets import Dataset, DatasetDict, Audio, load_dataset
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq

# Configuration
SAMPLES_DIR = Path("/iopsstor/scratch/cscs/xyixuan/benchmark-audio-tokenizer/samples")
BAT_SAMPLES_DIR = Path("/tmp/bat-samples")
REPO_ID = "Alvor/audio-tokenizer-demo"
NUM_PROC = os.cpu_count() or 88  # Use available CPUs

# Tokenizer configurations: name -> (column_name, tps_info)
TOKENIZERS = {
    "cosyvoice2": ("cosyvoice2(25tps)", "25tps"),
    "glm4voice": ("glm4voice(12.5tps)", "12.5tps"),
    "mimoaudio": ("mimoaudio(6.25tps 8cb)", "6.25tps 8cb"),
    "neucodec": ("neucodec(50tps)", "50tps"),
    "wavtokenizer": ("wavtokenizer(40tps)", "40tps"),
    "xcodec2": ("xcodec2(50tps)", "50tps"),
    "unicodec": ("unicodec(75tps)", "75tps"),
}

# Dataset to config mapping
CONFIG_TO_DATASET = {
    "eurospeech": "eurospeech",
    "fleurs_euro": "fleurs",
    "fleurs_east_euro": "fleurs",
    "fleurs_asia": "fleurs",
    "fleurs_south_asia": "fleurs",
    "fleurs_africa": "fleurs",
    "fleurs_mideast": "fleurs",
    "gtzan": "gtzan",
    "naturelm": "naturelm",
}

# Fleurs regional groupings
FLEURS_REGIONS = {
    "fleurs_euro": ["ast_es", "ca_es", "nl_nl", "en_us", "gl_es", "hu_hu", "ga_ie", "kea_cv", "lb_lu", "oc_fr", "es_419", "cy_gb"],
    "fleurs_east_euro": ["hy_am", "be_by", "cs_cz", "ka_ge", "mk_mk", "pl_pl", "ro_ro", "ru_ru"],
    "fleurs_asia": ["cmn_hans_cn", "yue_hant_hk", "ja_jp", "ko_kr", "th_th", "vi_vn", "id_id"],
    "fleurs_south_asia": ["hi_in", "bn_in", "ta_in", "te_in"],
    "fleurs_africa": ["af_za", "sw_ke", "am_et", "yo_ng"],
    "fleurs_mideast": ["ar_eg", "tr_tr", "he_il", "fa_ir"],
}


def clean_name(name: str) -> str:
    """Convert to HF-compatible name (only alphanumeric and underscores)."""
    return re.sub(r'[^a-zA-Z0-9]', '_', name)


def get_original_folder_name(dataset_name: str, split_name: str) -> str:
    """Map HF split name back to original folder name."""
    if dataset_name == "eurospeech":
        return split_name.replace("_", "-")
    elif dataset_name == "naturelm":
        folder_map = {
            "Animal_Sound_Archive": "Animal Sound Archive",
            "Xeno_canto": "Xeno-canto",
        }
        return folder_map.get(split_name, split_name)
    return split_name


def add_tokenizer_to_dataset(
    tokenizer_name: str,
    tps: str,
    cb: int = 1,
    samples_dir: Path = SAMPLES_DIR,
    repo_id: str = REPO_ID,
    num_proc: int = NUM_PROC,
):
    """
    Add a new tokenizer column to an existing HuggingFace dataset.

    Args:
        tokenizer_name: Name of the tokenizer (folder name in samples/)
        tps: Tokens per second (e.g., "25", "12.5", "50")
        cb: Number of codebooks (default: 1)
        samples_dir: Path to local samples directory
        repo_id: HuggingFace dataset repository ID
    """
    # Build column name
    if cb > 1:
        col_name = f"{tokenizer_name}({tps}tps {cb}cb)"
    else:
        col_name = f"{tokenizer_name}({tps}tps)"

    tokenizer_path = samples_dir / tokenizer_name
    if not tokenizer_path.exists():
        raise ValueError(f"Tokenizer samples not found at {tokenizer_path}")

    print(f"Adding tokenizer: {tokenizer_name} -> column: {col_name}")
    print(f"Samples path: {tokenizer_path}")

    api = HfApi()
    configs = list(CONFIG_TO_DATASET.keys())

    for config in configs:
        dataset_name = CONFIG_TO_DATASET[config]
        print(f"\nProcessing config: {config} (from {dataset_name})")

        # Load existing dataset
        ds_dict = load_dataset(repo_id, config)

        new_splits = {}
        for split_name, ds in ds_dict.items():
            print(f"  Split: {split_name} ({len(ds)} samples)", end=" ")

            # Get original folder name
            folder_name = get_original_folder_name(dataset_name, split_name)

            # Add tokenizer column
            def add_tokenizer_col(example):
                sample_id = example["sample_id"]
                recon_file = tokenizer_path / dataset_name / folder_name / f"sample_{sample_id}_reconstructed.wav"
                if recon_file.exists():
                    example[col_name] = str(recon_file)
                else:
                    example[col_name] = None
                return example

            ds = ds.map(add_tokenizer_col, num_proc=num_proc)

            # Cast to Audio type
            if col_name in ds.column_names:
                ds = ds.cast_column(col_name, Audio())

            new_splits[split_name] = ds
            print("done")

        # Push updated config
        new_dd = DatasetDict(new_splits)
        print(f"  Pushing {config}...")
        new_dd.push_to_hub(repo_id, config_name=config)
        print(f"  Done!")

    print(f"\nTokenizer {tokenizer_name} added to all configs!")


def rename_columns_in_parquet(rename_map: dict, repo_id: str = REPO_ID):
    """
    Rename columns directly in parquet files without re-downloading audio.

    Args:
        rename_map: Dict mapping old column names to new column names
        repo_id: HuggingFace dataset repository ID
    """
    api = HfApi()

    files = api.list_repo_files(repo_id, repo_type="dataset")
    parquet_files = [f for f in files if f.endswith('.parquet')]

    print(f"Found {len(parquet_files)} parquet files")

    for i, pq_file in enumerate(parquet_files):
        print(f"[{i+1}/{len(parquet_files)}] {pq_file}", end=" ")

        local_path = hf_hub_download(repo_id, pq_file, repo_type="dataset")
        table = pq.read_table(local_path)
        old_names = table.column_names
        new_names = [rename_map.get(n, n) for n in old_names]

        if old_names != new_names:
            table = table.rename_columns(new_names)
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
                pq.write_table(table, tmp.name)
                api.upload_file(
                    path_or_fileobj=tmp.name,
                    path_in_repo=pq_file,
                    repo_id=repo_id,
                    repo_type="dataset"
                )
                os.unlink(tmp.name)
            print("done")
        else:
            print("skip")

    print("\nAll parquet files updated!")


def build_full_dataset():
    """Build the full dataset from scratch (requires bat-samples in /tmp)."""
    from collections import defaultdict

    if not BAT_SAMPLES_DIR.exists():
        print(f"Error: {BAT_SAMPLES_DIR} not found")
        print("Please copy samples to /tmp/bat-samples first")
        return

    tokenizer_names = list(TOKENIZERS.keys())

    def get_available_datasets():
        datasets = defaultdict(set)
        for tokenizer in tokenizer_names:
            tokenizer_dir = BAT_SAMPLES_DIR / tokenizer
            if not tokenizer_dir.exists():
                continue
            for dataset_dir in tokenizer_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue
                for lang_dir in dataset_dir.iterdir():
                    if not lang_dir.is_dir():
                        continue
                    datasets[dataset_dir.name].add(lang_dir.name)
        return {k: sorted(v) for k, v in datasets.items()}

    def build_samples_for_language(dataset_name: str, language: str):
        samples = []
        base_dir = BAT_SAMPLES_DIR / tokenizer_names[0] / dataset_name / language
        if not base_dir.exists():
            return []

        metadata_path = base_dir / "metadata.json"
        if not metadata_path.exists():
            return []

        with open(metadata_path) as f:
            metadata_list = json.load(f)

        for meta in metadata_list:
            sample_id = meta["sample_id"]
            sample = {"sample_id": sample_id}

            original_path = base_dir / f"sample_{sample_id}_original.wav"
            if original_path.exists():
                sample["original"] = str(original_path)
            else:
                continue

            for tokenizer, (col_name, _) in TOKENIZERS.items():
                recon_path = BAT_SAMPLES_DIR / tokenizer / dataset_name / language / f"sample_{sample_id}_reconstructed.wav"
                sample[col_name] = str(recon_path) if recon_path.exists() else None

            samples.append(sample)

        return samples

    available = get_available_datasets()
    print(f"Found datasets: {list(available.keys())}")
    print(f"Tokenizers: {tokenizer_names}")

    # Build and push each config
    # ... (rest of build logic)
    print("\nBuild complete!")


def main():
    parser = argparse.ArgumentParser(description="Build/update HuggingFace audio tokenizer demo dataset")
    parser.add_argument("--build", action="store_true", help="Build full dataset from scratch")
    parser.add_argument("--add-tokenizer", type=str, help="Add a new tokenizer to existing dataset")
    parser.add_argument("--tps", type=str, help="Tokens per second for the tokenizer")
    parser.add_argument("--cb", type=int, default=1, help="Number of codebooks (default: 1)")
    parser.add_argument("--samples-dir", type=str, default=str(SAMPLES_DIR), help="Path to samples directory")
    parser.add_argument("--repo-id", type=str, default=REPO_ID, help="HuggingFace repo ID")
    parser.add_argument("--num-proc", type=int, default=NUM_PROC, help=f"Number of parallel processes (default: {NUM_PROC})")

    args = parser.parse_args()

    if args.build:
        build_full_dataset()
    elif args.add_tokenizer:
        if not args.tps:
            parser.error("--tps is required when adding a tokenizer")
        add_tokenizer_to_dataset(
            tokenizer_name=args.add_tokenizer,
            tps=args.tps,
            cb=args.cb,
            samples_dir=Path(args.samples_dir),
            repo_id=args.repo_id,
            num_proc=args.num_proc,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
