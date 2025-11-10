import os
from datasets import load_dataset, Dataset, Audio, Features, Value
import json
import time
import argparse
from collections import defaultdict

# Use capstor for the cached datasets
cache_dir = "/capstor/store/cscs/swissai/infra01/audio-datasets/naturelm_cache"
os.makedirs(cache_dir, exist_ok=True)

def download_naturelm_by_source(n_per_source=100, split="train"):
    """
    Download samples from NatureLM audio training dataset, balanced across source datasets.

    Uses streaming to avoid downloading all 10K+ parquet files.

    Args:
        n_per_source: Number of samples per source_dataset (default: 100)
        split: Dataset split to use - only "train" is available (default: "train")

    Note: NatureLM has 26.4M samples from 6 source datasets:
          - Xeno-canto (bird sounds)
          - iNaturalist (nature sounds)
          - WavCaps (general audio with captions)
          - Watkins Marine Mammal Sound Database
          - And others
    """
    print(f"\n{'='*60}")
    print(f"Downloading NatureLM ({split}): {n_per_source} samples per source")
    print(f"Using streaming to avoid full dataset download")
    print(f"{'='*60}")

    out_dir = os.path.join(cache_dir, "naturelm")
    if os.path.exists(os.path.join(out_dir, "summary.json")):
        print(f"Already exists, skipping...")
        return 0

    try:
        # Use streaming to avoid downloading all files
        print(f"Loading dataset with streaming...")
        dataset = load_dataset(
            "EarthSpeciesProject/NatureLM-audio-training",
            split=split,
            streaming=True
        )

        # Collect samples until we have n_per_source from each source
        print(f"\nCollecting samples (streaming)...")
        source_samples = defaultdict(list)
        source_counts = defaultdict(int)

        # Track progress
        total_seen = 0
        total_collected = 0

        for sample in dataset:
            total_seen += 1
            source = sample.get("source_dataset", "unknown")

            # Check if we need more samples from this source
            if source_counts[source] < n_per_source:
                audio_data = sample["audio"]
                source_samples[source].append({
                    "audio": {
                        "array": audio_data["array"],
                        "sampling_rate": audio_data["sampling_rate"],
                        "path": audio_data.get("path", sample.get("file_name", f"naturelm_{total_collected}"))
                    },
                    "file_name": sample["file_name"],
                    "metadata": sample["metadata"],
                    "source_dataset": sample["source_dataset"],
                    "id": sample["id"],
                    "license": sample["license"],
                    "instruction": sample["instruction"],
                    "instruction_text": sample["instruction_text"],
                    "output": sample["output"],
                    "task": sample["task"],
                    "sample_id": total_collected
                })
                source_counts[source] += 1
                total_collected += 1

                # Print progress every 100 samples
                if total_collected % 100 == 0:
                    print(f"  Collected {total_collected} samples (seen {total_seen} total)")
                    print(f"    Current counts: {dict(source_counts)}")

            # Check if we have enough samples from all sources
            # We'll stop after seeing 50000 samples or when all sources have enough
            if total_seen >= 50000:
                print(f"\n  Reached iteration limit (50000 samples seen)")
                break

            # Check if we have enough from all discovered sources
            if len(source_counts) >= 3:  # At least 3 different sources found
                all_full = all(count >= n_per_source for count in source_counts.values())
                if all_full:
                    print(f"\n  Collected enough samples from all sources")
                    break

        print(f"\n  Total samples seen: {total_seen:,}")
        print(f"  Total samples collected: {total_collected}")
        print(f"  Sources found: {len(source_samples)}")

        # Flatten all samples
        samples = []
        for source, source_list in source_samples.items():
            samples.extend(source_list)

        if not samples:
            print(f"No samples downloaded")
            return 0

        print(f"\nSaving {len(samples)} samples...")

        features = Features({
            'audio': Audio(sampling_rate=16000),
            'file_name': Value('string'),
            'metadata': Value('string'),
            'source_dataset': Value('string'),
            'id': Value('string'),
            'license': Value('string'),
            'instruction': Value('string'),
            'instruction_text': Value('string'),
            'output': Value('string'),
            'task': Value('string'),
            'sample_id': Value('int64')
        })

        ds = Dataset.from_list(samples, features=features)
        ds.save_to_disk(out_dir)

        total_dur = sum(len(s["audio"]["array"]) / s["audio"]["sampling_rate"] for s in samples)

        # Count tasks
        task_counts = {}
        for s in samples:
            tsk = s["task"]
            task_counts[tsk] = task_counts.get(tsk, 0) + 1

        summary = {
            "dataset": "NatureLM-audio-training",
            "num_samples": len(samples),
            "sampling_rate": 16000,
            "total_duration_sec": total_dur,
            "total_duration_min": total_dur / 60,
            "total_duration_hours": total_dur / 3600,
            "split": split,
            "samples_per_source": n_per_source,
            "total_samples_seen": total_seen,
            "source_datasets": dict(source_counts),
            "task_distribution": task_counts
        }

        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n✓ Saved {len(samples)} samples ({total_dur/60:.1f} min / {total_dur/3600:.2f} hours)")
        print(f"\nSource dataset distribution:")
        for src, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {src}: {count} samples")
        print(f"\nTop 10 tasks:")
        for task, count in sorted(task_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {task}: {count} samples")
        return len(samples)

    except Exception as e:
        print(f"Error downloading NatureLM: {e}")
        import traceback
        traceback.print_exc()
        return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download NatureLM audio training dataset')
    parser.add_argument('--samples-per-source', '-n', type=int, default=100,
                        help='Number of samples per source dataset (default: 100)')
    parser.add_argument('--split', type=str, default='train',
                        help='Dataset split to use (default: train)')
    args = parser.parse_args()

    print("="*60)
    print("NatureLM Audio Training Dataset Download Script")
    print("="*60)
    print(f"Samples per source: {args.samples_per_source}")
    print(f"Split: {args.split}")
    print(f"Total dataset size: 26.4M samples from 6 source datasets")
    print(f"Method: Streaming (avoids downloading all 10K+ parquet files)")
    print("="*60)

    start_time = time.time()
    result = download_naturelm_by_source(
        n_per_source=args.samples_per_source,
        split=args.split
    )
    overall_elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"Total samples downloaded: {result:,}")
    print(f"Execution time: {overall_elapsed/60:.2f} min ({overall_elapsed:.1f} sec)")
    print(f"Data saved to: {cache_dir}/naturelm")
    print(f"{'='*60}")
