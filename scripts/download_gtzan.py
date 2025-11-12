import os
from datasets import load_dataset, Dataset, Audio, Features, Value
import json
import time
import argparse
from collections import defaultdict

cache_dir = "/capstor/store/cscs/swissai/infra01/audio-datasets/gtzan_cache"
os.makedirs(cache_dir, exist_ok=True)

def download_gtzan_by_genre(n_per_genre=25, split="train"):
    """
    Download samples from GTZAN Music Genre dataset, balanced across genres.

    The GTZAN dataset contains 1000 audio tracks (100 per genre) across 10 genres:
    blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

    Args:
        n_per_genre: Number of samples per genre (default: 10)
        split: Dataset split to use - only "train" is available (default: "train")

    Note: GTZAN specifications:
          - 1000 tracks total (100 per genre)
          - 30 seconds per track
          - 22,050 Hz sample rate
          - Mono, 16-bit WAV
          - Total size: ~1.2 GB
    """
    print(f"\n{'='*60}")
    print(f"Downloading GTZAN Music Genre Dataset ({split})")
    print(f"{n_per_genre} samples per genre (10 genres total)")
    print(f"{'='*60}")

    out_dir = os.path.join(cache_dir, "gtzan")
    if os.path.exists(os.path.join(out_dir, "summary.json")):
        print(f"Already exists, skipping...")
        return 0

    try:
        print(f"Loading dataset...")
        dataset = load_dataset(
            "storylinez/gtzan-music-genre-dataset",
            split=split,
            verification_mode="no_checks"
        )

        print(f"\nDataset loaded: {len(dataset)} total samples")
        
        genre_names = dataset.features["label"].names
        print(f"Genres available: {genre_names}")
        
        print(f"\nCollecting {n_per_genre} samples per genre...")
        genre_samples = defaultdict(list)
        genre_counts = defaultdict(int)

        for idx, sample in enumerate(dataset):
            label_id = sample["label"]
            genre_name = genre_names[label_id]
        
            if genre_counts[genre_name] < n_per_genre:
                audio_data = sample["audio"]
                genre_samples[genre_name].append({
                    "audio": {
                        "array": audio_data["array"],
                        "sampling_rate": audio_data["sampling_rate"],
                        "path": audio_data.get("path", f"gtzan_{genre_name}_{genre_counts[genre_name]}")
                    },
                    "label": label_id,
                    "genre_name": genre_name,
                    "sample_id": len([s for genre_list in genre_samples.values() for s in genre_list])
                })
                genre_counts[genre_name] += 1
                
                total_collected = sum(genre_counts.values())
                if total_collected % 25 == 0:
                    print(f"  Collected {total_collected} samples")
                    print(f"    Current counts: {dict(genre_counts)}")

            if len(genre_counts) == len(genre_names) and all(count >= n_per_genre for count in genre_counts.values()):
                print(f"\n  Collected enough samples from all genres")
                break

        samples = []
        for genre, genre_list in genre_samples.items():
            samples.extend(genre_list)

        if not samples:
            print(f"No samples downloaded")
            return 0

        print(f"\nSaving {len(samples)} samples...")

        features = Features({
            'audio': Audio(sampling_rate=22050),
            'label': Value('int64'),
            'genre_name': Value('string'),
            'sample_id': Value('int64')
        })

        ds = Dataset.from_list(samples, features=features)
        ds.save_to_disk(out_dir)

        total_dur = sum(len(s["audio"]["array"]) / s["audio"]["sampling_rate"] for s in samples)

        summary = {
            "dataset": "GTZAN-music-genre",
            "num_samples": len(samples),
            "sampling_rate": 22050,
            "total_duration_sec": total_dur,
            "total_duration_min": total_dur / 60,
            "total_duration_hours": total_dur / 3600,
            "split": split,
            "samples_per_genre": n_per_genre,
            "num_genres": len(genre_counts),
            "genre_distribution": dict(genre_counts),
            "genre_names": genre_names,
            "track_duration_sec": 30,
            "description": "GTZAN Music Genre Classification Dataset - 10 genres of music"
        }

        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n✓ Saved {len(samples)} samples ({total_dur/60:.1f} min / {total_dur/3600:.2f} hours)")
        print(f"\nGenre distribution:")
        for genre, count in sorted(genre_counts.items()):
            print(f"  {genre}: {count} samples")
        
        return len(samples)

    except Exception as e:
        print(f"Error downloading GTZAN: {e}")
        import traceback
        traceback.print_exc()
        return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download GTZAN Music Genre dataset')
    parser.add_argument('--samples-per-genre', '-n', type=int, default=25,
                        help='Number of samples per genre (default: 25, max: 100)')
    parser.add_argument('--split', type=str, default='train',
                        help='Dataset split to use (default: train)')
    args = parser.parse_args()

    print("="*60)
    print("GTZAN Music Genre Dataset Download Script")
    print("="*60)
    print(f"Samples per genre: {args.samples_per_genre}")
    print(f"Total genres: 10 (blues, classical, country, disco, hiphop,")
    print(f"              jazz, metal, pop, reggae, rock)")
    print(f"Expected total samples: {args.samples_per_genre * 10}")
    print(f"Split: {args.split}")
    print(f"Full dataset: 1000 tracks (100 per genre), 30s each")
    print(f"Sample rate: 22,050 Hz, Mono, 16-bit")
    print("="*60)

    start_time = time.time()
    result = download_gtzan_by_genre(
        n_per_genre=args.samples_per_genre,
        split=args.split
    )
    overall_elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"Total samples downloaded: {result:,}")
    print(f"Execution time: {overall_elapsed/60:.2f} min ({overall_elapsed:.1f} sec)")
    print(f"Data saved to: {cache_dir}/gtzan")
    print(f"{'='*60}")