import os
from datasets import load_dataset, Dataset, Audio, Features, Value
import json

# scratch_dir = os.path.expandvars("$SCRATCH/benchmark-audio-tokenizer/datasets")
# cache_dir = os.path.join(scratch_dir, "eurospeech_cache")
# Modified to use capstor for xyixuan
cache_dir = "/capstor/store/cscs/swissai/infra01/audio-datasets/eurospeech_cache"
os.makedirs(cache_dir, exist_ok=True)

languages = [
    "bosnia-herzegovina", "bulgaria", "croatia", "denmark", "estonia", "finland",
    "france", "germany", "greece", "iceland", "italy", "latvia", "lithuania",
    "malta", "norway", "portugal", "serbia", "slovakia", "slovenia",
    "sweden", "uk", "ukraine"
]

def download_language(lang, n=100, split="train"):
    print(f"\n{'='*60}")
    print(f"Downloading {lang} ({split}): {n} samples")
    print(f"{'='*60}")
    
    out_dir = os.path.join(cache_dir, lang)
    if os.path.exists(os.path.join(out_dir, "summary.json")):
        print(f"Already exists, skipping...")
        return 0
    
    # Use num_proc for parallel downloading
    # Use split slicing to only load n samples (e.g., "train[:100]")
    dataset = load_dataset("disco-eth/EuroSpeech", lang, split=f"{split}[:{n}]", num_proc=32)

    # Use map with batch processing for better performance
    def process_batch(examples, indices):
        """Process a batch of examples efficiently"""
        batch_size = len(examples["audio"])
        processed = {
            "audio": examples["audio"],
            "text": examples.get("text", [""] * batch_size),
            "language": [lang] * batch_size,
            "sample_id": indices
        }
        return processed

    # Apply batch processing with multiple workers
    dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=100,
        num_proc=4,  # Use 4 processes for mapping
        with_indices=True,
        desc=f"Processing {lang}"
    )

    # Convert to list format for compatibility
    samples = []
    for i, sample in enumerate(dataset):
        audio_data = sample["audio"]
        samples.append({
            "audio": {
                "array": audio_data["array"],
                "sampling_rate": audio_data["sampling_rate"],
                "path": audio_data.get("path", f"{lang}_{i}")
            },
            "text": sample["text"],
            "language": sample["language"],
            "sample_id": sample["sample_id"]
        })
    
    print(f"Saving {len(samples)} samples...")
    
    features = Features({
        'audio': Audio(sampling_rate=samples[0]['audio']['sampling_rate']),
        'text': Value('string'),
        'language': Value('string'),
        'sample_id': Value('int64')
    })
    
    ds = Dataset.from_list(samples, features=features)
    ds.save_to_disk(out_dir)
    
    total_dur = sum(len(s["audio"]["array"]) / s["audio"]["sampling_rate"] for s in samples)
    summary = {
        "language": lang,
        "num_samples": len(samples),
        "sampling_rate": samples[0]["audio"]["sampling_rate"],
        "total_duration_sec": total_dur,
        "split": split 
    }
    
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved {len(samples)} samples ({total_dur/60:.1f} min)")
    return len(samples)

if __name__ == "__main__":
    total = 0
    for lang in languages:
        # Use validation split for Slovakia, train for others
        split = "validation" if lang == "slovakia" else "train"
        total += download_language(lang, n=100, split=split)
    
    print(f"\nCOMPLETE: {total} samples in {cache_dir}")