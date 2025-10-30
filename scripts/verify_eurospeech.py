import os
from datasets import load_from_disk

cache_dir = os.path.expandvars("$SCRATCH/benchmark-audio-tokenizer/datasets/eurospeech_cache")

for lang in ["france", "germany", "uk"]:
    ds = load_from_disk(os.path.join(cache_dir, lang))
    sample = ds[0]
    print(f"{lang}: {len(ds)} samples, audio shape: {len(sample['audio']['array'])}")
