import os
from datasets import load_from_disk

cache_dir1 = os.path.expandvars("$SCRATCH/benchmark-audio-tokenizer/datasets/eurospeech_cache")
cache_dir2 = os.path.expandvars("$SCRATCH/benchmark-audio-tokenizer/datasets/fleurs_cache")

for lang in ["germany"]:
    ds = load_from_disk(os.path.join(cache_dir1, lang))
    sample = ds[0]
    print(sample)
    print(f"{lang}: {len(ds)} samples, audio shape: {len(sample['audio']['array'])}") 


for lang in ["sw_ke", "he_il", "en_us"]:
    ds = load_from_disk(os.path.join(cache_dir2, lang))
    sample = ds[0]
    print(sample)
    print(f"{lang}: {len(ds)} samples, audio shape: {len(sample['audio']['array'])}")
