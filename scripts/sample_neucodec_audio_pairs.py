import os
import random
import json
import torch
import numpy as np
import soundfile as sf
from datasets import load_from_disk
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from audio_tokenizers import get_tokenizer


# ---------------------- CONFIG ----------------------
SCRATCH_DIR = os.path.expandvars("$SCRATCH/benchmark-audio-tokenizer/datasets")
OUTPUT_ROOT = os.path.join(os.getcwd(), "samples", "neucodec")
NUM_SAMPLES = 5
SEED = 42
random.seed(SEED)

DATASETS = {
    "eurospeech": {
        "cache_dir": os.path.join(SCRATCH_DIR, "eurospeech_cache"),
        "languages": [
            "bosnia-herzegovina", "bulgaria", "croatia", "denmark", "estonia",
            "finland", "france", "germany", "greece", "iceland", "italy", "latvia",
            "lithuania", "malta", "norway", "portugal", "serbia", "slovakia",
            "slovenia", "sweden", "uk", "ukraine"
        ],
    },
    "fleurs": {
        "cache_dir": os.path.join(SCRATCH_DIR, "fleurs_cache"),
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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ---------------------- TOKENIZER ----------------------
print("\nLoading NeuCodec tokenizer...")
tokenizer = get_tokenizer("neucodec", device=device)
print(f"Tokenizer loaded. Input SR={tokenizer.sample_rate}, Output SR={tokenizer.output_sample_rate}")

# ---------------------- MAIN LOOP ----------------------
for dataset_name, cfg in DATASETS.items():
    print(f"\n=== Processing dataset: {dataset_name.upper()} ===")
    cache_dir = cfg["cache_dir"]

    for lang in cfg["languages"]:
        lang_dir = os.path.join(cache_dir, lang)
        if not os.path.exists(lang_dir):
            print(f"Skipping {lang} (missing at {lang_dir})")
            continue

        ds = load_from_disk(lang_dir)
        n = len(ds)
        if n == 0:
            print(f"Empty dataset for {lang}")
            continue

        indices = random.sample(range(n), min(NUM_SAMPLES, n))
        output_dir = os.path.join(OUTPUT_ROOT, dataset_name, lang)
        os.makedirs(output_dir, exist_ok=True)

        meta = []
        print(f"Sampling {len(indices)} clips from {lang} ({dataset_name})...")

        for i, idx in enumerate(tqdm(indices, desc=f"{lang}")):
            sample = ds[idx]
            audio = sample["audio"]["array"]
            sr = sample["audio"]["sampling_rate"]

            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).float().to(device)

            try:
                tokens, _ = tokenizer.encode(audio_tensor, sr=sr)
                reconstructed, decode_info = tokenizer.decode(tokens)
                print(f"Processed sample {i+1} / {len(indices)}: tokens={tokens.numel()}, recon_len={reconstructed.numel()}")
                recon_sr = decode_info.get("output_sample_rate", sr)
                recon_audio = reconstructed.squeeze().cpu().numpy()

                # Normalize to avoid clipping
                audio = audio / (np.max(np.abs(audio)) + 1e-8)
                recon_audio = recon_audio / (np.max(np.abs(recon_audio)) + 1e-8)

                sample_id = f"{i+1:04d}"
                orig_path = os.path.join(output_dir, f"sample_{sample_id}_original.wav")
                recon_path = os.path.join(output_dir, f"sample_{sample_id}_reconstructed.wav")

                sf.write(orig_path, audio, sr)
                sf.write(recon_path, recon_audio, recon_sr)

                meta.append({
                    "dataset": dataset_name,
                    "language": lang,
                    "sample_id": sample_id,
                    "original_file": orig_path,
                    "reconstructed_file": recon_path,
                    "original_sr": sr,
                    "reconstructed_sr": recon_sr,
                })

            except Exception as e:
                print(f"Error processing {lang} idx={idx}: {e}")
                continue

        # Save metadata
        if meta:
            meta_path = os.path.join(output_dir, "metadata.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            print(f"Saved {len(meta)} pairs for {lang} at {output_dir}")
        else:
            print(f"No samples processed successfully for {lang}")

print("\nAll done! Samples saved under:", OUTPUT_ROOT)
