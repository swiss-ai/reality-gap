import os
import sys
import torch
import numpy as np
from datasets import load_from_disk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from audio_tokenizers import get_tokenizer

os.environ["OPENBLAS_NUM_THREADS"] = "72"
os.environ["OMP_NUM_THREADS"] = "72"


# Test mit einem EuroSpeech Sample
SCRATCH_DIR = os.path.expandvars("$SCRATCH/benchmark-audio-tokenizer/datasets")
cache_dir = os.path.join(SCRATCH_DIR, "eurospeech_cache")
lang = "germany"

print("Loading tokenizer...")
tokenizer = get_tokenizer('xcodec2', device='cpu')  # Start with CPU for testing
print(f"Tokenizer loaded: {tokenizer}")
print(f"  Input sample rate: {tokenizer.sample_rate} Hz")
print(f"  Output sample rate: {tokenizer.output_sample_rate} Hz")
print(f"  Codebook size: {tokenizer.codebook_size}")
print(f"  Downsample rate: {tokenizer.downsample_rate}x")

print(f"\nLoading dataset: {lang}")
lang_dir = os.path.join(cache_dir, lang)
if not os.path.exists(lang_dir):
    print(f"ERROR: Dataset not found at {lang_dir}")
    print("Please download EuroSpeech first: python scripts/download_eurospeech.py")
    sys.exit(1)

ds = load_from_disk(lang_dir)
print(f"Loaded {len(ds)} samples")

# Test with first sample
print("\nTesting with first sample...")
sample = ds[0]
audio_array = sample['audio']['array']
sr = sample['audio']['sampling_rate']
print(f"  Original audio: shape={audio_array.shape}, sr={sr}")

audio_tensor = torch.from_numpy(audio_array).float()

try:
    # Save original audio
    import soundfile as sf
    original_fn = "xcodec2_original.wav"
    sf.write(original_fn, audio_array, sr)
    print(f"  Saved original audio to: {original_fn}")
    
    # Encode
    print("  Encoding...")
    tokens, encode_info = tokenizer.encode(audio_tensor, sr=sr)
    print(f"  Tokens: shape={tokens.shape}, num_tokens={tokens.numel()}")
    
    # Decode
    print("  Decoding...")
    reconstructed, decode_info = tokenizer.decode(tokens)
    print(f"  Reconstructed: shape={reconstructed.shape}")

    out_fn = "xcodec2_reconstructed.wav"
    # Handle shape: (1, 1, T) or (1, T) or (T,)
    rec = reconstructed
    if rec.dim() == 3:
        rec = rec[0, 0]
    elif rec.dim() == 2:
        rec = rec[0]
    rec_np = rec.detach().cpu().numpy()
    sf.write(out_fn, rec_np, decode_info.get('output_sample_rate', sr))
    print(f"  Saved reconstructed audio to: {out_fn}")
    
    print("\n✅ SUCCESS! XCodec2 tokenizer works!")
    print(f"   Encode time: {encode_info['encode_time']:.3f}s")
    print(f"   Decode time: {decode_info['decode_time']:.3f}s")

    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()