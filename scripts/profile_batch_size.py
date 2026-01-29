#!/usr/bin/env python
"""Profile WavTokenizer batch sizes with Nsight.

Usage:
    # Profile single batch size
    nsys profile -t cuda,nvtx -o profile_bs8 python scripts/profile_batch_size.py --batch_size 8

    # Compare multiple batch sizes
    for bs in 1 2 4 8 16 32 64; do
        nsys profile -t cuda,nvtx -o profile_bs${bs} python scripts/profile_batch_size.py --batch_size $bs 2>/dev/null
    done

    # View stats
    nsys stats profile_bs8.nsys-rep
"""

import argparse
import torch
import torch.cuda.nvtx as nvtx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_iterations', type=int, default=10)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, '/iopsstor/scratch/cscs/xyixuan/benchmark-audio-tokenizer/src')
    from audio_tokenizers.implementations.wavtokenizer import WavTokenizer40

    # Fixed: 10 second audio at 24kHz
    SAMPLE_RATE = 24000
    DURATION = 10.0
    AUDIO_LEN = int(SAMPLE_RATE * DURATION)

    print(f"Batch size: {args.batch_size}")
    print(f"Audio: {DURATION}s @ {SAMPLE_RATE}Hz ({AUDIO_LEN} samples)")
    print(f"GPU: {torch.cuda.get_device_name()}")

    # Init model
    with nvtx.range("init_model"):
        model = WavTokenizer40(device='cuda')

    # Create batch of 10-second audio
    with nvtx.range("create_audio"):
        audio = torch.randn(args.batch_size, AUDIO_LEN, device='cuda')

    # Warmup
    print("Warmup...")
    for _ in range(3):
        with torch.no_grad():
            _ = model.encode_audio(audio)
        torch.cuda.synchronize()

    # Clear memory stats
    torch.cuda.reset_peak_memory_stats()

    # Profile iterations
    print(f"Profiling {args.num_iterations} iterations...")
    for i in range(args.num_iterations):
        with nvtx.range(f"iter_{i}"):
            with torch.no_grad():
                with nvtx.range("encode"):
                    tokens = model.encode_audio(audio)
            torch.cuda.synchronize()

    # Results
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\nResults:")
    print(f"  Output shape: {tokens.shape}")
    print(f"  Peak memory: {peak_mem:.2f} GB")

if __name__ == '__main__':
    main()
