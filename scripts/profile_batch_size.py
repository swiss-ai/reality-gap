#!/usr/bin/env python
"""Profile WavTokenizer batch sizes with Nsight.

Usage:
    # Profile single batch size
    nsys profile -t cuda,nvtx -o profile_bs8 python scripts/profile_batch_size.py --batch_size 8

    # Profile specific duration (seconds)
    nsys profile -t cuda,nvtx -o profile_bs8_dur30 python scripts/profile_batch_size.py --batch_size 8 --duration 30

    # Compare multiple batch sizes
    for bs in 1 2 4 8 16 32 64; do
        nsys profile -t cuda,nvtx -o profile_bs${bs} python scripts/profile_batch_size.py --batch_size $bs 2>/dev/null
    done

    # Compare multiple durations at fixed batch size
    for dur in 5 10 30 60 120 200; do
        nsys profile -t cuda,nvtx -o profile_bs4_dur${dur} python scripts/profile_batch_size.py --batch_size 4 --duration $dur 2>/dev/null
    done

    # View stats
    nsys stats profile_bs8.nsys-rep
"""

import argparse
import time

import torch
import torch.cuda.nvtx as nvtx


def _is_cuda_oom(exc: BaseException) -> bool:
    """Return True if exception indicates CUDA OOM."""
    cuda_oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if cuda_oom_type is not None and isinstance(exc, cuda_oom_type):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return "cuda out of memory" in msg or "out of memory" in msg
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_iterations', type=int, default=10)
    parser.add_argument('--duration', type=float, default=10.0, help='Audio duration in seconds')
    parser.add_argument('--sample_rate', type=int, default=24000, help='Audio sample rate in Hz')
    parser.add_argument(
        '--raise_on_oom',
        action='store_true',
        help='Re-raise CUDA OOM instead of reporting status=oom',
    )
    args = parser.parse_args()

    import sys
    sys.path.insert(0, '/iopsstor/scratch/cscs/xyixuan/benchmark-audio-tokenizer/src')
    from audio_tokenizers.implementations.wavtokenizer import WavTokenizer40

    if args.duration <= 0:
        raise ValueError(f"--duration must be > 0, got {args.duration}")
    if args.sample_rate <= 0:
        raise ValueError(f"--sample_rate must be > 0, got {args.sample_rate}")

    sample_rate = int(args.sample_rate)
    duration = float(args.duration)
    audio_len = int(sample_rate * duration)

    print(f"Batch size: {args.batch_size}")
    print(f"Audio: {duration}s @ {sample_rate}Hz ({audio_len} samples)")
    print(f"GPU: {torch.cuda.get_device_name()}")

    status = "ok"
    oom_phase = None
    completed_iterations = 0
    tokens = None
    start_time = time.time()

    try:
        # Init model
        with nvtx.range("init_model"):
            model = WavTokenizer40(device='cuda')

        # Create synthetic audio batch
        with nvtx.range("create_audio"):
            audio = torch.randn(args.batch_size, audio_len, device='cuda')

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
            try:
                with nvtx.range(f"iter_{i}"):
                    with torch.no_grad():
                        with nvtx.range("encode"):
                            tokens = model.encode_audio(audio)
                    torch.cuda.synchronize()
                completed_iterations = i + 1
            except Exception as exc:  # noqa: BLE001 - intentional OOM handling for profiling
                if _is_cuda_oom(exc):
                    status = "oom"
                    oom_phase = f"iter_{i}"
                    if args.raise_on_oom:
                        raise
                    torch.cuda.empty_cache()
                    break
                raise
    except Exception as exc:  # noqa: BLE001 - intentional OOM handling for profiling
        if _is_cuda_oom(exc):
            status = "oom"
            if oom_phase is None:
                oom_phase = "setup_or_warmup"
            if args.raise_on_oom:
                raise
            torch.cuda.empty_cache()
        else:
            raise

    elapsed_s = time.time() - start_time
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\nResults:")
    print(f"  Status: {status}")
    if oom_phase is not None:
        print(f"  OOM phase: {oom_phase}")
    print(f"  Completed iterations: {completed_iterations}/{args.num_iterations}")
    if tokens is not None:
        print(f"  Output shape: {tokens.shape}")
    print(f"  Peak memory: {peak_mem:.2f} GB")
    print(f"  Elapsed: {elapsed_s:.2f} s")

if __name__ == '__main__':
    main()
