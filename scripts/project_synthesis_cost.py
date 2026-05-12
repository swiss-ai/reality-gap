#!/usr/bin/env python3
"""Project GPU-hours to synthesize a target volume of Polish audio.

Reads `results/tts_bench/comparison.json` (produced by
`benchmark_tts.py aggregate`) and uses each backend's aggregate RTF
to project compute cost for a target number of audio hours.

Usage:
    python scripts/project_synthesis_cost.py --hours 100
    python scripts/project_synthesis_cost.py --hours 1000 --gpus 8
    python scripts/project_synthesis_cost.py --hours 16089       # full VoxPopuli pl

If `--gpus N` is given, wall-clock projection assumes linear scaling — a
best-case ceiling. Actual scaling depends on per-backend batching support.
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="results/tts_bench",
                   help="Directory containing comparison.json")
    p.add_argument("--hours", type=float, required=True,
                   help="Target hours of synthetic audio to produce")
    p.add_argument("--gpus", type=int, default=1,
                   help="Number of GPUs (linear-scaling assumption)")
    p.add_argument("--commercial-only", action="store_true",
                   help="Skip reference-only (non-commercial) backends")
    args = p.parse_args()

    comp_path = Path(args.results_dir) / "comparison.json"
    if not comp_path.exists():
        raise SystemExit(f"Not found: {comp_path}. Run `aggregate` first.")

    payload = json.loads(comp_path.read_text(encoding="utf-8"))
    rows = payload.get("summary", payload) if isinstance(payload, dict) else payload

    target_seconds = args.hours * 3600.0

    print(
        f"\nProjecting {args.hours} h of Polish synthetic audio "
        f"on {args.gpus} GPU(s):\n"
    )
    print(
        f"  {'backend':<14} {'license':<22} {'comm':>5} "
        f"{'agg RTF':>8} {'GPU-h':>10} {'wall-h (≤)':>12}"
    )
    print("  " + "-" * 76)

    for r in rows:
        if args.commercial_only and not r.get("commercial_usable", True):
            continue
        rtf = r.get("aggregate_rtf") or r.get("rtf_mean")
        if rtf is None:
            continue
        gpu_seconds = target_seconds * rtf
        gpu_hours = gpu_seconds / 3600.0
        wall_hours = gpu_hours / max(args.gpus, 1)
        comm = "yes" if r.get("commercial_usable") else "no"
        lic = (r.get("license") or "Unknown")[:22]
        print(
            f"  {r['backend']:<14} {lic:<22} {comm:>5} "
            f"{rtf:>8.4f} {gpu_hours:>10.1f} {wall_hours:>12.1f}"
        )

    print()


if __name__ == "__main__":
    main()
