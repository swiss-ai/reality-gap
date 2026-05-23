#!/usr/bin/env python3
"""Augment a synth manifest with per-item reference assignments.

Round-robin assign each manifest item to a speaker in the ref pool via
`hash(id) % K`. Stable across reruns: same input ID always lands on same speaker.

Stdlib-only — runs on Clariden login node directly.

Output schema (per item):
    {
      "id": "...",
      "text": "...",
      "language": "zh",
      "source_duration": 4.2,
      "ref_wav": "/.../SSB0668_0023.wav",
      "ref_text": "...",
      "ref_spk_id": "SSB0668"
    }

Usage:
    python3 scripts/augment_manifest_with_refs.py \\
        --manifest data/manifests/zh_wenetspeech_1kh.json \\
        --ref-pool references/zh_K50_seed42_cps50.json \\
        --output data/manifests/zh_wenetspeech_1kh_K50cps50.json
"""

import argparse
import hashlib
import json
from pathlib import Path


def stable_hash(s):
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:4], "big")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--ref-pool", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--max-items", type=int, default=None,
                   help="Truncate to first N items (for pilot)")
    args = p.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    with open(args.ref_pool) as f:
        pool = json.load(f)

    speakers = [s for s in pool["speakers"] if s.get("ref_wav")]
    dropped = len(pool["speakers"]) - len(speakers)
    K = len(speakers)
    print(f"[manifest] {len(manifest):,} items")
    print(f"[pool] K={K} usable speakers from {pool.get('ref_pool_id', '?')}"
          + (f" ({dropped} dropped: null ref_wav)" if dropped else ""))

    if args.max_items:
        manifest = manifest[:args.max_items]
        print(f"[trunc] kept first {len(manifest):,} items")

    # Round-robin via stable hash → reproducible mapping
    out_items = []
    spk_counts = {s["spk_id"]: 0 for s in speakers}
    for it in manifest:
        spk_idx = stable_hash(it["id"]) % K
        ref = speakers[spk_idx]
        out_items.append({
            **it,
            "ref_wav": ref["ref_wav"],
            "ref_text": ref["ref_text"],
            "ref_spk_id": ref["spk_id"],
        })
        spk_counts[ref["spk_id"]] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_items, f, ensure_ascii=False, indent=2)

    counts = sorted(spk_counts.values())
    print(f"[assign] items/speaker: min={counts[0]}, median={counts[len(counts)//2]}, max={counts[-1]}")
    print(f"[out] {args.output} — {len(out_items):,} items")


if __name__ == "__main__":
    main()
