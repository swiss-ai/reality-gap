#!/usr/bin/env python3
"""Build a K=50 multi-speaker reference pool from AISHELL-3.

Selection methodology (gender × cadence × quality):
  1. Group AISHELL-3 utterances by speaker (id prefix SSB0XXX).
  2. Quality filter: drop speakers with < --min-duration-sec total speech.
  3. Compute chars-per-second (cps) per speaker.
  4. Keep only speakers in the slowest --cps-percentile of the pool.
  5. Stratify by gender (from spk-info.txt) and sample K/2 each side with --seed.
  6. Force-include --anchor-spk (default SSB0668) regardless of sampling.
  7. For each chosen speaker, pick a reference utterance whose duration is
     closest to that speaker's median duration (not the longest — those ramble).

Stdlib-only — runs on Clariden login node directly.

Usage:
    python3 scripts/build_zh_ref_pool.py \
        --manifest data/manifests/zh_aishell3.json \
        --spk-info /capstor/store/cscs/swissai/infra01/audio-datasets/raw/aishell/aishell3/spk-info.txt \
        --wav-root /capstor/store/cscs/swissai/infra01/audio-datasets/raw/aishell/aishell3/train/wav \
        --cps-percentile 30 \
        --output references/zh_K50_seed42_cps30.json
"""

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def parse_spk_info(path):
    """AISHELL-3 spk-info.txt: tab-separated <spk_id>\\t<age>\\t<gender>\\t<accent>.
    Age groups: A=<14, B=14-25, C=26-40, D=>41. Gender: 'male'/'female'.
    """
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            spk = parts[0]
            age = parts[1]
            gender_field = parts[2].lower()
            gender = "M" if gender_field.startswith("m") else ("F" if gender_field.startswith("f") else "?")
            out[spk] = {"gender": gender, "age": age}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path,
                   help="zh_aishell3.json — items with id/text/source_duration")
    p.add_argument("--spk-info", required=True, type=Path,
                   help="AISHELL-3 spk-info.txt")
    p.add_argument("--wav-root", required=True, type=Path,
                   help="AISHELL-3 train/wav root (contains SSB0XXX/ subdirs)")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--cps-percentile", type=int, default=50,
                   help="Keep only speakers in slowest X percentile by cps")
    p.add_argument("--min-duration-sec", type=float, default=600.0,
                   help="Drop speakers with < this much total speech (default 10 min)")
    p.add_argument("--anchor-spk", default="SSB0668",
                   help="Force-include this speaker (K=1 baseline anchor)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--language", default="zh")
    args = p.parse_args()

    spk_meta = parse_spk_info(args.spk_info)
    print(f"[spk-info] {len(spk_meta)} speakers with gender labels")

    # Load manifest, group by speaker.
    with open(args.manifest) as f:
        items = json.load(f)
    print(f"[manifest] {len(items):,} items")

    by_spk = defaultdict(list)
    for it in items:
        # AISHELL-3 IDs: SSB<4digit-spk><4digit-utt>[-segIdx] → speaker = first 7 chars
        spk = it["id"][:7]
        by_spk[spk].append(it)
    print(f"[group] {len(by_spk)} unique speakers in manifest")

    # Quality filter + cps computation.
    spk_stats = {}
    for spk, its in by_spk.items():
        total_dur = sum(i["source_duration"] for i in its)
        total_chars = sum(len(i["text"]) for i in its)
        if total_dur < args.min_duration_sec:
            continue
        cps = total_chars / total_dur
        spk_stats[spk] = {
            "n_clips": len(its),
            "total_dur": total_dur,
            "total_chars": total_chars,
            "cps": cps,
            "gender": spk_meta.get(spk, {}).get("gender", "?"),
            "items": its,
        }
    print(f"[quality] {len(spk_stats)} speakers passed duration filter ({args.min_duration_sec}s)")

    # Cadence cutoff: slowest X percentile.
    cps_values = sorted(s["cps"] for s in spk_stats.values())
    if not cps_values:
        raise SystemExit("No speakers after quality filter")
    cutoff_idx = int(len(cps_values) * args.cps_percentile / 100)
    cps_cutoff = cps_values[cutoff_idx] if cutoff_idx < len(cps_values) else cps_values[-1]
    slow_pool = {spk: s for spk, s in spk_stats.items() if s["cps"] <= cps_cutoff}
    print(f"[cadence] kept slowest {args.cps_percentile}%: {len(slow_pool)} speakers (cps ≤ {cps_cutoff:.2f})")
    print(f"   cps range overall: {cps_values[0]:.2f} – {cps_values[-1]:.2f}")
    print(f"   cps range in slow pool: {min(s['cps'] for s in slow_pool.values()):.2f} – {max(s['cps'] for s in slow_pool.values()):.2f}")

    # Anchor: force-include if present in slow pool; if not, add it explicitly.
    selected = {}
    if args.anchor_spk in spk_stats:
        selected[args.anchor_spk] = spk_stats[args.anchor_spk]
        print(f"[anchor] included {args.anchor_spk} (cps={spk_stats[args.anchor_spk]['cps']:.2f}, gender={spk_stats[args.anchor_spk]['gender']})")
    else:
        print(f"[anchor] WARNING: {args.anchor_spk} not in spk_stats — skipping anchor")

    # Gender-stratified sampling of remaining K-1 slots.
    remaining_k = args.k - len(selected)
    target_per_gender = remaining_k // 2  # if odd, one extra goes to M

    rng = random.Random(args.seed)
    candidates_m = sorted([spk for spk, s in slow_pool.items() if s["gender"] == "M" and spk not in selected])
    candidates_f = sorted([spk for spk, s in slow_pool.items() if s["gender"] == "F" and spk not in selected])
    rng.shuffle(candidates_m)
    rng.shuffle(candidates_f)

    pick_m = candidates_m[:target_per_gender + (remaining_k % 2)]
    pick_f = candidates_f[:target_per_gender]
    for spk in pick_m + pick_f:
        selected[spk] = spk_stats[spk]
    print(f"[sample] picked {len(pick_m)}M + {len(pick_f)}F = {len(pick_m)+len(pick_f)} additional")

    if len(selected) < args.k:
        # Backfill from whichever gender still has candidates.
        backfill_pool = sorted(set(slow_pool) - set(selected))
        rng.shuffle(backfill_pool)
        for spk in backfill_pool:
            if len(selected) >= args.k:
                break
            selected[spk] = spk_stats[spk]
        print(f"[backfill] expanded to {len(selected)} after gender-imbalanced backfill")

    # Reference clip per speaker: closest-to-median duration.
    out_speakers = []
    for spk, s in selected.items():
        durs = sorted(i["source_duration"] for i in s["items"])
        median_dur = durs[len(durs) // 2]
        best = min(s["items"], key=lambda i: abs(i["source_duration"] - median_dur))
        # Strip Lhotse segment suffix "-N" from id to get base AISHELL-3 utt name
        base_utt = best["id"].rsplit("-", 1)[0] if "-" in best["id"] else best["id"]
        wav_path = args.wav_root / spk / f"{base_utt}.wav"
        out_speakers.append({
            "spk_id": spk,
            "gender": s["gender"],
            "cps": round(s["cps"], 3),
            "n_clips": s["n_clips"],
            "total_dur": round(s["total_dur"], 1),
            "ref_utt_id": best["id"],
            "ref_wav": str(wav_path),
            "ref_text": best["text"],
            "ref_duration": best["source_duration"],
        })

    # Sort for stable output.
    out_speakers.sort(key=lambda x: (x["gender"], x["cps"]))

    output_pool = {
        "ref_pool_id": args.output.stem,
        "language": args.language,
        "k": len(out_speakers),
        "selection": {
            "seed": args.seed,
            "cps_percentile": args.cps_percentile,
            "min_duration_sec": args.min_duration_sec,
            "anchor_spk": args.anchor_spk,
        },
        "stats": {
            "n_speakers_total": len(by_spk),
            "n_passed_quality": len(spk_stats),
            "n_slow_pool": len(slow_pool),
            "cps_cutoff": round(cps_cutoff, 3),
            "n_m": sum(1 for s in out_speakers if s["gender"] == "M"),
            "n_f": sum(1 for s in out_speakers if s["gender"] == "F"),
            "n_other_gender": sum(1 for s in out_speakers if s["gender"] not in ("M", "F")),
            "mean_cps": round(statistics.mean(s["cps"] for s in out_speakers), 3),
        },
        "speakers": out_speakers,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_pool, f, indent=2, ensure_ascii=False)

    print(json.dumps(output_pool["stats"], indent=2))
    print(f"[out] {args.output}")


if __name__ == "__main__":
    main()
