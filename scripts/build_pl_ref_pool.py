#!/usr/bin/env python3
"""Build a K=50 multi-speaker reference pool for Polish from Common Voice pl.

CV pl is appropriate for synth→STT training (Mozilla's stated intent). The TSV
gives us client_id (anonymized speaker), gender, and a per-clip duration table.

Methodology:
  1. Group validated.tsv rows by client_id.
  2. Quality filter: drop speakers with < --min-duration-sec total speech
     (most CV contributors only submit a few clips — keep only the ones who
     contributed enough audio to be statistically useful).
  3. Compute chars-per-second per speaker.
  4. Keep only speakers in the slowest --cps-percentile (preserves synth hours).
  5. Stratify by gender (male_masculine/female_feminine), sample K/2 each side
     with --seed. Skip gender-unspecified contributors.
  6. For each chosen speaker, pick the clip with duration closest to the
     speaker's median (avoid outlier-length clips that ramble or got truncated).
  7. Optionally force-include an anchor (e.g. an MLS spk1636 ref) for K=1
     baseline comparability.

Output: JSON pool spec (no audio extracted yet — that's step 2).

Stdlib-only — runs on Clariden login node.
"""

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def parse_clip_durations(path):
    """clip_durations.tsv: <clip_filename>\\t<duration_ms>. Returns {clip: sec}."""
    d = {}
    with open(path, newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            d[row["clip"]] = int(row["duration[ms]"]) / 1000.0
    return d


def parse_validated_tsv(path):
    """Yields row dicts from validated.tsv."""
    with open(path, newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            yield row


def normalize_gender(g):
    g = (g or "").lower()
    if "male_masc" in g or g == "male":
        return "M"
    if "female_fem" in g or g == "female":
        return "F"
    return "?"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--validated-tsv", required=True, type=Path,
                   help="CV pl validated.tsv")
    p.add_argument("--clip-durations", required=True, type=Path,
                   help="CV pl clip_durations.tsv")
    p.add_argument("--clips-archive", required=True, type=Path,
                   help="Path to validated_clips.tar.zst (for downstream extract)")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--cps-percentile", type=int, default=30,
                   help="Keep only speakers in slowest X percentile (PL: stricter than ZH)")
    p.add_argument("--min-duration-sec", type=float, default=600.0,
                   help="Drop speakers with < this much total speech (default 10 min)")
    p.add_argument("--anchor-wav", default=None,
                   help="Optional pre-existing ref wav to force-include (e.g. MLS spk1636 K=1 anchor)")
    p.add_argument("--anchor-text", default=None,
                   help="Transcript for the anchor wav")
    p.add_argument("--anchor-id", default="anchor",
                   help="Speaker id label for the anchor")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--language", default="pl")
    args = p.parse_args()

    print(f"[durations] reading {args.clip_durations}")
    durations = parse_clip_durations(args.clip_durations)
    print(f"[durations] {len(durations):,} clips with durations")

    # Per-speaker aggregation.
    spk_stats = defaultdict(lambda: {
        "clips": [],  # list of (clip_filename, duration, text)
        "gender_votes": defaultdict(int),
        "ages": defaultdict(int),
    })
    print(f"[validated] reading {args.validated_tsv}")
    n_rows = 0
    for row in parse_validated_tsv(args.validated_tsv):
        n_rows += 1
        spk = row["client_id"]
        clip = row["path"]
        text = row.get("sentence", "")
        dur = durations.get(clip)
        if dur is None or dur <= 0:
            continue
        g = normalize_gender(row.get("gender", ""))
        spk_stats[spk]["clips"].append((clip, dur, text))
        spk_stats[spk]["gender_votes"][g] += 1
        spk_stats[spk]["ages"][row.get("age", "")] += 1
    print(f"[validated] read {n_rows:,} rows → {len(spk_stats):,} unique speakers")

    # Reduce to one summary per speaker.
    summary = {}
    for spk, s in spk_stats.items():
        total_dur = sum(c[1] for c in s["clips"])
        if total_dur < args.min_duration_sec:
            continue
        total_chars = sum(len(c[2]) for c in s["clips"])
        # Pick the dominant gender (most-voted across this speaker's clips).
        best_g = max(s["gender_votes"].items(), key=lambda kv: kv[1])[0]
        summary[spk] = {
            "n_clips": len(s["clips"]),
            "total_dur": total_dur,
            "total_chars": total_chars,
            "cps": total_chars / total_dur,
            "gender": best_g,
            "clips": s["clips"],
        }
    print(f"[quality] {len(summary)} speakers passed {args.min_duration_sec}s floor")

    if not summary:
        raise SystemExit("No speakers passed quality filter.")

    cps_values = sorted(s["cps"] for s in summary.values())
    cutoff_idx = int(len(cps_values) * args.cps_percentile / 100)
    cps_cutoff = cps_values[cutoff_idx] if cutoff_idx < len(cps_values) else cps_values[-1]
    slow_pool = {spk: s for spk, s in summary.items() if s["cps"] <= cps_cutoff}
    print(f"[cadence] slowest {args.cps_percentile}%: {len(slow_pool)} speakers (cps ≤ {cps_cutoff:.2f})")
    print(f"  cps range overall: {cps_values[0]:.2f} – {cps_values[-1]:.2f}")
    if slow_pool:
        slow_cps = [s["cps"] for s in slow_pool.values()]
        print(f"  cps range slow pool: {min(slow_cps):.2f} – {max(slow_cps):.2f}")

    selected = {}
    n_anchor = 0
    if args.anchor_wav:
        # Anchor is a pre-existing wav, not from CV. Add directly to output.
        n_anchor = 1
        print(f"[anchor] force-including {args.anchor_id} from {args.anchor_wav}")

    remaining_k = args.k - n_anchor
    target_per_gender = remaining_k // 2

    rng = random.Random(args.seed)
    candidates_m = sorted([spk for spk, s in slow_pool.items() if s["gender"] == "M"])
    candidates_f = sorted([spk for spk, s in slow_pool.items() if s["gender"] == "F"])
    rng.shuffle(candidates_m)
    rng.shuffle(candidates_f)
    pick_m = candidates_m[:target_per_gender + (remaining_k % 2)]
    pick_f = candidates_f[:target_per_gender]
    for spk in pick_m + pick_f:
        selected[spk] = slow_pool[spk]
    print(f"[sample] picked {len(pick_m)}M + {len(pick_f)}F from slow pool")

    if len(selected) + n_anchor < args.k:
        backfill_pool = sorted(set(slow_pool) - set(selected))
        rng.shuffle(backfill_pool)
        for spk in backfill_pool:
            if len(selected) + n_anchor >= args.k:
                break
            selected[spk] = slow_pool[spk]
        print(f"[backfill] {len(selected) + n_anchor} after gender-imbalanced backfill")

    # Build output speaker list.
    out_speakers = []
    if args.anchor_wav:
        out_speakers.append({
            "spk_id": args.anchor_id,
            "gender": "?",
            "cps": None,
            "n_clips": 1,
            "total_dur": None,
            "ref_clip_name": Path(args.anchor_wav).name,
            "ref_wav": args.anchor_wav,
            "ref_text": args.anchor_text or "",
            "ref_duration": None,
            "source": "anchor",
        })

    for spk, s in selected.items():
        durs = sorted(c[1] for c in s["clips"])
        median_dur = durs[len(durs) // 2]
        best = min(s["clips"], key=lambda c: abs(c[1] - median_dur))
        out_speakers.append({
            "spk_id": spk[:16],  # truncate the 128-char hash for log readability
            "spk_id_full": spk,
            "gender": s["gender"],
            "cps": round(s["cps"], 3),
            "n_clips": s["n_clips"],
            "total_dur": round(s["total_dur"], 1),
            "ref_clip_name": best[0],
            "ref_wav": None,  # filled in by step 2 (extract_pl_refs.py)
            "ref_text": best[2],
            "ref_duration": best[1],
            "source": "cv24_pl_validated",
        })

    out_speakers.sort(key=lambda x: (x["source"], x["gender"], x["cps"] or 0))

    output_pool = {
        "ref_pool_id": args.output.stem,
        "language": args.language,
        "k": len(out_speakers),
        "clips_archive": str(args.clips_archive),
        "selection": {
            "seed": args.seed,
            "cps_percentile": args.cps_percentile,
            "min_duration_sec": args.min_duration_sec,
            "anchor": args.anchor_id if args.anchor_wav else None,
        },
        "stats": {
            "n_speakers_total": len(spk_stats),
            "n_passed_quality": len(summary),
            "n_slow_pool": len(slow_pool),
            "cps_cutoff": round(cps_cutoff, 3),
            "n_m": sum(1 for s in out_speakers if s["gender"] == "M"),
            "n_f": sum(1 for s in out_speakers if s["gender"] == "F"),
            "n_other_gender": sum(1 for s in out_speakers if s["gender"] not in ("M", "F")),
            "mean_cps": round(statistics.mean(s["cps"] for s in out_speakers if s["cps"]), 3) if any(s["cps"] for s in out_speakers) else None,
        },
        "speakers": out_speakers,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_pool, f, indent=2, ensure_ascii=False)

    print(json.dumps(output_pool["stats"], indent=2))
    print(f"[out] {args.output}")
    print("[next] run scripts/extract_pl_refs.py to materialize the 50 wavs")


if __name__ == "__main__":
    main()
