#!/usr/bin/env python3
"""Sum audio hours across every synth set in the delivery dir.

For each subdir under /capstor/.../synthesized/voxcpm2/<set>/ that has a
data/ subdir of parquets, read all parquets' `duration` column and sum.
Prints a sorted per-set table + per-language totals + ASCII bar chart.

Bypasses the per-set manifest_summary files (some sets don't have them)
and goes straight to the ground truth (parquet durations).
"""
import argparse, glob, os
from pathlib import Path

import pyarrow.parquet as pq


DEFAULT_ROOT = "/capstor/store/cscs/swissai/infra01/audio-datasets/synthesized/voxcpm2"


def hours_of(set_dir: Path) -> tuple[int, float]:
    """Return (n_rows, total_seconds) summed across all data/*.parquet."""
    data = set_dir / "data"
    if not data.is_dir():
        return (0, 0.0)
    n_rows = 0
    total_s = 0.0
    for f in sorted(data.glob("*.parquet")):
        try:
            t = pq.read_table(f, columns=["duration"])
            n_rows += t.num_rows
            total_s += t["duration"].to_pandas().sum()
        except Exception as e:
            print(f"  WARN {f}: {e}")
    return n_rows, total_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out-csv", default=None,
                    help="Optional CSV output path")
    args = ap.parse_args()

    root = Path(args.root)
    sets = sorted([d for d in root.iterdir()
                   if d.is_dir() and not d.name.startswith("_")])
    print(f"Scanning {len(sets)} synth sets under {root}\n")

    rows = []
    for s in sets:
        n, sec = hours_of(s)
        if n == 0:
            continue
        lang = "pl" if s.name.startswith("pl_") else "zh" if s.name.startswith("zh_") else "?"
        rows.append({"set": s.name, "lang": lang,
                     "rows": n, "hours": sec / 3600})

    # Sort by hours desc within language
    rows.sort(key=lambda r: (r["lang"], -r["hours"]))

    # Print table
    print(f"{'set':<35} {'lang':<5} {'rows':>14} {'hours':>12}")
    print("-" * 70)
    for r in rows:
        print(f"{r['set']:<35} {r['lang']:<5} {r['rows']:>14,} {r['hours']:>12,.1f}")
    print("-" * 70)

    pl_h = sum(r["hours"] for r in rows if r["lang"] == "pl")
    zh_h = sum(r["hours"] for r in rows if r["lang"] == "zh")
    pl_n = sum(r["rows"] for r in rows if r["lang"] == "pl")
    zh_n = sum(r["rows"] for r in rows if r["lang"] == "zh")
    print(f"\nPL total: {pl_n:>14,} rows, {pl_h:>12,.1f} h")
    print(f"ZH total: {zh_n:>14,} rows, {zh_h:>12,.1f} h")
    print(f"GRAND:    {pl_n+zh_n:>14,} rows, {pl_h+zh_h:>12,.1f} h")

    # ASCII bar chart, sorted by hours overall
    print(f"\n{'='*70}")
    print("Hours per set (bar chart, log scale)")
    print("=" * 70)
    import math
    all_sorted = sorted(rows, key=lambda r: -r["hours"])
    max_h = max(r["hours"] for r in all_sorted)
    log_max = math.log10(max(max_h, 1))
    for r in all_sorted:
        bar_len = int(40 * (math.log10(max(r["hours"], 1)) / log_max))
        bar = "█" * bar_len
        tag = "PL" if r["lang"] == "pl" else "ZH"
        print(f"  {r['set']:<35} {tag} {bar} {r['hours']:,.1f}h")

    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["set", "lang", "rows", "hours"])
            w.writeheader()
            for r in rows:
                r2 = dict(r); r2["hours"] = round(r["hours"], 1)
                w.writerow(r2)
        print(f"\nWrote CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
