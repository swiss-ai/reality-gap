#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --environment=nemo_25_11
#SBATCH --job-name=extract_pl_fw2
#SBATCH --output=logs/%j_extract_pl_fw2.out
#SBATCH --error=logs/%j_extract_pl_fw2.err

# Extract Polish text from FineWeb-2 — 2 shards × 14 h cap → 28 h source manifest.
# Feeds the 20k h synth target (28 h source × 0.85 expansion - 5% dropout ≈ 22 kh).

set -euo pipefail
cd "/users/$(whoami)/benchmark-audio-tokenizer"
mkdir -p logs data/manifests

SHARD_DIR=/capstor/store/cscs/swissai/infra01/datasets/HuggingFaceFW/fineweb-2/data/pol_Latn/train
OUT_DIR=data/manifests

echo "=== Extract pl shard0 ==="
python3 scripts/extract_text_from_fineweb2.py \
    --shard "$SHARD_DIR/000_00000.parquet" \
    --language pl --output "$OUT_DIR/pl_fineweb2_shard0.json" --max-hours 14

echo "=== Extract pl shard1 ==="
python3 scripts/extract_text_from_fineweb2.py \
    --shard "$SHARD_DIR/000_00001.parquet" \
    --language pl --output "$OUT_DIR/pl_fineweb2_shard1.json" --max-hours 14

echo "=== Merge ==="
python3 - <<'PY'
import json, glob
out = "data/manifests/pl_fineweb2_28kh.json"
items = []
for f in sorted(glob.glob("data/manifests/pl_fineweb2_shard*.json")):
    items.extend(json.load(open(f)))
total_h = sum(it["source_duration"] for it in items) / 3600.0
json.dump(items, open(out, "w"), ensure_ascii=False, indent=2)
print(f"merged {len(items)} items, ~{total_h:.1f} h source estimate → {out}")
PY
