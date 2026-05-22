#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --environment=nemo_25_11
#SBATCH --job-name=extract_zh_fw2
#SBATCH --output=logs/%j_extract_zh_fw2.out
#SBATCH --error=logs/%j_extract_zh_fw2.err

# Extract Mandarin text from FineWeb-2 cmn_Hani — 1 shard × 22 h cap.
# Feeds the 20k h ZH synth target (22 h × 1.09 expansion - 5% dropout ≈ 22 kh).
# cmn_Hani mixes SC/TC; we synthesize as-is. If listening flags TC artefacts,
# re-extract with OpenCC t2s pass and re-synth a subset.

set -euo pipefail
cd "/users/$(whoami)/benchmark-audio-tokenizer"
mkdir -p logs data/manifests

echo "=== Extract zh shard0 ==="
python3 scripts/extract_text_from_fineweb2.py \
    --shard /capstor/store/cscs/swissai/infra01/datasets/HuggingFaceFW/fineweb-2/data/cmn_Hani/train/000_00000.parquet \
    --language zh \
    --output data/manifests/zh_fineweb2_22kh.json \
    --max-hours 22

# Sanity report
python3 - <<'PY'
import json
out = "data/manifests/zh_fineweb2_22kh.json"
items = json.load(open(out))
total_h = sum(it["source_duration"] for it in items) / 3600.0
print(f"{len(items)} items, ~{total_h:.1f} h source estimate")
PY
