#!/bin/bash
#SBATCH --account=infra01
#SBATCH --environment=./ngc-24.11.toml
#SBATCH --job-name=shar_to_parquet
#SBATCH --output=logs/%j_shar_to_parquet.out
#SBATCH --error=logs/%j_shar_to_parquet.err
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=normal

# Convert all 4 Lhotse Shar dirs under DEST to per-set parquet files with
# tokenized text. CPU-only (no GPU needed).

set -e
cd ~/benchmark-audio-tokenizer
echo "Job $SLURM_JOB_ID started at $(date)"

# SRC = where the Lhotse Shar dirs live (our scratch, written by synthesize_to_shar)
# OUT = where the parquets should land (the supervisor's audio-datasets/synthesized
#       path by default, since that's where his tokenization pipeline picks them up).
SRC="${SRC:-/capstor/scratch/cscs/mkwapniewska/synthetic-polish-voxcpm2}"
OUT="${OUT:-/capstor/store/cscs/swissai/infra01/audio-datasets/synthesized/voxcpm2}"
SETS="${SETS:-pl_50 pl_fleurs pl_commonvoice pl_voxpopuli}"

# Reuse the voxcpm2 venv — already has lhotse (=> pyarrow).
source .venv-voxcpm2/bin/activate

mkdir -p "$OUT"
for set_name in $SETS; do
    shar_in="$SRC/$set_name"
    out_dir="$OUT/$set_name/data"
    echo "=== Converting $set_name → $out_dir ==="
    if [ ! -d "$shar_in" ]; then
        echo "SKIP: $shar_in not found"
        continue
    fi
    # Per-shard parquets (slopalspeech-style train-NNNNN-of-MMMMM.parquet).
    # Memory bounded by single shard (~150 MB) instead of full set (~8 GB).
    python scripts/shar_to_parquet.py \
        --shar-in "$shar_in" \
        --parquet-out-dir "$out_dir"
done

echo "=== Output ==="
ls -lh "$OUT"/*/data/*.parquet 2>/dev/null || echo "No parquet files produced"

echo "Job $SLURM_JOB_ID finished at $(date)"
