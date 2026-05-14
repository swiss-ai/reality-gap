#!/bin/bash
#SBATCH --account=infra01
#SBATCH --environment=./ngc-24.11.toml
#SBATCH --job-name=shar_to_parquet
#SBATCH --output=logs/%j_shar_to_parquet.out
#SBATCH --error=logs/%j_shar_to_parquet.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=normal

# Convert all 4 Lhotse Shar dirs under DEST to per-set parquet files with
# tokenized text. CPU-only (no GPU needed).

set -e
cd ~/benchmark-audio-tokenizer
echo "Job $SLURM_JOB_ID started at $(date)"

DEST="${DEST:-/capstor/scratch/cscs/mkwapniewska/synthetic-polish-voxcpm2}"
SETS="${SETS:-pl_50 pl_fleurs pl_commonvoice pl_voxpopuli}"

# Reuse the voxcpm2 venv — already has transformers + lhotse (=> pyarrow).
source .venv-voxcpm2/bin/activate

for set_name in $SETS; do
    shar_in="$DEST/$set_name"
    out="$DEST/$set_name/$set_name.parquet"
    echo "=== Converting $set_name ==="
    if [ ! -d "$shar_in" ]; then
        echo "SKIP: $shar_in not found"
        continue
    fi
    python scripts/shar_to_parquet.py \
        --shar-in "$shar_in" \
        --parquet-out "$out"
done

echo "=== Output ==="
ls -lh "$DEST"/*/*.parquet 2>/dev/null || echo "No parquet files produced"

echo "Job $SLURM_JOB_ID finished at $(date)"
