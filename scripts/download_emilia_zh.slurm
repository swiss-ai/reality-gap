#!/bin/bash
#SBATCH --account=infra01
#SBATCH --environment=./ngc-24.11.toml
#SBATCH --job-name=dl_emilia_zh
#SBATCH --output=logs/%j_dl_emilia_zh.out
#SBATCH --error=logs/%j_dl_emilia_zh.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=normal

set -e
cd ~/benchmark-audio-tokenizer
echo "Job $SLURM_JOB_ID started at $(date)"

# Defaults can be overridden via --export=ALL,VAR=val
MAX_FILES="${MAX_FILES:-50}"
MAX_GB="${MAX_GB:-200}"
OUTPUT_DIR="${OUTPUT_DIR:-/capstor/scratch/cscs/$USER/emilia_zh_extra/Emilia-YODAS/ZH}"

source .venv-voxcpm2/bin/activate

# huggingface_hub already in voxcpm2 venv; if not, install on the fly.
python -c "import huggingface_hub" 2>/dev/null || pip install --quiet --no-deps huggingface_hub

python scripts/download_emilia_zh.py \
    --output-dir "$OUTPUT_DIR" \
    --max-files "$MAX_FILES" \
    --max-gb "$MAX_GB"

echo "Done at $(date)"
echo "Downloaded files:"
ls -la "$OUTPUT_DIR" | head -20
du -sh "$OUTPUT_DIR"
