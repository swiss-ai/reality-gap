#!/bin/bash
#SBATCH --account=infra01
#SBATCH --environment=./ngc-24.11.toml
#SBATCH --job-name=verify_datasets
#SBATCH --output=logs/%j_verify_datasets.out
#SBATCH --error=logs/%j_verify_datasets.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=normal

set -e
cd ~/benchmark-audio-tokenizer
echo "Job $SLURM_JOB_ID started at $(date)"

source .venv-cosyvoice2/bin/activate
python -c "import datasets" 2>/dev/null || uv pip install -q datasets
python -c "import lhotse" 2>/dev/null || uv pip install -q lhotse

ROOT=/capstor/store/cscs/swissai/infra01/audio-datasets

for D in \
    $ROOT/benchmark/fleurs_cache/pl_pl \
    $ROOT/SHAR/stage_2/voxpopuli_asr/pl \
    $ROOT/SHAR/stage_2/mls/mls_polish_train \
    $ROOT/SHAR/stage_2/commonvoice/pl_train \
    $ROOT/SHAR/stage_2/granary_ytc_asr/pl ; do
  echo
  echo "================================================================"
  echo "Verifying: $D"
  echo "================================================================"
  python scripts/extract_tts_test_set.py --source "$D" --verify-only \
        --scan-limit 5000 --min-words 3 --max-words 60 \
        || echo "  [FAILED] $D"
done

echo
echo "Done at $(date)"
