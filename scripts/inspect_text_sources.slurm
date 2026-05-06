#!/bin/bash
#SBATCH --account=infra01
#SBATCH --environment=./ngc-24.11.toml
#SBATCH --job-name=inspect_text
#SBATCH --output=logs/%j_inspect_text.out
#SBATCH --error=logs/%j_inspect_text.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=normal

set -e
cd ~/benchmark-audio-tokenizer
echo "Job $SLURM_JOB_ID started at $(date)"

# datasets ships with NGC 24.11 in some images but not all — install if missing
python -c "import datasets" 2>/dev/null || pip install --quiet datasets

python scripts/inspect_text_sources.py \
    --sample-size 500 --language pl \
    --sources \
        oasst:OpenAssistant/oasst1::train \
        fleurs_pl:/capstor/store/cscs/swissai/infra01/audio-datasets/benchmark/fleurs_cache/pl_pl::train

echo "Done at $(date). Report: results/text_sources_inspection/comparison.md"
