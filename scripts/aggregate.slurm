#!/bin/bash
#SBATCH --account=infra01
#SBATCH --environment=./ngc-24.11.toml
#SBATCH --job-name=tts_aggregate
#SBATCH --output=logs/%j_aggregate.out
#SBATCH --error=logs/%j_aggregate.err
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --partition=normal

set -e
cd ~/benchmark-audio-tokenizer
source .venv-cosyvoice2/bin/activate
python scripts/benchmark_tts.py aggregate --output-dir results/tts_bench
