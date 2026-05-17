#!/bin/bash
#SBATCH --account=infra01
#SBATCH --environment=./ngc-24.11.toml
#SBATCH --job-name=make_backend
#SBATCH --output=logs/%j_make_%x.out
#SBATCH --error=logs/%j_make_%x.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=normal

# Wrapper for `make <backend>` so we don't need a long sbatch one-liner.
# Usage:
#   sbatch --export=ALL,BACKEND=cosyvoice2 --job-name=make_cosyvoice2 scripts/make_backend.slurm

set -e
cd ~/benchmark-audio-tokenizer
echo "Building venv for ${BACKEND} at $(date)"
make "${BACKEND}"
echo "Done building ${BACKEND} at $(date)"
