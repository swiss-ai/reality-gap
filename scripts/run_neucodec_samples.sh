#!/bin/bash
#SBATCH --account=root
#SBATCH --job-name=neucodec_samples
#SBATCH --output=/users/mrieff/benchmark-audio-tokenizer/logs/neucodec_samples_%j.out
#SBATCH --error=/users/mrieff/benchmark-audio-tokenizer/logs/neucodec_samples_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=normal

# Set threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Activate your venv
source /users/mrieff/benchmark-audio-tokenizer/neucodec-venv/bin/activate

# Go to project root
cd /users/mrieff/benchmark-audio-tokenizer

# Run the Python script
python scripts/sample_neucodec_audio_pairs.py
