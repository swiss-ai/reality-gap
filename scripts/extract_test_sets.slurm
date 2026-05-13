#!/bin/bash
#SBATCH --account=infra01
#SBATCH --environment=./ngc-24.11.toml
#SBATCH --job-name=extract_test_sets
#SBATCH --output=logs/%j_extract_test_sets.out
#SBATCH --error=logs/%j_extract_test_sets.err
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=normal

set -e
cd ~/benchmark-audio-tokenizer
echo "Job $SLURM_JOB_ID started at $(date)"

source .venv-cosyvoice2/bin/activate

ROOT=/capstor/store/cscs/swissai/infra01/audio-datasets
N=100

mkdir -p data/tts_bench

# FLEURS — Wikipedia-style, standard benchmark
python scripts/extract_tts_test_set.py \
    --source "$ROOT/benchmark/fleurs_cache/pl_pl" \
    --n "$N" \
    --output data/tts_bench/pl_fleurs.json \
    --category fleurs \
    --min-words 3 --max-words 60

# Common Voice — short sentences, clean
python scripts/extract_tts_test_set.py \
    --source "$ROOT/SHAR/stage_2/commonvoice/pl_train" \
    --n "$N" \
    --output data/tts_bench/pl_commonvoice.json \
    --category commonvoice \
    --min-words 3 --max-words 60

# VoxPopuli — long-form parliamentary stress test
python scripts/extract_tts_test_set.py \
    --source "$ROOT/SHAR/stage_2/voxpopuli_asr/pl" \
    --n "$N" \
    --output data/tts_bench/pl_voxpopuli.json \
    --category voxpopuli \
    --min-words 3 --max-words 60

echo
echo "Done at $(date)"
ls -la data/tts_bench/*.json
