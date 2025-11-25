#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=vox-norm
#SBATCH --output=/users/mrieff/benchmark-audio-tokenizer/logs/voxpopuli_%A_%a.out
#SBATCH --error=/users/mrieff/benchmark-audio-tokenizer/logs/voxpopuli_%A_%a.err
#SBATCH --time=11:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=normal
#SBATCH --array=0-14

echo "[INFO] Starting job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Memory limit: $(ulimit -v)"

# Activate Conda
eval "$(conda shell.bash hook)"
conda activate voxpopuli

# Change to scripts folder
cd /users/mrieff/benchmark-audio-tokenizer/scripts

# Detect languages automatically
ROOT=/capstor/store/cscs/swissai/infra01/audio-datasets
LANGS=($(ls -d $ROOT/voxpopuli/raw_audios/*/ | xargs -n1 basename | grep -v '.tar'))
LANG=${LANGS[$SLURM_ARRAY_TASK_ID]}

echo "[INFO] Processing language: $LANG"

# Run normalization with much smaller batch size and streaming mode
python normalize.py --root $ROOT \
                    --out /capstor/store/cscs/swissai/infra01/audio-datasets/normalized-arrow \
                    --batch_size 10 \
                    --language $LANG \
                    --streaming

echo "[INFO] Completed language: $LANG"