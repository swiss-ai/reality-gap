#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=ups-resume
#SBATCH --environment=nemo
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/benchmark-audio-tokenizer/logs/ups_resume_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/benchmark-audio-tokenizer/logs/ups_resume_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --exclusive

# Resume specific failed ranks from a previous multi-node run on a single node.
#
# Usage:
#   sbatch /iopsstor/scratch/cscs/xyixuan/benchmark-audio-tokenizer/scripts/resume_failed_ranks.slurm 160 139          # resume rank 139 of 160
#   sbatch /iopsstor/scratch/cscs/xyixuan/benchmark-audio-tokenizer/scripts/resume_failed_ranks.slurm 160 139 42 77    # resume ranks 139,42,77 (up to 4)
#
# Args:
#   $1              : original world_size (e.g. 160)
#   $2, $3, $4, $5  : failed rank IDs to resume (max 4, one per GPU)

set -e

ORIG_WORLD_SIZE=${1:?Usage: sbatch resume_failed_ranks.slurm <orig_world_size> <rank> [rank] [rank] [rank]}
shift
FAILED_RANKS=("$@")

if [ ${#FAILED_RANKS[@]} -eq 0 ]; then
    echo "ERROR: specify at least one failed rank"
    exit 1
fi
if [ ${#FAILED_RANKS[@]} -gt 4 ]; then
    echo "ERROR: max 4 ranks per node (got ${#FAILED_RANKS[@]})"
    exit 1
fi

echo "=========================================="
echo "Resume failed ranks"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Original world_size: ${ORIG_WORLD_SIZE}"
echo "Failed ranks: ${FAILED_RANKS[*]}"
echo "=========================================="

mkdir -p /iopsstor/scratch/cscs/xyixuan/benchmark-audio-tokenizer/logs

srun --kill-on-bad-exit=0 -u bash -c '
    set -e
    export LD_PRELOAD=/opt/venv/lib/python3.12/site-packages/ray/core/libjemalloc.so
    export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/dev/lhotse:/iopsstor/scratch/cscs/xyixuan/benchmark-audio-tokenizer:$PYTHONPATH

    # Install torchaudio once per node
    if [ "${SLURM_LOCALID}" = "0" ]; then
        uv pip install --python /opt/venv/bin/python --no-deps --no-build-isolation \
            git+https://github.com/pytorch/audio.git@release/2.9
        touch /tmp/.torchaudio_ready_${SLURM_JOB_ID}
    else
        while [ ! -f /tmp/.torchaudio_ready_${SLURM_JOB_ID} ]; do sleep 1; done
    fi

    # Map SLURM_LOCALID (0-3) to the failed rank list
    FAILED_RANKS=('"$(printf '%s ' "${FAILED_RANKS[@]}")"')
    LOCAL=${SLURM_LOCALID}

    if [ ${LOCAL} -ge ${#FAILED_RANKS[@]} ]; then
        echo "Local rank ${LOCAL}: no failed rank assigned, exiting."
        exit 0
    fi

    FAKE_RANK=${FAILED_RANKS[${LOCAL}]}

    # Override env vars so pipeline sees the original rank/world_size
    export SLURM_PROCID=${FAKE_RANK}
    export SLURM_NTASKS='"${ORIG_WORLD_SIZE}"'
    export SLURM_LOCALID=${LOCAL}

    echo "Local GPU ${LOCAL} -> resuming as rank ${FAKE_RANK}/${SLURM_NTASKS}"

    cd /iopsstor/scratch/cscs/xyixuan/benchmark-audio-tokenizer

    python -m audio_tokenization.tokenize \
        dataset=unsupervised_peoples_speech_lhotse \
        num_gpus='"${ORIG_WORLD_SIZE}"' \
        resume=true \
        wandb.name="ups_resume_rank${FAKE_RANK}" \
        wandb.tags="[ups,lhotse,resume]"
'

echo ""
echo "=========================================="
echo "Resume completed!"
echo "=========================================="
