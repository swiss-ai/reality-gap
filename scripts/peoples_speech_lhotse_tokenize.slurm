#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=ps-tokenize
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/benchmark-audio-tokenizer/logs/ps_tokenize_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/benchmark-audio-tokenizer/logs/ps_tokenize_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --exclusive

set -e

echo "=========================================="
echo "People's Speech Lhotse Tokenization"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "GPUs per node: 4"
echo "=========================================="

mkdir -p /iopsstor/scratch/cscs/xyixuan/benchmark-audio-tokenizer/logs

NODES=($(scontrol show hostnames ${SLURM_JOB_NODELIST}))
HEAD_NODE=${NODES[0]}
MASTER_PORT=29500
TOTAL_GPUS=$((SLURM_JOB_NUM_NODES * 4))

echo "Head node: ${HEAD_NODE}:${MASTER_PORT}"
echo "Total ranks: ${TOTAL_GPUS}"
echo ""

# srun spawns 1 task/node, torchrun forks 4 GPU workers per node
srun -N ${SLURM_JOB_NUM_NODES} --tasks-per-node=1 -u bash -c '
    set -e
    export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/benchmark-audio-tokenizer:$PYTHONPATH

    echo "Installing torchaudio on $(hostname -s)..."
    uv pip install --python /opt/venv/bin/python --no-deps --no-build-isolation \
        git+https://github.com/pytorch/audio.git@release/2.9

    cd /iopsstor/scratch/cscs/xyixuan/benchmark-audio-tokenizer

    torchrun \
        --nproc_per_node=4 \
        --nnodes='"${SLURM_JOB_NUM_NODES}"' \
        --node_rank=${SLURM_PROCID} \
        --master_addr='"${HEAD_NODE}"' \
        --master_port='"${MASTER_PORT}"' \
        -m audio_tokenization.tokenize \
        dataset=unsupervised_peoples_speech_lhotse \
        num_gpus='"${TOTAL_GPUS}"'
'

echo ""
echo "=========================================="
echo "Tokenization completed!"
echo "=========================================="
