#!/bin/bash
#SBATCH --account=infra01
#SBATCH --environment=./vllm-voxcpm2.toml
#SBATCH --job-name=vllm_voxcpm2
#SBATCH --output=logs/%j_vllm_voxcpm2.out
#SBATCH --error=logs/%j_vllm_voxcpm2.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu
#SBATCH --partition=normal

# Server launcher for Stage-3 batched VoxCPM2 via vllm-omni.
# Holds a GPU for up to 4h so client jobs (synthesize_to_shar.py with the
# vLLM backend) can hit the endpoint.
#
# USAGE:
#   sbatch scripts/launch_vllm_voxcpm2.slurm
#
# Then watch logs/<jobid>_vllm_voxcpm2.out for the endpoint URL.
#
# Container: vllm-voxcpm2-cuda13.sqsh — pre-built by swiss-ai infra with
# vllm-omni + VoxCPM2 + CUDA 13 + matching torch already installed. No venv
# build needed; the entry point lives inside the container at /opt.
#
# Note: this is a DIFFERENT environment file than ngc-24.11.toml — our
# regular synthesis/scoring jobs keep using NGC 24.11. Only the vLLM server
# uses this container.

set -e
ulimit -c 0
cd ~/benchmark-audio-tokenizer
echo "Job $SLURM_JOB_ID started at $(date)"

PORT="${PORT:-8000}"
MODEL="${MODEL:-openbmb/VoxCPM2}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

# Tell clients where to reach us. Clariden compute nodes share an
# internal network; the IB-routable IP is what other jobs hit.
HOST=$(hostname -i | awk '{print $1}')
echo
echo "======================================================================"
echo "  vllm-omni VoxCPM2 server starting"
echo "  Endpoint: http://${HOST}:${PORT}"
echo "  Model:    ${MODEL}"
echo "  Hostname: $(hostname)"
echo "  Job ID:   ${SLURM_JOB_ID}"
echo "======================================================================"
echo

# Entry point inside the pre-built container.
# UNVERIFIED on first run — if `python -m vllm_omni.entrypoints.openai.api_server`
# is wrong, look inside the container with:
#   srun --environment=./vllm-voxcpm2.toml --pty bash
#   python -c "import vllm_omni; print(vllm_omni.__file__)"
# and adjust below.
python -m vllm_omni.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --dtype bfloat16
