#!/bin/bash
#SBATCH --account=infra01
#SBATCH --environment=./ngc-24.11.toml
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
# Then watch logs/<jobid>_vllm_voxcpm2.out for the endpoint URL. Set
# VLLM_ENDPOINT env var to that URL in client jobs.
#
# CAUTION: This is a spike. Two unknowns to verify on first run:
#   1. Entry-point module name. vllm-omni's actual server entry-point
#      is undocumented for VoxCPM2 specifically (docs page 404). Best
#      guess below; if it ImportErrors, check `python -m vllm_omni
#      --help` and `pip show vllm-omni` for the right module.
#   2. Request/response format. The TTS API likely differs from
#      OpenAI's chat completion. See voxcpm2_vllm_tts.py for the
#      client-side assumption that has to match.

set -e
ulimit -c 0
cd ~/benchmark-audio-tokenizer
echo "Job $SLURM_JOB_ID started at $(date)"

PORT="${PORT:-8000}"
MODEL="${MODEL:-openbmb/VoxCPM2}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

# ── venv build (flock-protected, matches other recipes) ───────────────
LOCKFILE=".venv-vllm-omni.lock"
exec 200>"$LOCKFILE"
echo "=== Acquiring build lock for vllm-omni ==="
flock 200 || { echo "ERROR: failed to acquire build lock"; exit 1; }

if [ ! -f ".venv-vllm-omni/.build_complete" ]; then
    echo "=== Building vllm-omni venv (FIRST RUN — ~30-60 min CUDA compile) ==="
    if [ -e ".venv-vllm-omni" ]; then
        STALE=".venv-vllm-omni.stale.$$"
        mv ".venv-vllm-omni" "$STALE" || true
        rm -rf "$STALE" 2>/dev/null &
    fi
    make vllm-omni
else
    echo "=== Reusing existing vllm-omni venv ==="
fi
flock -u 200
source ".venv-vllm-omni/bin/activate"

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

# ── server ────────────────────────────────────────────────────────────
# UNVERIFIED ENTRY-POINT — see header CAUTION block.
# vllm-omni's release notes mention CUDA graphs and torch.compile, so
# expect first ~1 min of warm-up before /health flips green.
python -m vllm_omni.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --dtype bfloat16
