#!/bin/bash
#SBATCH --account=infra01
#SBATCH --environment=./vllm-voxcpm2-colocated.toml
#SBATCH --job-name=synth_vllm_coloc
#SBATCH --output=logs/%A_%a_synth_vllm_coloc.out
#SBATCH --error=logs/%A_%a_synth_vllm_coloc.err
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu
#SBATCH --partition=normal

# Co-located vllm-omni server + synthesize_to_shar client in a single SLURM job.
# Server lives only as long as the client needs it — no orphan idle GPUs, no
# server-expired-before-client-ran timing mismatches.
#
# USAGE:
#   export BACKEND=voxcpm2_vllm
#   export INPUT=data/tts_bench/pl_50.json
#   export OUT_DIR=/capstor/scratch/cscs/mkwapniewska/synthetic-polish-voxcpm2/pl_50_coloc
#   export BATCH_SIZE=32
#   sbatch --export=ALL scripts/synthesize_vllm_colocated.slurm
#
# SCALE-OUT:
#   sbatch --array=0-15 --export=ALL scripts/synthesize_vllm_colocated.slurm
#   Each array task gets its own GPU node, its own server, its own shard
#   of INPUT. Throughput scales linearly.

set -e
ulimit -c 0
echo "Job $SLURM_JOB_ID (array task ${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}) started at $(date)"

BACKEND="${BACKEND:-voxcpm2_vllm}"
INPUT="${INPUT:-data/tts_bench/pl_50.json}"
OUT_NAME="${OUT_NAME:-$(basename "$INPUT" .json)}"
OUT_DIR="${OUT_DIR:-results/synthetic_shar/${BACKEND}/${OUT_NAME}}"
REF_WAV="${REF_WAV:-/users/$(whoami)/benchmark-audio-tokenizer/outputs/reference_audio/pl_speaker_0.wav}"
TARGET_SR="${TARGET_SR:-24000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SERVER_PORT="${SERVER_PORT:-8000}"
MODEL="${MODEL:-openbmb/VoxCPM2}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

# Sharding from SLURM array env (round-robin slice in the python script).
SHARD_IDX="${SLURM_ARRAY_TASK_ID:-0}"
NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"

REPO_ROOT="/users/$(whoami)/benchmark-audio-tokenizer"

# ── start vllm-omni server in background ──────────────────────────────
SERVER_LOG="logs/${SLURM_JOB_ID}_server.log"
mkdir -p logs
echo "=== Starting vllm-omni server (log: ${SERVER_LOG}) ==="
python -m vllm_omni.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --host 127.0.0.1 \
    --port "${SERVER_PORT}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --dtype bfloat16 \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "Server PID: ${SERVER_PID}"

# Make sure we kill the server on any exit (success, error, or signal).
cleanup() {
    echo "=== Cleanup: killing server PID ${SERVER_PID} ==="
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── wait for /health ──────────────────────────────────────────────────
ENDPOINT="http://127.0.0.1:${SERVER_PORT}"
echo "=== Waiting for ${ENDPOINT}/health (up to 5 min) ==="
for i in {1..60}; do
    if curl -fs "${ENDPOINT}/health" > /dev/null 2>&1; then
        echo "Server ready after ${i}*5 s"
        break
    fi
    # If the server process died, bail rather than waiting the full 5 min
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: server process died before /health came up"
        tail -50 "${SERVER_LOG}"
        exit 1
    fi
    sleep 5
done

if ! curl -fs "${ENDPOINT}/health" > /dev/null; then
    echo "ERROR: /health never returned 200 after 5 min"
    tail -50 "${SERVER_LOG}"
    exit 1
fi

# ── install client deps if missing ────────────────────────────────────
# The vllm container ships with torch/torchaudio/soundfile/requests already
# (vllm depends on them). lhotse is NOT in the container — install at
# runtime via pip --no-deps + small dep list.
python -c "import lhotse" 2>/dev/null || {
    echo "=== Installing lhotse client deps ==="
    pip install --quiet --no-deps lhotse cytoolz click intervaltree audioread tabulate
}

# ── synthesize ────────────────────────────────────────────────────────
echo "=== Synthesizing ${BACKEND} → ${OUT_DIR} (shard ${SHARD_IDX}/${NUM_SHARDS}, batch=${BATCH_SIZE}) ==="
mkdir -p "$(dirname "$OUT_DIR")"
python "${REPO_ROOT}/scripts/synthesize_to_shar.py" \
    --backend "${BACKEND}" \
    --input "${REPO_ROOT}/${INPUT}" \
    --output-shar "${OUT_DIR}" \
    --reference-audio "${REF_WAV}" \
    --target-sr "${TARGET_SR}" \
    --shard-idx "${SHARD_IDX}" \
    --num-shards "${NUM_SHARDS}" \
    --batch-size "${BATCH_SIZE}" \
    --vllm-endpoint "${ENDPOINT}" \
    --device cpu

# When sharded, each task writes to <OUT_DIR>/shard_NNNN/. Report stats.
if [ "${NUM_SHARDS}" -gt 1 ]; then
    SHARD_PATH=$(printf "%s/shard_%04d" "${OUT_DIR}" "${SHARD_IDX}")
else
    SHARD_PATH="${OUT_DIR}"
fi
echo "=== Output Shar contents (${SHARD_PATH}) ==="
ls -la "${SHARD_PATH}" 2>/dev/null || true
[ -f "${SHARD_PATH}/synthesis_stats.json" ] && cat "${SHARD_PATH}/synthesis_stats.json"

echo "Job $SLURM_JOB_ID (array task ${SHARD_IDX}/${NUM_SHARDS}) finished at $(date)"
# trap will kill the server on exit
