#!/bin/bash
#SBATCH --account=infra01
#SBATCH --environment=./ngc-24.11.toml
#SBATCH --job-name=synth_to_shar
#SBATCH --output=logs/%A_%a_synth_to_shar.out
#SBATCH --error=logs/%A_%a_synth_to_shar.err
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu
#SBATCH --partition=normal

# SLURM driver for scripts/synthesize_to_shar.py — the bridge script
# that turns a JSON text manifest into a Lhotse Shar dataset, ready for
# audio_tokenization/tokenize.py downstream.
#
# Array-aware: pass --array=0-N-1 to scale across N GPUs. The bridge
# script slices the input round-robin (items[shard_idx::num_shards]) so
# each task processes its own subset and writes to its own shard_NNNN/
# subdirectory — no inter-task coordination needed.
#
# Usage:
#   # single task (smoke):
#   sbatch --export=ALL,BACKEND=voxcpm2,INPUT=data/tts_bench/pl_voxpopuli.json \
#       scripts/synthesize_to_shar.slurm
#
#   # 32-way scale-out:
#   sbatch --array=0-31 --export=ALL,BACKEND=voxcpm2,INPUT=data/big_pl.json \
#       scripts/synthesize_to_shar.slurm

set -e
ulimit -c 0
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd ~/benchmark-audio-tokenizer
echo "Job $SLURM_JOB_ID (array task ${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}) started at $(date)"

BACKEND="${BACKEND:-voxcpm2}"
INPUT="${INPUT:-data/tts_bench/pl_voxpopuli.json}"
OUT_NAME="${OUT_NAME:-$(basename "$INPUT" .json)}"
OUT_DIR="${OUT_DIR:-results/synthetic_shar/${BACKEND}/${OUT_NAME}}"
REF_WAV="${REF_WAV:-outputs/reference_audio/pl_speaker_0.wav}"
TARGET_SR="${TARGET_SR:-24000}"

# Sharding from SLURM array env. Defaults (0,1) = non-array single task.
SHARD_IDX="${SLURM_ARRAY_TASK_ID:-0}"
NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"

# ── venv build (same flock pattern as benchmark_tts.slurm) ────────────
LOCKFILE=".venv-${BACKEND}.lock"
exec 200>"$LOCKFILE"
echo "=== Acquiring build lock for ${BACKEND} ==="
flock 200 || { echo "ERROR: failed to acquire build lock"; exit 1; }

if [ ! -f ".venv-${BACKEND}/.build_complete" ]; then
    echo "=== Building ${BACKEND} venv ==="
    if [ -e ".venv-${BACKEND}" ]; then
        STALE=".venv-${BACKEND}.stale.$$"
        mv ".venv-${BACKEND}" "$STALE" || true
        rm -rf "$STALE" 2>/dev/null &
    fi
    make "${BACKEND}"
else
    echo "=== Reusing existing ${BACKEND} venv ==="
fi
flock -u 200
source ".venv-${BACKEND}/bin/activate"

# ── sanity checks ─────────────────────────────────────────────────────
if [ ! -f "$INPUT" ]; then
    echo "ERROR: input manifest not found: $INPUT"
    exit 2
fi
if [ ! -f "$REF_WAV" ]; then
    echo "ERROR: reference audio not found: $REF_WAV"
    echo "Run benchmark_tts.slurm once first — it extracts Polish FLEURS clips."
    exit 2
fi
python -c "import lhotse" || {
    echo "ERROR: lhotse missing from .venv-${BACKEND}. Rebuild venv: rm .venv-${BACKEND}/.build_complete"
    exit 2
}

# ── synthesize ────────────────────────────────────────────────────────
echo "=== Synthesizing ${BACKEND} → ${OUT_DIR} (shard ${SHARD_IDX}/${NUM_SHARDS}) ==="
mkdir -p "$(dirname "$OUT_DIR")"
python scripts/synthesize_to_shar.py \
    --backend "${BACKEND}" \
    --input "${INPUT}" \
    --output-shar "${OUT_DIR}" \
    --reference-audio "${REF_WAV}" \
    --target-sr "${TARGET_SR}" \
    --shard-idx "${SHARD_IDX}" \
    --num-shards "${NUM_SHARDS}" \
    --device cuda

# When sharded, each task writes to <OUT_DIR>/shard_NNNN/. Report that
# task's stats. When non-sharded, output sits directly in OUT_DIR.
if [ "${NUM_SHARDS}" -gt 1 ]; then
    SHARD_PATH=$(printf "%s/shard_%04d" "${OUT_DIR}" "${SHARD_IDX}")
else
    SHARD_PATH="${OUT_DIR}"
fi
echo "=== Output Shar contents (${SHARD_PATH}) ==="
ls -la "${SHARD_PATH}" || true
[ -f "${SHARD_PATH}/synthesis_stats.json" ] && cat "${SHARD_PATH}/synthesis_stats.json"

echo "Job $SLURM_JOB_ID (array task ${SHARD_IDX}/${NUM_SHARDS}) finished at $(date)"
