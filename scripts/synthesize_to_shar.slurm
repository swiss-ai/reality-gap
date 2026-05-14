#!/bin/bash
#SBATCH --account=infra01
#SBATCH --environment=./ngc-24.11.toml
#SBATCH --job-name=synth_to_shar
#SBATCH --output=logs/%j_synth_to_shar.out
#SBATCH --error=logs/%j_synth_to_shar.err
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
# Usage (smoke test):
#   sbatch --export=ALL,BACKEND=voxcpm2,INPUT=data/tts_bench/pl_voxpopuli.json \
#       scripts/synthesize_to_shar.slurm

set -e
ulimit -c 0
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd ~/benchmark-audio-tokenizer
echo "Job $SLURM_JOB_ID started at $(date)"

BACKEND="${BACKEND:-voxcpm2}"
INPUT="${INPUT:-data/tts_bench/pl_voxpopuli.json}"
OUT_NAME="${OUT_NAME:-$(basename "$INPUT" .json)}"
OUT_DIR="${OUT_DIR:-results/synthetic_shar/${BACKEND}/${OUT_NAME}}"
REF_WAV="${REF_WAV:-outputs/reference_audio/pl_speaker_0.wav}"
TARGET_SR="${TARGET_SR:-24000}"

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
echo "=== Synthesizing ${BACKEND} → ${OUT_DIR} ==="
mkdir -p "$(dirname "$OUT_DIR")"
python scripts/synthesize_to_shar.py \
    --backend "${BACKEND}" \
    --input "${INPUT}" \
    --output-shar "${OUT_DIR}" \
    --reference-audio "${REF_WAV}" \
    --target-sr "${TARGET_SR}" \
    --device cuda

echo "=== Output Shar contents ==="
ls -la "${OUT_DIR}" || true
[ -f "${OUT_DIR}/synthesis_stats.json" ] && cat "${OUT_DIR}/synthesis_stats.json"

echo "Job $SLURM_JOB_ID finished at $(date)"
