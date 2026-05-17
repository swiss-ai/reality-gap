#!/bin/bash
#SBATCH --account=infra01
#SBATCH --environment=./ngc-24.11.toml
#SBATCH --job-name=tts_bench_pl
#SBATCH --output=logs/%j_tts_bench_pl.out
#SBATCH --error=logs/%j_tts_bench_pl.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu
#SBATCH --partition=normal

set -e
ulimit -c 0
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd ~/benchmark-audio-tokenizer
echo "Job $SLURM_JOB_ID started at $(date)"

BACKEND="${BACKEND:-cosyvoice2}"
LANG="${LANG:-pl}"
SENTENCES="${SENTENCES:-data/tts_bench/pl_50.json}"
# Output dir defaults to results/tts_bench/<sentences_stem>/ so different test
# sets land in different directories and don't overwrite each other.
SET_NAME="$(basename "$SENTENCES" .json)"
OUT_DIR="${OUT_DIR:-results/tts_bench/${SET_NAME}}"
# VENV defaults to BACKEND; override when reusing another backend's venv
# (e.g. mms_tts has no Makefile target — it reuses cosyvoice2's venv).
VENV="${VENV:-$BACKEND}"

# ── venv ──────────────────────────────────────────────────────────────
# Concurrency-safe venv build:
#  - flock serializes builds when multiple jobs target the same backend;
#    other jobs wait at the lock, then find .build_complete and skip rebuild.
#  - Atomic `mv` (not `rm -rf`) for cleanup — NFS sometimes refuses to
#    remove partial-install dirs; rename is always atomic.
LOCKFILE=".venv-${VENV}.lock"
exec 200>"$LOCKFILE"
echo "=== Acquiring build lock for ${VENV} ==="
flock 200 || { echo "ERROR: failed to acquire build lock"; exit 1; }

if [ ! -f ".venv-${VENV}/.build_complete" ]; then
    echo "=== Creating ${VENV} venv (no build-complete marker) ==="
    if [ -e ".venv-${VENV}" ]; then
        STALE=".venv-${VENV}.stale.$$"
        mv ".venv-${VENV}" "$STALE" || true
        rm -rf "$STALE" 2>/dev/null &
    fi
    make "${VENV}"
else
    echo "=== Reusing existing ${VENV} venv ==="
fi
flock -u 200
source ".venv-${VENV}/bin/activate"
pip install --quiet num2words "ruamel.yaml==0.18.6" "ruamel.yaml.clib==0.2.8" 2>/dev/null || true

# ── reference audio (Polish FLEURS by default) ────────────────────────
# REF_DIR can be overridden via env to A/B different reference voices.
REF_DIR="${REF_DIR:-outputs/reference_audio}"
mkdir -p "$REF_DIR"
# Speaker file prefix lets the same slurm cover non-Polish runs (zh_speaker_*,
# etc.) without renaming files.
SPEAKER_PREFIX="${SPEAKER_PREFIX:-pl_speaker}"
if [ ! -f "${REF_DIR}/${SPEAKER_PREFIX}_0.wav" ] || [ ! -f "${REF_DIR}/${SPEAKER_PREFIX}_0.txt" ]; then
    echo "=== Extracting Polish reference audio + transcripts ==="
    python -c "
from datasets import load_from_disk
import soundfile as sf
ds = load_from_disk('/capstor/store/cscs/swissai/infra01/audio-datasets/benchmark/fleurs_cache/pl_pl')
for i in range(3):
    base = '${REF_DIR}/${SPEAKER_PREFIX}_' + str(i)
    sf.write(base + '.wav', ds[i]['audio']['array'], ds[i]['audio']['sampling_rate'])
    text = ds[i].get('transcription') or ds[i].get('text') or ''
    open(base + '.txt', 'w', encoding='utf-8').write(text)
print('Saved 3 reference clips + transcripts')
"
fi

# ── 1. generate ───────────────────────────────────────────────────────
echo "=== Generate (${BACKEND}, ${LANG}) ==="
python scripts/benchmark_tts.py generate \
    --backend "${BACKEND}" \
    --language "${LANG}" \
    --sentences-file "${SENTENCES}" \
    --reference-audio ${REF_DIR}/${SPEAKER_PREFIX}_0.wav ${REF_DIR}/${SPEAKER_PREFIX}_1.wav ${REF_DIR}/${SPEAKER_PREFIX}_2.wav \
    --output-dir "${OUT_DIR}" \
    --device cuda

# ── swap to shared scorer venv ────────────────────────────────────────
# Whisper + future metrics live in a single shared venv so we don't
# reinstall heavy deps (numba, llvmlite, tiktoken) into every backend's
# venv, and so backend pins can't fight Whisper's.
deactivate || true

SCORER_LOCK=".venv-scorer.lock"
exec 201>"$SCORER_LOCK"
echo "=== Acquiring build lock for scorer ==="
flock 201 || { echo "ERROR: failed to acquire scorer build lock"; exit 1; }
if [ ! -f ".venv-scorer/.build_complete" ]; then
    echo "=== Building scorer venv ==="
    if [ -e ".venv-scorer" ]; then
        STALE=".venv-scorer.stale.$$"
        mv ".venv-scorer" "$STALE" || true
        rm -rf "$STALE" 2>/dev/null &
    fi
    make scorer
else
    echo "=== Reusing existing scorer venv ==="
fi
flock -u 201
source ".venv-scorer/bin/activate"

# ── 2. score (Whisper WER) ────────────────────────────────────────────
echo "=== Score (Whisper) ==="
python scripts/benchmark_tts.py score \
    --output-dir "${OUT_DIR}" \
    --language "${LANG}" \
    --whisper-model large-v3 \
    --device cuda

# ── 3. aggregate ──────────────────────────────────────────────────────
echo "=== Aggregate ==="
python scripts/benchmark_tts.py aggregate --output-dir "${OUT_DIR}"

echo "Job $SLURM_JOB_ID finished at $(date)"
