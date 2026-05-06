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
OUT_DIR="${OUT_DIR:-results/tts_bench}"
# VENV defaults to BACKEND; override when reusing another backend's venv
# (e.g. mms_tts has no Makefile target — it reuses cosyvoice2's venv).
VENV="${VENV:-$BACKEND}"

# ── venv ──────────────────────────────────────────────────────────────
if [ ! -d ".venv-${VENV}" ]; then
    echo "=== Creating ${VENV} venv ==="
    make "${VENV}"
fi
source ".venv-${VENV}/bin/activate"
pip install --quiet num2words "ruamel.yaml==0.18.6" "ruamel.yaml.clib==0.2.8"

# ── reference audio (Polish FLEURS) ───────────────────────────────────
REF_DIR="outputs/reference_audio"
mkdir -p "$REF_DIR"
if [ ! -f "${REF_DIR}/pl_speaker_0.wav" ] || [ ! -f "${REF_DIR}/pl_speaker_0.txt" ]; then
    echo "=== Extracting Polish reference audio + transcripts ==="
    python -c "
from datasets import load_from_disk
import soundfile as sf
ds = load_from_disk('/capstor/store/cscs/swissai/infra01/audio-datasets/benchmark/fleurs_cache/pl_pl')
for i in range(3):
    base = '${REF_DIR}/pl_speaker_' + str(i)
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
    --reference-audio ${REF_DIR}/pl_speaker_0.wav ${REF_DIR}/pl_speaker_1.wav ${REF_DIR}/pl_speaker_2.wav \
    --output-dir "${OUT_DIR}" \
    --device cuda

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
