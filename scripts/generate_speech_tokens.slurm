#!/bin/bash
#SBATCH --account=infra01
#SBATCH --environment=./ngc-24.11.toml
#SBATCH --job-name=speech_tokens_poc
#SBATCH --output=logs/%j_speech_tokens_poc.out
#SBATCH --error=logs/%j_speech_tokens_poc.err
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu
#SBATCH --partition=normal

set -e  # exit on first error
ulimit -c 0
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd ~/benchmark-audio-tokenizer
echo "Job $SLURM_JOB_ID started at $(date)"

# ── Step 1: Create cosyvoice2 venv if missing ────────────────────────
if [ ! -d ".venv-cosyvoice2" ]; then
    echo "=== Creating cosyvoice2 venv ==="
    make cosyvoice2
fi

# ── Step 2: Activate venv and install extras ──────────────────────────
source .venv-cosyvoice2/bin/activate
pip install --quiet num2words

# ── Step 3: Extract Polish reference audio from FLEURS ────────────────
REF_DIR="outputs/reference_audio"
mkdir -p "$REF_DIR"

if [ ! -f "${REF_DIR}/pl_speaker_0.wav" ]; then
    echo "=== Extracting Polish reference audio ==="
    python -c "
from datasets import load_from_disk
import soundfile as sf
ds = load_from_disk('/capstor/store/cscs/swissai/infra01/audio-datasets/benchmark/fleurs_cache/pl_pl')
for i in range(3):
    sf.write('${REF_DIR}/pl_speaker_' + str(i) + '.wav', ds[i]['audio']['array'], ds[i]['audio']['sampling_rate'])
print('Saved 3 reference clips')
"
fi

# ── Step 4: Run speech token generation ───────────────────────────────
echo "=== Generating speech tokens ==="
python scripts/generate_speech_tokens.py \
    --reference-audio ${REF_DIR}/pl_speaker_0.wav ${REF_DIR}/pl_speaker_1.wav ${REF_DIR}/pl_speaker_2.wav \
    --output-dir outputs/speech_tokens \
    --render-audio \
    --device cuda \
    --checkpoint iic/CosyVoice2-0.5B

echo "Job $SLURM_JOB_ID finished at $(date)"
