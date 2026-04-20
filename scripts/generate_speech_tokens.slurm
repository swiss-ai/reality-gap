#!/bin/bash
#SBATCH --account=infra01
#SBATCH --environment=./ngc-24.11.toml
#SBATCH --job-name=speech_tokens_poc
#SBATCH --output=logs/%j_speech_tokens_poc.out
#SBATCH --error=logs/%j_speech_tokens_poc.err
#SBATCH --time=01:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu
#SBATCH --partition=normal

ulimit -c 0
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "Job $SLURM_JOB_ID started at $(date)"

# Activate cosyvoice2 venv
source .venv-cosyvoice2/bin/activate

# Install num2words if not present
pip install --quiet num2words

# Extract a reference audio clip from FLEURS Polish split
REF_DIR="outputs/reference_audio"
mkdir -p "$REF_DIR"
python -c "
from datasets import load_from_disk
import soundfile as sf
ds = load_from_disk('/capstor/store/cscs/swissai/infra01/audio-datasets/fleurs_cache/pl_pl')
for i in range(3):
    sf.write('${REF_DIR}/pl_speaker_${i}.wav', ds[i]['audio']['array'], ds[i]['audio']['sampling_rate'])
print(f'Saved 3 reference clips to ${REF_DIR}/')
"

# Run speech token generation
python scripts/generate_speech_tokens.py \
    --reference-audio ${REF_DIR}/pl_speaker_0.wav ${REF_DIR}/pl_speaker_1.wav ${REF_DIR}/pl_speaker_2.wav \
    --output-dir outputs/speech_tokens \
    --render-audio \
    --device cuda \
    --checkpoint iic/CosyVoice2-0.5B

echo "Job $SLURM_JOB_ID finished at $(date)"
