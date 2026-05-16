#!/bin/bash
#SBATCH --account=infra01
#SBATCH --environment=./nanovllm-voxcpm2-colocated.toml
#SBATCH --job-name=nanovllm_smoke
#SBATCH --output=logs/%j_nanovllm_smoke.out
#SBATCH --error=logs/%j_nanovllm_smoke.err
#SBATCH --time=00:45:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu
#SBATCH --partition=normal

# Run the nano-vllm-voxcpm concurrency smoke on pl_50.
# Container has nano-vllm-voxcpm + torch 2.11 + flash-attn at /opt/venv —
# no venv build needed. The colocated toml mounts /users so we can reach
# this repo at ~/benchmark-audio-tokenizer.
#
# Submit:
#   sbatch scripts/smoke_nanovllm.slurm
#
# Override sweep:
#   sbatch --export=ALL,BATCH_SIZES=1,8,32,64 scripts/smoke_nanovllm.slurm

set -e
ulimit -c 0

cd ~/benchmark-audio-tokenizer
echo "Job $SLURM_JOB_ID started at $(date) on $(hostname)"

BATCH_SIZES="${BATCH_SIZES:-1,4,16,32}"
INPUT="${INPUT:-data/tts_bench/pl_50.json}"
REF_WAV="${REF_WAV:-outputs/reference_audio_slow/pl_speaker_0.wav}"
REF_TXT="${REF_TXT:-outputs/reference_audio_slow/pl_speaker_0.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-results/nanovllm_smoke}"
MODEL_PATH="${MODEL_PATH:-/capstor/store/cscs/swissai/infra01/hf_models/models/openbmb/VoxCPM2}"

# Sanity-check inputs before burning GPU time
for f in "$INPUT" "$REF_WAV" "$REF_TXT"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required file missing: $f"
        exit 2
    fi
done
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: model dir missing: $MODEL_PATH"
    exit 2
fi

# Use the container's venv directly. /opt/venv is what the contributor
# installed nano-vllm-voxcpm into. No source-activate needed — explicit
# path is more robust under non-interactive SLURM.
PY=/opt/venv/bin/python

echo "=== Environment ==="
"$PY" --version
"$PY" -c "import nanovllm_voxcpm; print('nano-vllm-voxcpm', nanovllm_voxcpm.__version__)"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "=== Running smoke sweep: BATCH_SIZES=$BATCH_SIZES ==="
"$PY" scripts/smoke_nanovllm.py \
    --model-path "$MODEL_PATH" \
    --input "$INPUT" \
    --ref-wav "$REF_WAV" \
    --ref-txt "$REF_TXT" \
    --output-dir "$OUTPUT_DIR" \
    --batch-sizes "$BATCH_SIZES"

echo "=== Result summary ==="
cat "$OUTPUT_DIR/stats.json"
echo
echo "Sample WAVs: $OUTPUT_DIR/samples/"
ls -lh "$OUTPUT_DIR/samples/" 2>/dev/null | head

echo "Job $SLURM_JOB_ID finished at $(date)"
