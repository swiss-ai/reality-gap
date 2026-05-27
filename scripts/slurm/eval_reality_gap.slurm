#!/bin/bash
# Run reality-gap eval for ONE checkpoint over the 4 PL/ZH public test sets.
#
# Usage:
#   sbatch --job-name=eval_<ckpt_tag> scripts/slurm/eval_reality_gap.slurm <model_path> <output_subdir>
#
# Example (run after supervisor publishes a checkpoint):
#   sbatch --job-name=eval_synth scripts/slurm/eval_reality_gap.slurm \
#     /capstor/.../MLLM/ablations/audio/audio-weight-1-phase-transition-v3 \
#     synth-trained
#   sbatch --job-name=eval_real scripts/slurm/eval_reality_gap.slurm \
#     /capstor/.../MLLM/ablations/audio/audio-weight-0-phase-transition-v3 \
#     real-trained
#
# Then compute the gap with score_inference.py:
#   for ds in fleurs_pl fleurs_zh cv25_pl cv25_zh; do
#     python scripts/score_inference.py \
#       --synth-trained results/inference_eval/synth-trained/$ds/*.json \
#       --real-trained  results/inference_eval/real-trained/$ds/*.json
#   done

#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --environment=nemo_25_11
#SBATCH --output=logs/eval_%x_%j.out
#SBATCH --error=logs/eval_%x_%j.err
#SBATCH --exclusive

set -euo pipefail

MODEL_PATH=${1:?model_path required}
OUT_SUBDIR=${2:?output_subdir required}

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MANIFEST="$REPO_DIR/configs/eval/reality_gap.json"
OUTPUT_ROOT="$REPO_DIR/results/inference_eval/$OUT_SUBDIR"

mkdir -p "$OUTPUT_ROOT" "$REPO_DIR/logs"

echo "=========================================="
echo "Reality-gap eval"
echo "  model:    $MODEL_PATH"
echo "  manifest: $MANIFEST"
echo "  output:   $OUTPUT_ROOT"
echo "  backend:  vllm"
echo "=========================================="

cd "$REPO_DIR"
python scripts/audio_inference.py \
    --model-path "$MODEL_PATH" \
    --manifest "$MANIFEST" \
    --output-root "$OUTPUT_ROOT" \
    --backend vllm \
    --skip-existing
