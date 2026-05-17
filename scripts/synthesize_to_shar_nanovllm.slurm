#!/bin/bash
#SBATCH --account=infra01
#SBATCH --environment=./nanovllm-voxcpm2-colocated.toml
#SBATCH --job-name=synth_nanovllm
#SBATCH --output=logs/%A_%a_synth_nanovllm.out
#SBATCH --error=logs/%A_%a_synth_nanovllm.err
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu
#SBATCH --partition=normal

# Production scale-out for VoxCPM2 via nano-vllm-voxcpm. Counterpart to
# synthesize_to_shar.slurm (Direct-Python via NGC 24.11) — same script,
# different container + backend.
#
# Why this exists: nano-vllm-voxcpm batches concurrent in-flight generations
# (pl_50 smoke: wall_RTF 0.018 at B=32, ~10x over Direct-Python's 0.185).
# Use this instead of synthesize_to_shar.slurm for VoxCPM2 production work.
#
# Container: nanovllm-voxcpm2-cuda13.sqsh, ships nano-vllm-voxcpm + torch
# 2.11 + flash-attn at /opt/venv. lhotse/soundfile are NOT in the container
# — installed at runtime via pip --no-deps (same pattern as the omni
# colocated slurm).
#
# Usage:
#   # smoke (one shard, slice-as-if-64):
#   export INPUT=data/tts_bench/pl_voxpopuli_1kh.json
#   export OUT_DIR=/capstor/scratch/cscs/mkwapniewska/synthetic-polish-voxcpm2/pl_voxpopuli_1kh/slow_ref_nanovllm
#   export NUM_SHARDS=64
#   sbatch --array=0-0 --export=ALL scripts/synthesize_to_shar_nanovllm.slurm
#
#   # full 64-way scale-out:
#   sbatch --array=0-63 --export=ALL scripts/synthesize_to_shar_nanovllm.slurm

set -e
ulimit -c 0
echo "Job $SLURM_JOB_ID (array task ${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}) started at $(date) on $(hostname)"

BACKEND="voxcpm2_nanovllm"
INPUT="${INPUT:-data/tts_bench/pl_voxpopuli_1kh.json}"
OUT_NAME="${OUT_NAME:-$(basename "$INPUT" .json)}"
OUT_DIR="${OUT_DIR:-results/synthetic_shar/${BACKEND}/${OUT_NAME}}"
# Speaker 1636 is the current production ref (replaced 6892 after the
# parallel-session A/B; see feedback_voxcpm2_production memory).
REF_WAV="${REF_WAV:-outputs/reference_audio_spk1636/pl_speaker_0.wav}"
REF_TXT="${REF_TXT:-outputs/reference_audio_spk1636/pl_speaker_0.txt}"
TARGET_SR="${TARGET_SR:-24000}"
# Batch size matches the pl_50 smoke sweet spot. nano-vllm fuses up to
# max_num_seqs=512 in flight, but 32 captured the bulk of the speedup
# (wall_RTF 0.018) without saturating memory headroom.
BATCH_SIZE="${BATCH_SIZE:-32}"
MODEL_PATH="${MODEL_PATH:-/capstor/store/cscs/swissai/infra01/hf_models/models/openbmb/VoxCPM2}"

# Sharding: SHARD_IDX / NUM_SHARDS can be overridden via env (calibration
# pattern: --array=0-0 NUM_SHARDS=64 → run one shard's worth at full slicing).
SHARD_IDX="${SHARD_IDX:-${SLURM_ARRAY_TASK_ID:-0}}"
NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"

REPO_ROOT="/users/$(whoami)/benchmark-audio-tokenizer"
cd "$REPO_ROOT"

# Container's venv. Don't activate — explicit python path is more robust
# under non-interactive SLURM.
PY=/opt/venv/bin/python

# ── input sanity (cheap, before runtime install) ─────────────────────
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

# ── runtime install of client deps not in the container ──────────────
# nano-vllm container has nano-vllm-voxcpm + torch 2.11 + flash-attn but
# NOT lhotse / soundfile. Install with --no-deps to avoid torch fights.
# Probe with a deep import — `import lhotse` triggers the whole intervaltree
# → sortedcontainers chain, so a partial previous install gets re-completed
# instead of being treated as already-good.
"$PY" -c "from lhotse.shar import SharWriter; import soundfile, sortedcontainers" 2>/dev/null || {
    echo "=== Installing client deps (lhotse + soundfile + transitive) ==="
    # nano-vllm container is more minimal than the omni one — the omni
    # container shipped sortedcontainers/pyyaml/tqdm, this one doesn't.
    # Keep --no-deps so we don't pull in a conflicting torch.
    "$PY" -m pip install --quiet --no-deps \
        lhotse cytoolz toolz click intervaltree sortedcontainers \
        audioread tabulate soundfile pyyaml tqdm
}

# ── environment summary ──────────────────────────────────────────────
echo "=== Environment ==="
"$PY" --version
"$PY" -c "import nanovllm_voxcpm; print('nano-vllm-voxcpm', nanovllm_voxcpm.__version__)"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
"$PY" -c "import lhotse; print('lhotse', lhotse.__version__)"

# Reference transcript: read from file rather than passing the string on
# the command line — Polish transcripts can contain quotes / commas / etc.
REF_TEXT_STR=$(cat "$REF_TXT")

echo "=== Synthesizing (shard ${SHARD_IDX}/${NUM_SHARDS}, batch=${BATCH_SIZE}) ==="
echo "  INPUT     = $INPUT"
echo "  OUT_DIR   = $OUT_DIR"
echo "  REF_WAV   = $REF_WAV"
echo "  REF_TXT   = $REF_TXT  (len=${#REF_TEXT_STR})"
echo "  TARGET_SR = $TARGET_SR"

mkdir -p "$(dirname "$OUT_DIR")"

"$PY" scripts/synthesize_to_shar.py \
    --backend "$BACKEND" \
    --input "$INPUT" \
    --output-shar "$OUT_DIR" \
    --reference-audio "$REF_WAV" \
    --reference-text "$REF_TEXT_STR" \
    --checkpoint "$MODEL_PATH" \
    --target-sr "$TARGET_SR" \
    --shard-idx "$SHARD_IDX" \
    --num-shards "$NUM_SHARDS" \
    --batch-size "$BATCH_SIZE" \
    --device cuda

# Report shard stats
if [ "${NUM_SHARDS}" -gt 1 ]; then
    SHARD_PATH=$(printf "%s/shard_%04d" "${OUT_DIR}" "${SHARD_IDX}")
else
    SHARD_PATH="${OUT_DIR}"
fi
echo "=== Output Shar contents (${SHARD_PATH}) ==="
ls -la "${SHARD_PATH}" 2>/dev/null || true
[ -f "${SHARD_PATH}/synthesis_stats.json" ] && cat "${SHARD_PATH}/synthesis_stats.json"

echo "Job $SLURM_JOB_ID (array task ${SHARD_IDX}/${NUM_SHARDS}) finished at $(date)"
