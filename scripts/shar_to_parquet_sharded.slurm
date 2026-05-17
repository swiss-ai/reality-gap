#!/bin/bash
#SBATCH --account=infra01
#SBATCH --environment=./ngc-24.11.toml
#SBATCH --job-name=shar2pq
#SBATCH --output=logs/%A_%a_shar2pq.out
#SBATCH --error=logs/%A_%a_shar2pq.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=normal

# Convert a sharded Lhotse Shar output (one shard per SLURM array task) to
# parquet shards matching the supervisor's audio-datasets/synthesized layout:
#
#   $SRC/shard_NNNN/                                    →
#   $OUT/<set_name>/data/train-NNNNN-of-NNNNN.parquet
#
# Each task is CPU-only (no GPU needed) — reads one shard's tar + cuts and
# writes one parquet. Schema matches batch_tok prepare_parquet_to_shar.py.
#
# Usage (assumes SLURM array index = shard index):
#   export SRC=/capstor/scratch/cscs/mkwapniewska/synthetic-polish-voxcpm2/pl_voxpopuli_1kh/slow_ref_nanovllm
#   export OUT_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/synthesized/voxcpm2/pl_voxpopuli_1kh/data
#   export TOTAL_SHARDS=64
#   sbatch --array=0-63 --export=ALL scripts/shar_to_parquet_sharded.slurm

set -e
cd ~/benchmark-audio-tokenizer
echo "Job $SLURM_JOB_ID (array task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}) started at $(date)"

SRC="${SRC:?must set SRC=<dir containing shard_NNNN/>}"
OUT_DIR="${OUT_DIR:?must set OUT_DIR=<target parquet dir>}"
TOTAL_SHARDS="${TOTAL_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"

SHARD_IDX="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_DIR=$(printf "%s/shard_%04d" "$SRC" "$SHARD_IDX")
OUT_FILE=$(printf "%s/train-%05d-of-%05d.parquet" "$OUT_DIR" "$SHARD_IDX" "$TOTAL_SHARDS")

# Reuse voxcpm2 venv — already has lhotse + pyarrow.
source .venv-voxcpm2/bin/activate

if [ ! -d "$SHARD_DIR" ]; then
    echo "ERROR: shard input dir missing: $SHARD_DIR"
    exit 2
fi
mkdir -p "$OUT_DIR"

echo "=== Converting shard ${SHARD_IDX} ==="
echo "  SRC  = $SHARD_DIR"
echo "  OUT  = $OUT_FILE"

python scripts/shar_to_parquet.py \
    --shar-in "$SHARD_DIR" \
    --parquet-out "$OUT_FILE"

ls -lh "$OUT_FILE"

# Self-verify — login nodes can't run python, so do it here where the
# container + voxcpm2 venv are active.
echo "=== Parquet sanity ==="
python - <<PY
import pyarrow.parquet as pq
t = pq.read_table("$OUT_FILE")
print(f"  rows           : {t.num_rows}")
print(f"  columns        : {t.column_names}")
print(f"  audio.bytes[0] : {len(t.column('audio')[0]['bytes'].as_py())} bytes")
print(f"  audio.sr[0]    : {t.column('audio')[0]['sampling_rate'].as_py()} Hz")
print(f"  id[0]          : {t.column('id')[0].as_py()}")
print(f"  text[0][:80]   : {t.column('text')[0].as_py()[:80]!r}")
print(f"  duration[0]    : {t.column('duration')[0].as_py():.2f}s")
PY

echo "Job $SLURM_JOB_ID (shard ${SHARD_IDX}/${TOTAL_SHARDS}) finished at $(date)"
