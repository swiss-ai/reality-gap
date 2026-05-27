#!/bin/bash
# Stage the 4 public PL/ZH test sets into clean per-set dirs that
# audio_inference.py's --manifest mode can consume directly.
#
# CV25 needs preprocessing: raw CV25 stores audio_bytes as a flat binary
# column (no sampling_rate sibling). audio_inference.py expects a struct
# {bytes, sampling_rate}. We re-pack into our delivery schema via
# scripts/preprocess_cv25_test.py.
#
# FLEURS Arrow caches are used as-is (load_from_disk).
# For ZH we use cmn_hans_cn_full (945 rows); for PL only pl_pl exists (100 rows).
#
# Run via:
#   srun --account=infra01 --partition=debug --time=00:15:00 \
#        --nodes=1 --ntasks=1 --cpus-per-task=8 \
#        --environment=nemo_25_11 --job-name=eval_prep \
#        scripts/prep_eval_test_sets.sh

set -euo pipefail

STAGE_ROOT=${STAGE_ROOT:-/capstor/scratch/cscs/mkwapniewska/eval_test_sets}
DATASETS_ROOT=/capstor/store/cscs/swissai/infra01/audio-datasets
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

mkdir -p "$STAGE_ROOT"

echo "=== CV25 PL test → preprocessed parquet ==="
python "$REPO_DIR/scripts/preprocess_cv25_test.py" \
    --input "$DATASETS_ROOT/processed/commonvoice25/pl/test.parquet" \
    --output "$STAGE_ROOT/cv25_pl/test.parquet" \
    --language pl

echo
echo "=== CV25 ZH test → preprocessed parquet ==="
python "$REPO_DIR/scripts/preprocess_cv25_test.py" \
    --input "$DATASETS_ROOT/processed/commonvoice25/zh-CN/test.parquet" \
    --output "$STAGE_ROOT/cv25_zh/test.parquet" \
    --language zh

echo
echo "Staged eval test sets under: $STAGE_ROOT"
ls -la "$STAGE_ROOT"/*/ 2>/dev/null
