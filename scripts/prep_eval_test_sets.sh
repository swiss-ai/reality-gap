#!/bin/bash
# Stage the 4 public PL/ZH test sets into clean per-set dirs that
# audio_inference.py's --manifest mode can consume directly.
#
# audio_inference.py's load_parquet_dataset() globs '*.parquet' in the
# given dir — so the source dirs (which also contain dev/train/...) would
# accidentally include train data. We symlink ONLY the test parquet
# into a dedicated staging dir per set.
#
# FLEURS Arrow caches are used as-is (load_from_disk), no symlinking needed.

set -euo pipefail

STAGE_ROOT=${STAGE_ROOT:-/capstor/scratch/cscs/mkwapniewska/eval_test_sets}
DATASETS_ROOT=/capstor/store/cscs/swissai/infra01/audio-datasets

mkdir -p "$STAGE_ROOT"

# --- CV25 PL test ---
mkdir -p "$STAGE_ROOT/cv25_pl"
ln -sfn "$DATASETS_ROOT/processed/commonvoice25/pl/test.parquet" \
        "$STAGE_ROOT/cv25_pl/test.parquet"

# --- CV25 ZH-CN test ---
mkdir -p "$STAGE_ROOT/cv25_zh"
ln -sfn "$DATASETS_ROOT/processed/commonvoice25/zh-CN/test.parquet" \
        "$STAGE_ROOT/cv25_zh/test.parquet"

# --- FLEURS pl_pl + cmn_hans_cn (Arrow — no symlinking; manifest points at source) ---

echo "Staged eval test sets under: $STAGE_ROOT"
ls -la "$STAGE_ROOT"/*/ 2>/dev/null
