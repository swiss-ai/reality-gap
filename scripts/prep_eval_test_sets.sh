#!/bin/bash
# Stage the public PL/ZH test sets into clean per-set dirs that
# audio_inference.py's --manifest --parquet-dir mode can consume.
#
# CV25 PL + ZH-CN test.parquets at /capstor/store/.../processed/commonvoice25/
# happen to already match the (struct<bytes, sampling_rate>) audio schema
# audio_inference.py expects (see incident note 2026-05-27 in the eval README).
# We just symlink them into per-set dirs so the loader's `*.parquet` glob
# doesn't accidentally pick up train/dev/etc. from the source directory.
#
# FLEURS Arrow caches are used as-is (load_from_disk on the parent path);
# no symlinking needed there.

set -euo pipefail

STAGE_ROOT=${STAGE_ROOT:-/capstor/scratch/cscs/mkwapniewska/eval_test_sets}
DATASETS_ROOT=/capstor/store/cscs/swissai/infra01/audio-datasets

mkdir -p "$STAGE_ROOT/cv25_pl" "$STAGE_ROOT/cv25_zh"

ln -sfn "$DATASETS_ROOT/processed/commonvoice25/pl/test.parquet" \
        "$STAGE_ROOT/cv25_pl/test.parquet"
ln -sfn "$DATASETS_ROOT/processed/commonvoice25/zh-CN/test.parquet" \
        "$STAGE_ROOT/cv25_zh/test.parquet"

echo "Staged eval test sets under: $STAGE_ROOT"
ls -la "$STAGE_ROOT"/cv25_pl/ "$STAGE_ROOT"/cv25_zh/
