#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --job-name=rsync_k50
#SBATCH --output=/capstor/scratch/cscs/mkwapniewska/logs/rsync_k50-%j.out
#
# Usage:
#   sbatch scripts/slurm/rsync_k50_to_delivery.slurm
# Or with subset:
#   SETS="zh_aishell3_K50 zh_cv25_K50" sbatch scripts/slurm/rsync_k50_to_delivery.slurm

set -euo pipefail

SRC=/capstor/scratch/cscs/mkwapniewska/parquet
DST=/capstor/store/cscs/swissai/infra01/audio-datasets/synthesized/voxcpm2

# Default to all K=50 sets if not specified.
SETS=${SETS:-$(cd $SRC && ls -d *_K50 2>/dev/null | tr '\n' ' ')}

echo "[rsync] SRC=$SRC"
echo "[rsync] DST=$DST"
echo "[rsync] SETS=$SETS"
echo "==="

for SET in $SETS; do
  SRC_DIR="$SRC/$SET"
  DST_DIR="$DST/$SET"
  if [ ! -d "$SRC_DIR" ]; then
    echo "[skip] $SRC_DIR not found"; continue
  fi
  echo ">>> $SET ($(du -sh $SRC_DIR | cut -f1))"
  mkdir -p "$DST_DIR"
  rsync -av --info=progress2 --human-readable "$SRC_DIR/" "$DST_DIR/"
  echo "<<< $SET done"
done

echo "==="
echo "[done] all rsync complete"
