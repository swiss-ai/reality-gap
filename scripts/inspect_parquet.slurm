#!/bin/bash
#SBATCH --account=infra01
#SBATCH --environment=./ngc-24.11.toml
#SBATCH --job-name=inspect_parquet
#SBATCH --output=logs/%j_inspect_parquet.out
#SBATCH --error=logs/%j_inspect_parquet.err
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=normal

# Inspect a parquet file's schema + first row.
# Override PARQUET=... at sbatch time to look at a different file.

set -e
cd ~/benchmark-audio-tokenizer
echo "Job $SLURM_JOB_ID started at $(date)"

PARQUET="${PARQUET:-/capstor/store/cscs/swissai/infra01/audio-datasets/processed/slopalspeech_no_overlap/data/train-00013-of-00123.parquet}"

source .venv-voxcpm2/bin/activate

python <<EOF
import pyarrow.parquet as pq

path = "$PARQUET"
print(f"=== File: {path} ===\n")

t = pq.read_table(path)
print("=== SCHEMA ===")
print(t.schema)
print(f"\nrows: {t.num_rows}")
print(f"columns: {t.column_names}")

print("\n=== ROW 0 ===")
r0 = t.slice(0, 1).to_pydict()
for k, v in r0.items():
    val = v[0]
    if isinstance(val, (bytes, bytearray)):
        print(f"  {k}: bytes, len={len(val)}, first 16 bytes={bytes(val[:16])!r}")
    elif isinstance(val, dict):
        print(f"  {k}: dict with keys={list(val.keys())}")
        for kk, vv in val.items():
            if isinstance(vv, (bytes, bytearray)):
                print(f"    .{kk}: bytes, len={len(vv)}")
            else:
                print(f"    .{kk}: {type(vv).__name__} = {str(vv)[:80]}")
    elif isinstance(val, list):
        first = val[0] if val else None
        print(f"  {k}: list[{type(first).__name__ if first is not None else '?'}], "
              f"len={len(val)}, first 5={val[:5]}")
    else:
        print(f"  {k}: {type(val).__name__} = {str(val)[:120]}")
EOF

echo "Job $SLURM_JOB_ID finished at $(date)"
