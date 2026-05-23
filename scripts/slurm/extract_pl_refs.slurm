#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --environment=/users/mkwapniewska/.edf/nemo_25_11.toml
#SBATCH --output=/capstor/scratch/cscs/mkwapniewska/logs/extract_pl_refs-%j.out
#
# Extract 43 PL ref wavs from CV pl validated_clips.tar.zst.
# REQUIRES: ACL on raw/commonvoice24/pl/ granting group:infra01:rX
#   (default ACL only grants infra01adm; supervisor needs to fix before this runs)
#
# Submit with --hold so it stays PENDING until you confirm ACL is fixed:
#   sbatch --hold scripts/slurm/extract_pl_refs.slurm
# Then release: scontrol release <jobid>

set -euo pipefail
cd "/users/$(whoami)/benchmark-audio-tokenizer"

python3 scripts/extract_pl_refs.py \
    --pool references/pl_K50_seed42_cps30.json \
    --out-dir references/pl_K50_seed42_cps30
