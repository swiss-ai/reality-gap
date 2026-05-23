#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --environment=/users/mkwapniewska/.edf/nemo_25_11.toml
#SBATCH --output=/capstor/scratch/cscs/mkwapniewska/logs/voxpopuli_vad-%j.out

set -euo pipefail
cd "/users/$(whoami)/benchmark-audio-tokenizer"

python3 scripts/extract_vad_segments_voxpopuli.py \
  --vad-dir /capstor/store/cscs/swissai/infra01/audio-datasets/raw/voxpopuli/vad_results_merged/per_lang_year_dedup \
  --audio-root /capstor/store/cscs/swissai/infra01/audio-datasets/raw/voxpopuli/raw_audios \
  --language pl \
  --output /capstor/scratch/cscs/mkwapniewska/voxpopuli_pl_vad_all.jsonl \
  --min-duration 2.0 --max-duration 30.0
