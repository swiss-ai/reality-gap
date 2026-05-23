#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --environment=/users/mkwapniewska/.edf/nemo_25_11.toml
#SBATCH --output=/capstor/scratch/cscs/mkwapniewska/logs/zh_fw2_k50_orchestrator-%j.out
#
# Submit with --dependency=afterok:<extract_zh_fw2>:<zh_smoke>
#
# Augments zh_fineweb2_22kh.json with K=50 refs and fires synth+parquet
# chain. The ZH ref pool (references/zh_K50_seed42_cps70.json) is already
# built — only the manifest is gated on the upstream extract.

set -euo pipefail
cd "/users/$(whoami)/benchmark-audio-tokenizer"

POOL=references/zh_K50_seed42_cps70.json
REF_DEFAULT_WAV=outputs/reference_audio_zh/zh_speaker_0.wav
REF_DEFAULT_TXT=outputs/reference_audio_zh/zh_speaker_0.txt
SYNTH_ROOT=/capstor/scratch/cscs/mkwapniewska/synth
PARQ_ROOT=/capstor/scratch/cscs/mkwapniewska/parquet

SET=zh_fineweb2_22kh
SHARDS=128
WALLTIME=06:00:00

IN="data/manifests/${SET}.json"
OUT="data/manifests/${SET}_K50.json"

if [ ! -f "$IN" ]; then
  echo "[error] $IN not present — extract_zh_fw2 must complete first"
  exit 1
fi

echo "[augment] $IN → $OUT"
python3 scripts/augment_manifest_with_refs.py --manifest "$IN" --ref-pool "$POOL" --output "$OUT"

ARRAY_HI=$((SHARDS - 1))
EXP="ALL,INPUT=${OUT},OUT_DIR=${SYNTH_ROOT}/${SET}_K50,REF_WAV=${REF_DEFAULT_WAV},REF_TXT=${REF_DEFAULT_TXT}"
SYNTH_JOB=$(sbatch --parsable --array=0-${ARRAY_HI} --time=${WALLTIME} --export="$EXP" scripts/synthesize_to_shar_nanovllm.slurm)
echo "[chain] ${SET}_K50 synth (array=0-${ARRAY_HI}): $SYNTH_JOB"

PQ="ALL,SRC=${SYNTH_ROOT},OUT=${PARQ_ROOT},SETS=${SET}_K50"
PQ_JOB=$(sbatch --parsable --dependency=afterok:$SYNTH_JOB --export="$PQ" scripts/shar_to_parquet.slurm)
echo "[chain] ${SET}_K50 parquet: $PQ_JOB"

echo "[done] zh fineweb2 K=50 fan-out submitted"
