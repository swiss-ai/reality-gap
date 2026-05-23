#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --environment=/users/mkwapniewska/.edf/nemo_25_11.toml
#SBATCH --output=/capstor/scratch/cscs/mkwapniewska/logs/pl_k50_extra-%j.out
#
# Submit with --dependency=afterok:<extract_pl_refs_job>
#
# Companion to pl_k50_orchestrator.slurm — handles the manifests that were
# re-extracted on 2026-05-23 (granary_ytc, cv, voxpopuli_full). Does NOT
# fire a smoke test (the first orchestrator already does that); just augments
# + submits synth+parquet chains directly.

set -euo pipefail

# libjson-c.so.5 workaround for sbatch-from-container (pyxis SPANK plugin needs it).
export LD_LIBRARY_PATH="/capstor/scratch/cscs/${USER:-mkwapniewska}/libfix:${LD_LIBRARY_PATH:-}"

cd "/users/$(whoami)/benchmark-audio-tokenizer"

POOL=references/pl_K50_seed42_cps30.json
REF_DEFAULT_WAV=outputs/reference_audio_spk1636/pl_speaker_0.wav
REF_DEFAULT_TXT=outputs/reference_audio_spk1636/pl_speaker_0.txt
SYNTH_ROOT=/capstor/scratch/cscs/mkwapniewska/synth
PARQ_ROOT=/capstor/scratch/cscs/mkwapniewska/parquet

declare -A SHARDS=(
  [pl_granary_ytc]=1
  [pl_cv]=3
  [pl_voxpopuli_full]=768
)
declare -A WALLTIMES=(
  [pl_voxpopuli_full]=01:00:00
)

for SET in "${!SHARDS[@]}"; do
  IN="data/manifests/${SET}.json"
  OUT="data/manifests/${SET}_K50.json"
  if [ ! -f "$IN" ]; then
    echo "[skip] $IN not present"; continue
  fi
  echo "[augment] $IN → $OUT"
  python3 scripts/augment_manifest_with_refs.py --manifest "$IN" --ref-pool "$POOL" --output "$OUT"
done

for SET in "${!SHARDS[@]}"; do
  MAN="data/manifests/${SET}_K50.json"
  if [ ! -f "$MAN" ]; then
    echo "[skip-chain] $MAN missing"; continue
  fi
  ARRAY_HI=$((${SHARDS[$SET]} - 1))
  WT="${WALLTIMES[$SET]:-}"
  TIME_ARG=""
  [ -n "$WT" ] && TIME_ARG="--time=$WT"
  EXP="ALL,INPUT=${MAN},OUT_DIR=${SYNTH_ROOT}/${SET}_K50,REF_WAV=${REF_DEFAULT_WAV},REF_TXT=${REF_DEFAULT_TXT}"
  SYNTH_JOB=$(sbatch --parsable --array=0-${ARRAY_HI} $TIME_ARG --export="$EXP" scripts/synthesize_to_shar_nanovllm.slurm)
  echo "[chain] ${SET}_K50 synth (array=0-${ARRAY_HI}): $SYNTH_JOB"

  PQ="ALL,SRC=${SYNTH_ROOT},OUT=${PARQ_ROOT},SETS=${SET}_K50"
  PQ_JOB=$(sbatch --parsable --dependency=afterok:$SYNTH_JOB --export="$PQ" scripts/shar_to_parquet.slurm)
  echo "[chain] ${SET}_K50 parquet: $PQ_JOB"
done

echo "[done] PL K=50 extra fan-out submitted"
