#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --environment=/users/mkwapniewska/.edf/nemo_25_11.toml
#SBATCH --output=/capstor/scratch/cscs/mkwapniewska/logs/pl_k50_orchestrator-%j.out
#
# Submit with --dependency=afterok:<extract_pl_refs_job>
#
# What this does, once the PL ref pool wavs are materialized:
#   1. Augments pl_mls_v2.json / pl_yodas2.json / pl_cv25.json with refs.
#   2. Fires a PL smoke shard (1 of 32 for pl_yodas2) to validate the patch.
#   3. Fires the full K=50 synth+parquet chains, each afterok:smoke.
#
# Drop additional PL manifests into MANIFESTS_BLOCK below as they become
# available (granary chunks, fineweb2 extracts, etc.).

set -euo pipefail
cd "/users/$(whoami)/benchmark-audio-tokenizer"

POOL=references/pl_K50_seed42_cps30.json
REF_DEFAULT_WAV=outputs/reference_audio_spk1636/pl_speaker_0.wav
REF_DEFAULT_TXT=outputs/reference_audio_spk1636/pl_speaker_0.txt
SYNTH_ROOT=/capstor/scratch/cscs/mkwapniewska/synth
PARQ_ROOT=/capstor/scratch/cscs/mkwapniewska/parquet

# ── Augment manifests ─────────────────────────────────────────────────
declare -A SHARDS=(
  [pl_mls_v2]=4
  [pl_yodas2]=32
  [pl_cv25]=8
  [pl_fineweb2_28kh]=128
)
declare -A WALLTIMES=(
  [pl_fineweb2_28kh]=06:00:00
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

# ── Smoke shard (validate the per-item-refs path on PL audio) ─────────
SMOKE_SET=pl_mls_v2
SMOKE_MANIFEST="data/manifests/${SMOKE_SET}_K50.json"
EXP_SMOKE="ALL,INPUT=${SMOKE_MANIFEST},OUT_DIR=${SYNTH_ROOT}/${SMOKE_SET}_K50_smoke,NUM_SHARDS=32,REF_WAV=${REF_DEFAULT_WAV},REF_TXT=${REF_DEFAULT_TXT}"
SMOKE_JOB=$(sbatch --parsable --array=0-0 --time=00:30:00 --export="$EXP_SMOKE" scripts/synthesize_to_shar_nanovllm.slurm)
echo "[smoke] pl_mls_v2 1-shard smoke submitted: $SMOKE_JOB"

# ── Full K=50 chains (afterok:smoke) ──────────────────────────────────
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
  SYNTH_JOB=$(sbatch --parsable --dependency=afterok:$SMOKE_JOB --array=0-${ARRAY_HI} $TIME_ARG --export="$EXP" scripts/synthesize_to_shar_nanovllm.slurm)
  echo "[chain] ${SET}_K50 synth (array=0-${ARRAY_HI}): $SYNTH_JOB"

  PQ="ALL,SRC=${SYNTH_ROOT},OUT=${PARQ_ROOT},SETS=${SET}_K50"
  PQ_JOB=$(sbatch --parsable --dependency=afterok:$SYNTH_JOB --export="$PQ" scripts/shar_to_parquet.slurm)
  echo "[chain] ${SET}_K50 parquet: $PQ_JOB"
done

echo "[done] PL K=50 fan-out submitted"
