#!/bin/bash
# Sweep FAILED synth shards in the last N hours, refire them, and resubmit
# dead parquets (DependencyNeverSatisfied). Idempotent — safe to run multiple
# times.
#
# Usage:
#     bash scripts/auto_refire_failed.sh [hours_back]
#     DRY_RUN=1 bash scripts/auto_refire_failed.sh   # preview without submitting

set -euo pipefail

HOURS=${1:-12}
DRY_RUN=${DRY_RUN:-0}

SS=/capstor/scratch/cscs/mkwapniewska/synth
PP=/capstor/scratch/cscs/mkwapniewska/parquet
RW_ZH=outputs/reference_audio_zh/zh_speaker_0.wav
RT_ZH=outputs/reference_audio_zh/zh_speaker_0.txt
RW_PL=outputs/reference_audio_spk1636/pl_speaker_0.wav
RT_PL=outputs/reference_audio_spk1636/pl_speaker_0.txt

echo "=================================================="
echo "Step 1: Sweep FAILED synth shards (last ${HOURS}h)"
echo "=================================================="

mapfile -t FAILED < <(sacct -u "$USER" --starttime=now-${HOURS}hours --format=JobID,State -P -n \
                      | awk -F'|' '$2=="FAILED" && $1~/^[0-9]+_[0-9]+$/ {print $1}')

echo "Found ${#FAILED[@]} failed synth shards"

if [ ${#FAILED[@]} -eq 0 ]; then
  echo "Nothing to refire."
else
  # Group by parent: INPUT|OUT_DIR|NUM_SHARDS → list of indices
  declare -A INDICES
  declare -A META
  for FULL in "${FAILED[@]}"; do
    LOG="logs/${FULL}_synth_nanovllm.out"
    if [ ! -f "$LOG" ]; then
      echo "  [skip] no log for $FULL"; continue
    fi
    INPUT=$(grep -oP 'INPUT\s*=\s*\K\S+' "$LOG" | head -1)
    OUT_DIR=$(grep -oP 'OUT_DIR\s*=\s*\K\S+' "$LOG" | head -1)
    NUM_SHARDS=$(grep -oP 'shard [0-9]+/\K[0-9]+' "$LOG" | head -1)
    if [ -z "$INPUT" ] || [ -z "$OUT_DIR" ] || [ -z "$NUM_SHARDS" ]; then
      echo "  [skip] cannot parse $LOG"; continue
    fi
    KEY="${INPUT}|${OUT_DIR}|${NUM_SHARDS}"
    IDX="${FULL#*_}"
    INDICES[$KEY]+="${IDX},"
    META[$KEY]="$INPUT|$OUT_DIR|$NUM_SHARDS"
  done

  declare -A REFIRE_JOB_BY_SET
  for KEY in "${!INDICES[@]}"; do
    IDX_LIST="${INDICES[$KEY]%,}"
    IFS='|' read -r INPUT OUT_DIR NUM_SHARDS <<< "${META[$KEY]}"
    SET=$(basename "$OUT_DIR")
    if [[ "$INPUT" == *pl_* ]]; then RW=$RW_PL; RT=$RT_PL
    else                            RW=$RW_ZH; RT=$RT_ZH
    fi
    EXP="ALL,INPUT=${INPUT},OUT_DIR=${OUT_DIR},REF_WAV=${RW},REF_TXT=${RT},NUM_SHARDS=${NUM_SHARDS}"
    echo ""
    echo "[refire] $SET  indices=${IDX_LIST}  (of ${NUM_SHARDS} total)"
    if [ "$DRY_RUN" = "1" ]; then
      echo "  DRY_RUN — would: sbatch --array=${IDX_LIST} --job-name=${SET}_auto --time=01:00:00 --export=${EXP}"
    else
      J=$(sbatch --parsable --array="${IDX_LIST}" --job-name="${SET}_auto" --time=01:00:00 --export="$EXP" scripts/synthesize_to_shar_nanovllm.slurm)
      echo "  submitted: $J"
      REFIRE_JOB_BY_SET[$SET]=$J
    fi
  done
fi

echo ""
echo "=================================================="
echo "Step 2: Resubmit DEAD parquets (DependencyNeverSatisfied)"
echo "=================================================="

mapfile -t DEAD_PARQ < <(squeue --me -t PD -h -o "%i %r" 2>/dev/null \
                         | awk '$2=="DependencyNeverSatisfied" {print $1}')

echo "Found ${#DEAD_PARQ[@]} dead-dep jobs"

for J in "${DEAD_PARQ[@]}"; do
  NAME=$(scontrol show job "$J" 2>/dev/null | grep -oP 'JobName=\K\S+' | head -1)
  # Only handle shar_to_parquet jobs (not synth or orchestrator)
  if [[ "$NAME" != *parq* ]] && [[ "$NAME" != *shar_to_parquet* ]]; then
    echo "  [skip] $J ($NAME) — not a parquet job"; continue
  fi
  # Extract SETS from env vars (shar_to_parquet uses SETS=...)
  SET=$(scontrol show job "$J" 2>/dev/null | grep -oP 'SETS=\K[^,]+' | head -1)
  if [ -z "$SET" ]; then
    echo "  [skip] $J — cannot find SETS"; continue
  fi
  echo ""
  echo "[parquet] $SET  (was $J)"
  # Chain on the refire job we just submitted for this set (if any), so the
  # parquet only runs after the missing shards land.
  DEP_ARG=""
  if [ "${REFIRE_JOB_BY_SET[$SET]:-}" != "" ]; then
    DEP_ARG="--dependency=afterok:${REFIRE_JOB_BY_SET[$SET]}"
    echo "  chained on refire ${REFIRE_JOB_BY_SET[$SET]}"
  fi
  if [ "$DRY_RUN" = "1" ]; then
    echo "  DRY_RUN — would: scancel $J; sbatch $DEP_ARG --job-name=${SET}_parq --export=ALL,SRC=${SS},OUT=${PP},SETS=${SET}"
  else
    scancel "$J"
    NEW=$(sbatch --parsable $DEP_ARG --job-name="${SET}_parq" --export="ALL,SRC=${SS},OUT=${PP},SETS=${SET}" scripts/shar_to_parquet.slurm)
    echo "  resubmitted: $NEW"
  fi
done

echo ""
echo "Done. squeue --me to see new state."
