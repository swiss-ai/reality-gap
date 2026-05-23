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
echo "Step 0: Build dedup set (succeeded + pending shards)"
echo "=================================================="

declare -A HANDLED

# 0a. Recent successes: (OUT_DIR, idx) where a job already completed cleanly
while IFS= read -r J; do
  LOG="logs/${J}_synth_nanovllm.out"
  [ -f "$LOG" ] || continue
  OD=$(grep -oP 'OUT_DIR\s*=\s*\K\S+' "$LOG" 2>/dev/null | head -1 || true)
  [ -z "$OD" ] && continue
  IDX="${J##*_}"
  HANDLED["${OD}|${IDX}"]=succeeded
done < <(sacct -u "$USER" --starttime=now-${HOURS}hours --format=JobID,State -P -n \
         | awk -F'|' '$2=="COMPLETED" && $1~/^[0-9]+_[0-9]+$/ {print $1}')

# 0b. Currently pending or running array jobs — expand their indices
while IFS= read -r JOBID; do
  OD=$(scontrol show job "$JOBID" 2>/dev/null | grep -oP 'OUT_DIR=\K[^,[:space:]]+' | head -1 || true)
  [ -z "$OD" ] && continue
  IDX_SPEC=$(scontrol show job "$JOBID" 2>/dev/null | grep -oP 'ArrayTaskId=\K\S+' | head -1 || true)
  [ -z "$IDX_SPEC" ] && continue
  EXPANDED=$(python3 -c "
spec = '$IDX_SPEC'
out = []
for part in spec.split(','):
    if '-' in part:
        a, b = part.split('-', 1); b = b.split('%')[0]
        out.extend(range(int(a), int(b)+1))
    else:
        out.append(int(part))
print(','.join(str(i) for i in out))
" 2>/dev/null || true)
  for IDX in $(echo "$EXPANDED" | tr ',' ' '); do
    [ -n "$IDX" ] && HANDLED["${OD}|${IDX}"]=pending
  done
done < <(squeue --me -h -t PD,R -o "%F" | sort -u)

echo "Dedup set built: ${#HANDLED[@]} (out_dir, idx) pairs already-covered"

echo ""
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
    INPUT=$(grep -oP 'INPUT\s*=\s*\K\S+' "$LOG" 2>/dev/null | head -1 || true)
    OUT_DIR=$(grep -oP 'OUT_DIR\s*=\s*\K\S+' "$LOG" 2>/dev/null | head -1 || true)
    NUM_SHARDS=$(grep -oP 'shard [0-9]+/\K[0-9]+' "$LOG" 2>/dev/null | head -1 || true)
    if [ -z "$INPUT" ] || [ -z "$OUT_DIR" ] || [ -z "$NUM_SHARDS" ]; then
      echo "  [skip] cannot parse $LOG"; continue
    fi
    IDX="${FULL#*_}"
    # Skip if this (OUT_DIR, idx) is already covered by a success or pending retry
    if [ -n "${HANDLED["${OUT_DIR}|${IDX}"]:-}" ]; then
      echo "  [skip] $FULL — ${HANDLED["${OUT_DIR}|${IDX}"]} (dedup)"
      continue
    fi
    KEY="${INPUT}|${OUT_DIR}|${NUM_SHARDS}"
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
                         | awk '$2=="DependencyNeverSatisfied" {print $1}' || true)

echo "Found ${#DEAD_PARQ[@]} dead-dep jobs"

for J in "${DEAD_PARQ[@]}"; do
  NAME=$(scontrol show job "$J" 2>/dev/null | grep -oP 'JobName=\K\S+' 2>/dev/null | head -1 || true)
  # Only handle shar_to_parquet jobs (not synth or orchestrator)
  if [[ "$NAME" != *parq* ]] && [[ "$NAME" != *shar_to_parquet* ]]; then
    echo "  [skip] $J ($NAME) — not a parquet job"; continue
  fi
  # Extract SETS from env vars (shar_to_parquet uses SETS=...)
  SET=$(scontrol show job "$J" 2>/dev/null | grep -oP 'SETS=\K[^,]+' 2>/dev/null | head -1 || true)
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
