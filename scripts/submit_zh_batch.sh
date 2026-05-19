#!/bin/bash
# Submit Chinese synth (nano-vllm) + dependent parquet conversion for every
# ready zh manifest. Each set runs as its own array, with the parquet job
# chained via --dependency=afterok so it auto-fires when synth completes.
#
# Usage: bash scripts/submit_zh_batch.sh
# Edit the SETS array below to skip / add datasets.

set -e

# Shared env for synth
export BACKEND=voxcpm2_nanovllm
export REF_WAV=outputs/reference_audio_zh/zh_speaker_0.wav
export REF_TXT=outputs/reference_audio_zh/zh_speaker_0.txt

SCRATCH_BASE=/capstor/scratch/cscs/$USER
SYNTH_BASE=$SCRATCH_BASE/synthetic-chinese-voxcpm2
PARQ_BASE=$SCRATCH_BASE/parquet

# Format: manifest_file:set_name:num_shards
# num_shards picked so each shard handles ~20-50h source content
SETS=(
    "data/tts_bench/zh_aishell1.json:zh_aishell1_spk0668:8"
    "data/tts_bench/zh_aishell4_L.json:zh_aishell4_L_spk0668:2"
    "data/tts_bench/zh_aishell4_M.json:zh_aishell4_M_spk0668:4"
    "data/tts_bench/zh_aishell4_S.json:zh_aishell4_S_spk0668:2"
    "data/tts_bench/zh_cv24.json:zh_cv24_spk0668:8"
    "data/tts_bench/zh_emilia.json:zh_emilia_spk0668:4"
)

for entry in "${SETS[@]}"; do
    IFS=":" read -r manifest set_name num_shards <<< "$entry"

    export INPUT="$manifest"
    export OUT_DIR="$SYNTH_BASE/$set_name"
    export NUM_SHARDS="$num_shards"

    last_idx=$((num_shards - 1))
    echo "=== $set_name (manifest=$manifest, shards=$num_shards) ==="

    synth_jid=$(sbatch --parsable --array=0-$last_idx --export=ALL scripts/synthesize_to_shar_nanovllm.slurm)
    echo "  synth job: $synth_jid"

    pq_jid=$(sbatch --parsable \
        --dependency=afterok:$synth_jid \
        --export=ALL,SRC=$SYNTH_BASE,OUT=$PARQ_BASE,SETS=$set_name \
        scripts/shar_to_parquet.slurm)
    echo "  parquet job: $pq_jid  (waits for $synth_jid)"
done

echo
echo "All jobs submitted. Watch with:"
echo "  squeue -u $USER --format=\"%.10i %.20j %.8T %.16r\""
