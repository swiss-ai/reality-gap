#!/bin/bash
# Wrapper: find Chinese reference voice candidates from AISHELL-3.
# Login-node safe (uses raw FLAC fallback if soundfile missing).
set -e
AISHELL3=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/aishell/aishell3_train
python3 scripts/find_reference_candidates.py \
    --shar-in "$AISHELL3" \
    --output-dir outputs/ref_candidates_zh \
    --n-candidates 10 \
    --language zh \
    --min-cps 3.5 \
    --max-cps 5.0 \
    --min-duration 5.0 \
    --max-duration 12.0 \
    --min-text-chars 10
