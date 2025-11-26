#!/usr/bin/env bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the audio_tokenization directory (parent of scripts/)
AUDIO_TOKENIZATION_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to audio_tokenization directory
cd "$AUDIO_TOKENIZATION_DIR" || exit 1

if [ -d "data/raw/librispeech_asr/train.100" ]; then
  echo "LibriSpeech dataset already exists in data/raw/librispeech_asr. Skipping download."
  exit 0
fi

python ../src/repos/posttraining-data/01-hf-download/hf-download.py librispeech_asr --download-folder data/raw --subset clean
