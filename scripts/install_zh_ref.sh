#!/bin/bash
# Install a Chinese reference speaker into outputs/reference_audio_zh/
# Transcodes FLAC -> real WAV via voxcpm2 venv's soundfile (login-node fallback
# wrote raw FLAC bytes with .flac extension; downstream synth wants WAV).
#
# Usage:
#   SPK_PREFIX=mls_00_spkSSB06680163 bash scripts/install_zh_ref.sh
set -e
SRC_DIR="${SRC_DIR:-outputs/ref_candidates_zh}"
DST_DIR="${DST_DIR:-outputs/reference_audio_zh}"
SPK_PREFIX="${SPK_PREFIX:?must set SPK_PREFIX e.g. mls_00_spkSSB06680163}"

src_wav_or_flac=$(ls ${SRC_DIR}/${SPK_PREFIX}*.wav ${SRC_DIR}/${SPK_PREFIX}*.flac 2>/dev/null | head -1)
src_txt=$(ls ${SRC_DIR}/${SPK_PREFIX}*.txt | head -1)
echo "Source audio: $src_wav_or_flac"
echo "Source text:  $src_txt"

mkdir -p "$DST_DIR"

# Transcode to PCM_16 WAV using the voxcpm2 venv's soundfile.
source .venv-voxcpm2/bin/activate
python3 - <<PY
import soundfile as sf
wav, sr = sf.read("$src_wav_or_flac")
if wav.ndim > 1:
    wav = wav.mean(axis=1)
sf.write("$DST_DIR/zh_speaker_0.wav", wav, sr, subtype="PCM_16")
print(f"Wrote $DST_DIR/zh_speaker_0.wav (sr={sr}, len={len(wav)/sr:.2f}s)")
PY

cp "$src_txt" "$DST_DIR/zh_speaker_0.txt"

# Duplicate to 3-slot layout for the benchmark slurm.
for i in 1 2; do
    cp "$DST_DIR/zh_speaker_0.wav" "$DST_DIR/zh_speaker_${i}.wav"
    cp "$DST_DIR/zh_speaker_0.txt" "$DST_DIR/zh_speaker_${i}.txt"
done

echo "Installed at $DST_DIR/zh_speaker_{0,1,2}.{wav,txt}"
ls -la "$DST_DIR"
