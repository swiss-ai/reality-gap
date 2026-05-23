#!/usr/bin/env python3
"""Test which audio decoders in the NGC container handle MP3.

Streams one MP3 from the CV pl .tar.zst, tries torchaudio / soundfile /
librosa / miniaudio in turn, reports which work.

This is a pre-flight test for the extract_pl_refs no-ffmpeg path — we need
ONE decoder in the synth container that handles MP3 so torchaudio.load() in
synthesize_to_shar.py's precache_per_item_refs() works at synth time.

Usage (inside container):
    python3 scripts/test_mp3_decode.py
"""

import io
import tarfile

import zstandard as zstd


ARCHIVE = "/capstor/store/cscs/swissai/infra01/audio-datasets/raw/commonvoice24/pl/validated_clips.tar.zst"


def get_one_mp3():
    print(f"[mp3] streaming first mp3 from {ARCHIVE}")
    dctx = zstd.ZstdDecompressor()
    with open(ARCHIVE, "rb") as f, dctx.stream_reader(f) as stream:
        with tarfile.open(fileobj=stream, mode="r|") as tf:
            for m in tf:
                if m.name.endswith(".mp3"):
                    data = tf.extractfile(m).read()
                    return m.name, data
    raise RuntimeError("no mp3 found in archive")


def try_decoder(name, fn):
    try:
        result = fn()
        print(f"  ✓ {name}: {result}")
        return True
    except Exception as e:
        print(f"  ✗ {name}: {type(e).__name__}: {str(e)[:200]}")
        return False


def main():
    fname, data = get_one_mp3()
    print(f"[mp3] got {fname} ({len(data):,} bytes)\n")

    print("[decoders] testing options that bypass system ffmpeg binary")

    def torchaudio_test():
        import torchaudio
        w, sr = torchaudio.load(io.BytesIO(data))
        return f"shape={tuple(w.shape)}, sr={sr}"

    def soundfile_test():
        import soundfile as sf
        w, sr = sf.read(io.BytesIO(data))
        return f"shape={w.shape}, sr={sr}"

    def librosa_test():
        import librosa
        w, sr = librosa.load(io.BytesIO(data), sr=None)
        return f"shape={w.shape}, sr={sr}"

    def miniaudio_test():
        import miniaudio
        out = miniaudio.decode(data)
        return f"frames={out.num_frames}, sr={out.sample_rate}, ch={out.nchannels}"

    def imageio_ffmpeg_test():
        import imageio_ffmpeg
        return f"ffmpeg_exe={imageio_ffmpeg.get_ffmpeg_exe()}"

    results = {}
    results["torchaudio"]      = try_decoder("torchaudio.load",     torchaudio_test)
    results["soundfile"]       = try_decoder("soundfile.read",      soundfile_test)
    results["librosa"]         = try_decoder("librosa.load",        librosa_test)
    results["miniaudio"]       = try_decoder("miniaudio.decode",    miniaudio_test)
    results["imageio_ffmpeg"]  = try_decoder("imageio_ffmpeg.exe",  imageio_ffmpeg_test)

    print("\n[verdict]")
    winners = [k for k, v in results.items() if v]
    if winners:
        print(f"  WORKING decoder(s): {', '.join(winners)}")
        if "torchaudio" in winners:
            print("  → no-ffmpeg path is SAFE: synth's torchaudio.load() will decode raw mp3 at runtime")
        elif "miniaudio" in winners:
            print("  → pip install miniaudio, update extract_pl_refs to decode at extract time + save .wav")
        elif "imageio_ffmpeg" in winners:
            print("  → use imageio_ffmpeg.get_ffmpeg_exe() instead of system ffmpeg")
    else:
        print("  NO decoder works! Need to pip install miniaudio or imageio-ffmpeg.")


if __name__ == "__main__":
    main()
