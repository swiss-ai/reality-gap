import os
import shutil
import soundfile as sf
import numpy as np
import librosa

ROOT = "/capstor/store/cscs/swissai/infra01/audio-datasets/voxpopuli"
RAW = os.path.join(ROOT, "raw_audios")
TARGET_SR = 16000


def load_audio(path, target_sr=16000):
    audio, sr = sf.read(path)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return {"array": audio.astype(np.float32), "sampling_rate": sr}


def find_first_audio_file(lang_dir, delete_empty_years=True):
    """Returns (lang, path_to_audio) or None."""
    years = [
        y for y in os.listdir(lang_dir)
        if os.path.isdir(os.path.join(lang_dir, y))
    ]

    if len(years) == 0:
        print(f"No year folders in {lang_dir}")
        return None

    for year in sorted(years):
        year_dir = os.path.join(lang_dir, year)

        # list .ogg files
        oggs = [
            f for f in os.listdir(year_dir)
            if f.lower().endswith(".ogg")
        ]

        if len(oggs) == 0:
            # delete empty year directories if requested
            if delete_empty_years:
                print(f"Deleting empty year folder: {year_dir}")
                shutil.rmtree(year_dir)
            continue

        # found a valid audio file
        return os.path.join(year_dir, oggs[0])

    print(f"No audio files found in any year folder of {lang_dir}")
    return None


def main():
    print(f"Checking root: {RAW}")

    if not os.path.exists(RAW):
        print("raw_audios does not exist")
        return

    # detect languages
    langs = [
        d for d in os.listdir(RAW)
        if os.path.isdir(os.path.join(RAW, d))
    ]

    if len(langs) == 0:
        print("No languages found in raw_audios")
        return

    lang = langs[0]  # e.g. "hr"
    lang_dir = os.path.join(RAW, lang)
    print(f"Detected language: {lang}")

    # get first .ogg file, deleting empty years
    audio_path = find_first_audio_file(lang_dir, delete_empty_years=True)
    if audio_path is None:
        return

    print(f"Found audio file: {audio_path}")

    # load + resample if needed
    audio_obj = load_audio(audio_path, TARGET_SR)

    arr = audio_obj["array"]
    sr = audio_obj["sampling_rate"]
    duration = len(arr) / sr

    print("\n=== Audio Object ===")
    print("Sampling rate:", sr)
    print("Array shape: ", arr.shape)
    print(f"Duration: {duration:.2f} seconds")

    print("\n=== Derived Fields Preview ===")
    example_id = f"voxpopuli_{os.path.splitext(os.path.basename(audio_path))[0]}"

    print({
        "example_id": example_id,
        "dataset": "voxpopuli",
        "audio": arr,
        "sampling_rate": sr,
        "duration": duration,
        "language": lang,
    })


if __name__ == "__main__":
    main()
