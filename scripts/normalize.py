'''
Python module for normalizing audio datasets (VoxPopuli, more to come) into a unified schema:
{
    "example_id": str,
    "dataset": str,
    "audio": {"array": np.ndarray, "sampling_rate": int},
    "sampling_rate": int,     # always 16k after resampling
    "duration": float,        # seconds
    "language": Optional[str],
    "text": Optional[str],    # "" if unavailable
}

'''

import argparse
from pathlib import Path
import soundfile as sf
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import librosa
import numpy as np
import gc

# ----------------------------
# Normalize a single audio file
# ----------------------------
def normalize_one(dataset_name: str, path: Path):
    # Detect language from path: raw_audios/<lang>/<year>/file.ogg
    lang = path.parts[-3]  # assumes raw_audios/<lang>/<year>/file

    # --- load + resample ---
    audio_arr, sr = sf.read(path)
    if audio_arr.ndim > 1:
        audio_arr = audio_arr.mean(axis=1)  # convert to mono

    if sr != 16000:
        audio_arr = librosa.resample(audio_arr, orig_sr=sr, target_sr=16000)
        sr = 16000

    duration = len(audio_arr) / 16000.0
    example_id = f"{dataset_name}_{path.stem}"

    # Convert to float32 immediately to save memory
    audio_arr = audio_arr.astype(np.float32)

    return {
        "example_id": example_id,
        "dataset": dataset_name,
        "audio_array": audio_arr,
        "sampling_rate": sr,
        "duration": duration,
        "language": lang,
        "text": None,
        "filepath": str(path),
    }


# ----------------------------
# Build Arrow Table from rows
# ----------------------------
def build_arrow_table(rows):
    # Already float32, no need to convert again
    return pa.table({
        "example_id": [r["example_id"] for r in rows],
        "dataset": [r["dataset"] for r in rows],
        "audio": [r["audio_array"] for r in rows],  # Already float32
        "sampling_rate": [r["sampling_rate"] for r in rows],
        "duration": [r["duration"] for r in rows],
        "language": [r["language"] for r in rows],
        "text": [r["text"] if r["text"] else "" for r in rows],
        "filepath": [r["filepath"] for r in rows],
    })


# ----------------------------
# Process one language (batch) - MEMORY EFFICIENT VERSION
# ----------------------------
def process_language(dataset_root: Path, output_dir: Path, lang: str, batch_size: int):
    lang_root = dataset_root / lang
    audio_files = list(lang_root.rglob("*.flac")) \
                + list(lang_root.rglob("*.wav")) \
                + list(lang_root.rglob("*.ogg"))

    if len(audio_files) == 0:
        print(f"[WARN] No audio files found for {lang}, skipping")
        return

    print(f"[INFO] Found {len(audio_files)} audio files in {lang}")
    out_file = output_dir / f"voxpopuli_{lang}.parquet"
    
    # Remove existing file if it exists
    if out_file.exists():
        out_file.unlink()

    writer = None
    rows = []
    
    # Add memory tracking
    total_processed = 0
    
    for i, path in enumerate(tqdm(audio_files, desc=f"Normalizing {lang}", unit="audio")):
        try:
            row = normalize_one("voxpopuli", path)
            rows.append(row)
            total_processed += 1
        except Exception as e:
            print(f"[ERROR] Failed on {path}: {e}")
            continue

        # Write batch when reaching batch_size or at the end
        if len(rows) >= batch_size or i == len(audio_files) - 1:
            if len(rows) > 0:  # Only write if we have data
                table = build_arrow_table(rows)
                
                if writer is None:
                    # Initialize writer on first batch
                    writer = pq.ParquetWriter(out_file, table.schema, compression="ZSTD")
                
                writer.write_table(table)
                
                # Clear the rows list and force garbage collection
                rows.clear()
                del table
                gc.collect()
                
                # Reset rows list
                rows = []
    
    # Close the writer
    if writer is not None:
        writer.close()
        print(f"[INFO] Successfully processed {total_processed} files for {lang}")


# ----------------------------
# Alternative: Process in smaller chunks with immediate write
# ----------------------------
def process_language_streaming(dataset_root: Path, output_dir: Path, lang: str, chunk_size: int = 10):
    """
    Ultra memory-efficient version that processes files one at a time
    and writes in very small chunks
    """
    lang_root = dataset_root / lang
    audio_files = list(lang_root.rglob("*.flac")) \
                + list(lang_root.rglob("*.wav")) \
                + list(lang_root.rglob("*.ogg"))

    if len(audio_files) == 0:
        print(f"[WARN] No audio files found for {lang}, skipping")
        return

    print(f"[INFO] Found {len(audio_files)} audio files in {lang}")
    out_file = output_dir / f"voxpopuli_{lang}.parquet"
    
    if out_file.exists():
        out_file.unlink()

    writer = None
    rows = []
    total_processed = 0
    
    for i, path in enumerate(tqdm(audio_files, desc=f"Normalizing {lang}", unit="audio")):
        try:
            # Process one file
            row = normalize_one("voxpopuli", path)
            
            # Immediately write if this is a large audio file (>10 seconds)
            if row["duration"] > 10 and len(rows) > 0:
                # Write accumulated rows first
                table = build_arrow_table(rows)
                if writer is None:
                    writer = pq.ParquetWriter(out_file, table.schema, compression="ZSTD")
                writer.write_table(table)
                rows.clear()
                del table
                gc.collect()
            
            rows.append(row)
            total_processed += 1
            
            # Write small chunks frequently
            if len(rows) >= chunk_size or i == len(audio_files) - 1:
                if len(rows) > 0:
                    table = build_arrow_table(rows)
                    if writer is None:
                        writer = pq.ParquetWriter(out_file, table.schema, compression="ZSTD")
                    writer.write_table(table)
                    rows.clear()
                    del table
                    gc.collect()
                    
        except Exception as e:
            print(f"[ERROR] Failed on {path}: {e}")
            gc.collect()  # Clean up on error too
            continue
    
    if writer is not None:
        writer.close()
        print(f"[INFO] Successfully processed {total_processed} files for {lang}")


# ----------------------------
# Main
# ----------------------------
def main(root: str, out: str, batch_size: int, language: str = None, streaming: bool = False):
    dataset_root = Path(root) / "voxpopuli" / "raw_audios"
    output_dir = Path(out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect languages automatically
    all_langs = sorted([p.name for p in dataset_root.iterdir() if p.is_dir() and not p.name.endswith(".tar")])
    if language:
        langs_to_process = [language] if language in all_langs else []
    else:
        langs_to_process = all_langs

    print(f"[INFO] Detected languages: {langs_to_process}")

    for lang in langs_to_process:
        if streaming:
            # Use ultra memory-efficient version
            process_language_streaming(dataset_root, output_dir, lang, chunk_size=batch_size)
        else:
            process_language(dataset_root, output_dir, lang, batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str,
                        default="/capstor/store/cscs/swissai/infra01/audio-datasets")
    parser.add_argument("--out", type=str, default="normalized-arrow")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Batch size for writing (smaller = less memory)")
    parser.add_argument("--language", type=str, default=None,
                        help="Process only this language")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming mode for minimal memory usage")
    args = parser.parse_args()

    main(args.root, args.out, args.batch_size, args.language, args.streaming)