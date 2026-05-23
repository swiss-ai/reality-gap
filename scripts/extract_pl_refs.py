#!/usr/bin/env python3
"""Extract the K reference mp3 clips from CV pl validated_clips.tar.zst,
decode each to a 16 kHz mono wav, and update the pool JSON in place with
the materialized wav paths.

Needs zstandard + (ffmpeg via subprocess) or torchaudio. Runs inside NGC
container via srun.

Usage:
    python3 scripts/extract_pl_refs.py \\
        --pool references/pl_K50_seed42_cps30.json \\
        --out-dir references/pl_K50_seed42_cps30
"""

import argparse
import io
import json
import tarfile
from pathlib import Path

import soundfile as sf
import zstandard as zstd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--target-sr", type=int, default=16000)
    args = p.parse_args()

    with open(args.pool) as f:
        pool = json.load(f)

    archive = pool.get("clips_archive")
    if not archive:
        raise SystemExit(f"Pool {args.pool} has no `clips_archive` field")
    archive = Path(archive)
    if not archive.exists():
        raise SystemExit(f"Archive not found: {archive}")

    # Map clip filename → output wav path for items that don't already have refs.
    # Decode mp3 → WAV via soundfile (libsndfile, no ffmpeg binary needed).
    # Synth pipeline's torchaudio.load() then reads WAV cleanly without any
    # mp3-decoder dependency at synth-time.
    want = {}
    for spk in pool["speakers"]:
        if spk["source"] == "anchor" and spk.get("ref_wav"):
            continue  # anchor wav already exists, don't re-extract
        clip = spk["ref_clip_name"]
        out_name = f"{spk['spk_id']}__{Path(clip).stem}.wav"
        want[clip] = (spk, args.out_dir / out_name)

    print(f"[want] extracting + decoding {len(want)} mp3→wav from {archive}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dctx = zstd.ZstdDecompressor()
    found = 0
    failed = []
    with open(archive, "rb") as zfh, dctx.stream_reader(zfh) as stream:
        with tarfile.open(fileobj=stream, mode="r|") as tf:
            for m in tf:
                fname = Path(m.name).name
                if fname not in want:
                    continue
                spk, out_path = want[fname]
                mp3_bytes = tf.extractfile(m).read()
                try:
                    samples, sr = sf.read(io.BytesIO(mp3_bytes))
                    sf.write(str(out_path), samples, sr, subtype="PCM_16")
                    spk["ref_wav"] = str(out_path)
                    found += 1
                    if found % 5 == 0 or found == len(want):
                        print(f"  [{found}/{len(want)}] {fname} → {out_path.name} (sr={sr})")
                except Exception as e:
                    failed.append((fname, str(e)[:200]))
                    print(f"  [fail] {fname}: {e}")
                if found >= len(want):
                    break

    # Persist updated pool with materialized ref_wav paths.
    with open(args.pool, "w") as f:
        json.dump(pool, f, indent=2, ensure_ascii=False)
    print(f"[out] updated {args.pool} — {found}/{len(want)} refs materialized")
    if found < len(want):
        missing = [c for c in want if want[c][0].get("ref_wav") in (None, "")]
        print(f"[WARN] {len(missing)} clips not found in archive:")
        for c in missing[:10]:
            print(f"  {c}")


if __name__ == "__main__":
    main()
