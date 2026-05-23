#!/usr/bin/env python3
"""Extract K reference wavs from the AISHELL-3 Lhotse Shar `recording.*.tar`
files, into a user-writeable refs directory. Update the pool JSON in place
with materialized `ref_wav` paths.

Background: AISHELL-3 raw wavs at /capstor/.../raw/aishell/aishell3/ are
ACL-restricted to `infra01adm` only — group `infra01` cannot read them.
But the SHAR's `recording.*.tar` files have a `group:infra01:r-x` ACL, so
we can read those instead.

Stdlib-only (tarfile, json, glob). Runs on Clariden login or inside a
container srun — either works since we only need to read tars + write wavs.

Lhotse Shar layout (this repo):
    <shar_dir>/part-NNNNNN/cuts.NNNNNN.jsonl.gz
    <shar_dir>/part-NNNNNN/recording.NNNNNN.tar

Tar members are named by cut/recording id. AISHELL-3 IDs in our manifest
look like `SSB07510152-0` (cut id with Lhotse segment suffix). The tar
member is typically `SSB07510152-0.wav` (or `.flac`).
"""

import argparse
import glob
import gzip
import io
import json
import tarfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True, type=Path,
                   help="Pool JSON to update (e.g. references/zh_K50_seed42_cps70.json)")
    p.add_argument("--shar-dir", required=True, type=Path,
                   help="Lhotse Shar dir containing part-NNNNNN/ subdirs")
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args()

    pool = json.loads(args.pool.read_text())
    # Map cut_id → output wav path. Tarballs may have members named
    # either "<cut_id>.wav" or "<cut_id>" without ext or with .flac/.opus.
    want = {}
    for spk in pool["speakers"]:
        utt = spk["ref_utt_id"]
        out = args.out_dir / f"{spk['spk_id']}__{utt}.wav"
        want[utt] = (spk, out)
    print(f"[want] {len(want)} refs to extract")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    tars = sorted(glob.glob(str(args.shar_dir / "part-*/recording.*.tar")))
    if not tars:
        # Fallback: some Shar layouts use worker_NN
        tars = sorted(glob.glob(str(args.shar_dir / "worker_*/recording.*.tar")))
    if not tars:
        raise SystemExit(f"No recording tars found under {args.shar_dir}")
    print(f"[tars] {len(tars)} archives to scan")

    AUDIO_EXTS = {".flac", ".wav", ".opus", ".mp3", ".ogg"}
    found = 0
    for tar_path in tars:
        with tarfile.open(tar_path, mode="r") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                if Path(m.name).suffix.lower() not in AUDIO_EXTS:
                    continue  # skip cuts .json side files
                # Strip directory + extension to get raw cut id.
                name = Path(m.name).name
                # Try both forms: full name, and stem (without ext)
                candidates = {name, Path(name).stem}
                hit = None
                for c in candidates:
                    if c in want:
                        hit = c
                        break
                if hit is None:
                    continue
                spk, out = want[hit]
                spk["ref_wav"] = str(out)
                # Copy bytes as-is. The synth orchestrator's torchaudio.load
                # handles wav/flac/opus.
                ext = Path(m.name).suffix.lstrip(".") or "wav"
                out = out.with_suffix(f".{ext}")
                spk["ref_wav"] = str(out)
                out.write_bytes(tf.extractfile(m).read())
                found += 1
                if found % 5 == 0 or found == len(want):
                    print(f"  [{found}/{len(want)}] {hit} → {out.name}")
                if found >= len(want):
                    break
        if found >= len(want):
            break

    # Persist updated pool.
    args.pool.write_text(json.dumps(pool, indent=2, ensure_ascii=False))
    print(f"[out] updated {args.pool} — {found}/{len(want)} refs materialized")
    if found < len(want):
        missing = [utt for utt in want if want[utt][0].get("ref_wav") in (None, "")]
        # The above filter is imperfect since we mutate spk in place; recount
        missing = [utt for utt, (spk, _) in want.items()
                   if not (spk.get("ref_wav") and Path(spk["ref_wav"]).exists())]
        print(f"[WARN] {len(missing)} utts not found in any tar:")
        for utt in missing[:10]:
            print(f"  {utt}")


if __name__ == "__main__":
    main()
