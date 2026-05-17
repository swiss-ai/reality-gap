#!/usr/bin/env python3
"""Pull slow-paced reference clips from a Lhotse Shar dir (e.g. MLS Polish).

Walks **/cuts.*.jsonl.gz recursively (handles Spark-prepared part-NNNNN/
layouts), filters cuts by chars/sec speaking rate, then extracts audio for
the picked candidates from the matching recording.*.tar.

Polish speech rate reference:
    ~11-13 cps = slow/deliberate (audiobook narrator, good for TTS reference)
    ~14-16 cps = natural conversation
    ~17-20 cps = fast news anchor
    ~22+ cps  = very fast
"""

import argparse
import gzip
import io
import json
import random
import re
import tarfile
from collections import defaultdict
from pathlib import Path

try:
    import soundfile as sf
    _HAVE_SF = True
except ImportError:
    sf = None
    _HAVE_SF = False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shar-in", required=True, type=Path,
                   help="Lhotse Shar dir root (recurses into part-NNNNN/ etc.)")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Where to dump candidate WAVs + .txt transcripts")
    p.add_argument("--n-candidates", type=int, default=10)
    # cps ranges by language:
    #   pl (alphabetic): natural 14-16, slow audiobook 11-13. Use 11-14.
    #   zh (logographic): natural 4-6 chars/sec, slow narration 3-4. Use 3-5.
    # Override with --min-cps/--max-cps for other languages.
    p.add_argument("--min-cps", type=float, default=11.0)
    p.add_argument("--max-cps", type=float, default=14.0)
    p.add_argument("--min-duration", type=float, default=5.0)
    p.add_argument("--max-duration", type=float, default=12.0)
    p.add_argument("--min-text-chars", type=int, default=30,
                   help="Minimum text length. For Chinese (logographic), drop "
                        "this to ~10 since each char carries more info.")
    p.add_argument("--language", choices=["pl", "zh", "none"], default="pl",
                   help="Skip cuts whose text doesn't match the target language's "
                        "char set. 'none' disables the language filter.")
    p.add_argument("--scan-limit", type=int, default=2000,
                   help="Stop after this many matches")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find all cuts files. Each lives next to a matching recording.*.tar.
    cuts_files = sorted(args.shar_in.rglob("cuts.*.jsonl.gz"))
    if not cuts_files:
        raise SystemExit(f"No cuts.*.jsonl.gz under {args.shar_in}")
    print(f"Found {len(cuts_files)} cuts files. Scanning...")

    PL_CHARS = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")
    def _has_pl(t): return any(c in PL_CHARS for c in t)
    def _has_cjk(t): return any("一" <= c <= "鿿" for c in t)
    LANG_PRED = {"pl": _has_pl, "zh": _has_cjk, "none": lambda t: True}
    lang_pred = LANG_PRED[args.language]

    # candidates: list of (cps, cut_dict, rec_tar_path, text)
    candidates = []
    scanned = 0
    for cuts_path in cuts_files:
        # The matching recording tar is the same stem with "recording." prefix
        # and ".tar" extension. e.g. cuts.000003.jsonl.gz -> recording.000003.tar
        m = re.match(r"cuts\.(\d+)\.jsonl\.gz$", cuts_path.name)
        if not m:
            continue
        shard_id = m.group(1)
        rec_tar = cuts_path.parent / f"recording.{shard_id}.tar"
        if not rec_tar.exists():
            continue

        with gzip.open(cuts_path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                scanned += 1
                try:
                    cut = json.loads(line)
                except json.JSONDecodeError:
                    continue
                dur = float(cut.get("duration", 0.0))
                if not (args.min_duration <= dur <= args.max_duration):
                    continue
                supervisions = cut.get("supervisions") or []
                if not supervisions:
                    continue
                text = (supervisions[0].get("text") or "").strip()
                if len(text) < args.min_text_chars:
                    continue
                if not lang_pred(text):
                    continue
                cps = len(text) / dur
                if args.min_cps <= cps <= args.max_cps:
                    candidates.append((cps, cut, rec_tar, text))
                if len(candidates) >= args.scan_limit:
                    break
        if len(candidates) >= args.scan_limit:
            break

    print(f"Scanned {scanned} cuts, kept {len(candidates)} in "
          f"[{args.min_cps}, {args.max_cps}] cps range")
    if not candidates:
        raise SystemExit("No matches. Try widening --min-cps/--max-cps.")

    random.seed(args.seed)
    random.shuffle(candidates)

    # Dedupe by speaker. MLS cut IDs are <speaker>_<book>_<utt> so the first
    # underscore-separated token is the speaker. Picks at most one candidate
    # per speaker to maximize voice diversity.
    seen_speakers = set()
    picked = []
    for entry in candidates:
        cps, cut, rec_tar, text = entry
        speaker = cut["id"].split("_")[0]
        if speaker in seen_speakers:
            continue
        seen_speakers.add(speaker)
        picked.append(entry)
        if len(picked) >= args.n_candidates:
            break
    print(f"Picked {len(picked)} candidates from {len(seen_speakers)} distinct speakers")

    # Group by tar to avoid re-opening the same tar repeatedly.
    by_tar = defaultdict(list)
    for cps, cut, rec_tar, text in picked:
        by_tar[rec_tar].append((cps, cut, text))

    print(f"\nWriting {len(picked)} candidates to {args.output_dir}/")
    manifest = []
    idx = 0
    for rec_tar, items in by_tar.items():
        # Build a name->member map for fast lookup. Audio in Shar is
        # typically <cut_id>.flac or <cut_id>.wav.
        try:
            with tarfile.open(rec_tar, "r") as tf:
                # Shar tars often contain both <id>.flac and <id>.json for
                # each cut. Only consider audio files when matching by stem.
                AUDIO_EXTS = {".flac", ".wav", ".opus", ".ogg", ".mp3"}
                name_map = {Path(m.name).stem: m for m in tf.getmembers()
                            if Path(m.name).suffix.lower() in AUDIO_EXTS}
                for cps, cut, text in items:
                    cut_id = cut["id"]
                    member = name_map.get(cut_id)
                    if member is None:
                        print(f"  WARN: {cut_id} not found in {rec_tar.name}")
                        continue
                    audio_bytes = tf.extractfile(member).read()
                    speaker = cut["id"].split("_")[0]
                    src_ext = Path(member.name).suffix.lower()
                    name = (f"mls_{idx:02d}_spk{speaker}"
                            f"_cps{cps:.1f}_dur{cut['duration']:.1f}s")
                    if _HAVE_SF:
                        # Re-encode to PCM_16 WAV (consistent format for downstream)
                        wav, sr = sf.read(io.BytesIO(audio_bytes))
                        if wav.ndim > 1:
                            wav = wav.mean(axis=1)
                        sf.write(args.output_dir / f"{name}.wav", wav, sr,
                                 subtype="PCM_16")
                    else:
                        # Login-node fallback: dump raw audio bytes with
                        # original extension. QuickTime/VLC play .flac fine.
                        out_path = args.output_dir / f"{name}{src_ext}"
                        out_path.write_bytes(audio_bytes)
                    (args.output_dir / f"{name}.txt").write_text(
                        text + "\n", encoding="utf-8")
                    out_ext = ".wav" if _HAVE_SF else src_ext
                    manifest.append({
                        "filename": f"{name}{out_ext}",
                        "speaker_id": speaker,
                        "cut_id": cut["id"],
                        "duration": cut["duration"],
                        "cps": round(cps, 2),
                        "text": text,
                    })
                    print(f"  {name}.wav -- "
                          f"'{text[:70]}{'...' if len(text) > 70 else ''}'")
                    idx += 1
        except Exception as e:
            print(f"  ERROR opening {rec_tar}: {e}")

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(f"\nDone. {len(manifest)} candidates + manifest.json at "
          f"{args.output_dir}/")


if __name__ == "__main__":
    main()
