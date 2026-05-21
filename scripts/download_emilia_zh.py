#!/usr/bin/env python3
"""Download Emilia-YODAS ZH tar files from HuggingFace beyond what we already have.

The cluster already has ZH-B000000.tar through ZH-B000008.tar at
/capstor/store/cscs/swissai/infra01/audio-datasets/emilia_separated/Emilia-YODAS/ZH
(~200 h). Full Emilia ZH on HF is much larger. This script lists what's
available, skips what we have locally, and downloads a controlled batch
to scratch.

License: CC-BY 4.0 (Emilia-Dataset). Permissive for commercial use with
attribution.
"""

import argparse
import os
import re
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default="amphion/Emilia-Dataset",
                   help="HF repo")
    p.add_argument("--lang-prefix", default="Emilia-YODAS/ZH/ZH-B",
                   help="Repo path prefix for the language tars")
    p.add_argument("--local-existing-dir",
                   default="/capstor/store/cscs/swissai/infra01/audio-datasets/emilia_separated/Emilia-YODAS/ZH",
                   help="Where the existing ZH tars live — skip files already here")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Where to write newly-downloaded tars (use scratch)")
    p.add_argument("--max-files", type=int, default=50,
                   help="Cap downloads to this many new tars (~~5-10h each)")
    p.add_argument("--max-gb", type=float, default=200.0,
                   help="Cap downloads at this total size (GB)")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                   help="HF auth token if dataset is gated")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi(token=args.token)
    print(f"Listing files in {args.repo_id} ...")
    all_files = api.list_repo_files(args.repo_id, repo_type="dataset",
                                    token=args.token)
    zh_tars = sorted(f for f in all_files
                     if f.startswith(args.lang_prefix) and f.endswith(".tar"))
    print(f"  Found {len(zh_tars)} ZH tars in repo")

    # What we already have locally (by basename).
    existing = set()
    if Path(args.local_existing_dir).exists():
        existing = {f.name for f in Path(args.local_existing_dir).iterdir()
                    if f.suffix == ".tar"}
    print(f"  Already on cluster: {len(existing)} tars at {args.local_existing_dir}")

    # Pick the next N missing tars.
    to_download = []
    for f in zh_tars:
        basename = Path(f).name
        if basename in existing:
            continue
        if (args.output_dir / basename).exists():
            continue
        to_download.append(f)
        if len(to_download) >= args.max_files:
            break

    if not to_download:
        print("Nothing to download — all available tars already present.")
        return

    print(f"\nWill download {len(to_download)} new tar(s) to {args.output_dir}")
    print(f"  First: {to_download[0]}")
    print(f"  Last:  {to_download[-1]}")

    total_bytes = 0
    max_bytes = args.max_gb * 1024**3
    for i, repo_path in enumerate(to_download):
        if total_bytes >= max_bytes:
            print(f"\nHit --max-gb cap ({args.max_gb} GB). Stopping.")
            break
        print(f"\n[{i+1}/{len(to_download)}] {repo_path}")
        try:
            local = hf_hub_download(
                args.repo_id,
                filename=repo_path,
                repo_type="dataset",
                local_dir=str(args.output_dir.parent),
                token=args.token,
            )
            sz = Path(local).stat().st_size
            total_bytes += sz
            print(f"  -> {local} ({sz/1024**3:.2f} GB, total {total_bytes/1024**3:.2f} GB)")
        except Exception as e:
            print(f"  ERROR: {e}")
            # Continue with the next file rather than abort the whole run.

    print(f"\nDone. Downloaded {total_bytes/1024**3:.2f} GB total to "
          f"{args.output_dir.parent}")


if __name__ == "__main__":
    main()
