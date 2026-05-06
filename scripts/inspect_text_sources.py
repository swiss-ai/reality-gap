#!/usr/bin/env python3
"""Inspect candidate Polish text sources for the parallel-corpus deliverable.

Pulls samples from each candidate (HF dataset or local cache) and reports:
  - sentence-length distribution
  - vocabulary diversity
  - Polish diacritic coverage (proxy for "actually Polish")
  - ASCII-only ratio (proxy for English code-switching)
  - short / long outliers

Use the output to pick which source to feed into the synthetic-audio pipeline
in Weeks 5-6 (parallel real+synthetic) and Week 7 (text-only SFT to speech).

Sources (each can be HF dataset id or local path):
    --sources oasst:OpenAssistant/oasst1::train \\
              cv_pl:mozilla-foundation/common_voice_17_0:pl:test \\
              fleurs_pl:google/fleurs:pl_pl:test \\
              eurospeech_pl:/capstor/.../eurospeech_cache/pl::test

Format per --source: NAME:DATASET_ID_OR_PATH[:CONFIG][:SPLIT]
Empty CONFIG means no config; missing SPLIT defaults to 'train'.
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Iterable, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

POLISH_DIACRITICS = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")
WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)


# ── Source loaders ───────────────────────────────────────────────────
def load_source(spec: str, sample_size: int, language: str) -> list[str]:
    """Parse 'NAME:ID[:CONFIG][:SPLIT]' and return up to sample_size text strings."""
    parts = spec.split(":")
    if len(parts) < 2:
        raise ValueError(f"Bad source spec: {spec!r}. Need NAME:ID[:CONFIG][:SPLIT]")
    name, dataset_id = parts[0], parts[1]
    config = parts[2] if len(parts) > 2 and parts[2] else None
    split = parts[3] if len(parts) > 3 and parts[3] else "train"

    logger.info("Loading %s (id=%s, config=%s, split=%s)", name, dataset_id, config, split)

    # Local path? load_from_disk
    if Path(dataset_id).exists():
        from datasets import load_from_disk

        ds = load_from_disk(dataset_id)
        if split in ds:
            ds = ds[split]
    else:
        from datasets import load_dataset

        kwargs = {"split": split}
        if config:
            kwargs["name"] = config
        # OpenAssistant: streaming so we don't pull the whole dataset
        ds = load_dataset(dataset_id, streaming=True, **kwargs)

    text_col = _detect_text_column(ds)
    lang_col = _detect_lang_column(ds)
    return list(_iter_filtered(ds, text_col, lang_col, language, sample_size))


def _detect_text_column(ds) -> str:
    candidates = ["text", "sentence", "transcription", "transcript", "raw_transcription"]
    cols = ds.column_names if hasattr(ds, "column_names") and ds.column_names else None
    if cols is None:
        # streaming dataset: peek
        for sample in ds:
            cols = list(sample.keys())
            break
    for c in candidates:
        if c in cols:
            return c
    raise RuntimeError(f"No known text column in {cols}")


def _detect_lang_column(ds) -> Optional[str]:
    candidates = ["lang", "language", "locale"]
    cols = ds.column_names if hasattr(ds, "column_names") and ds.column_names else None
    if cols is None:
        for sample in ds:
            cols = list(sample.keys())
            break
    for c in candidates:
        if c in cols:
            return c
    return None


def _iter_filtered(
    ds, text_col: str, lang_col: Optional[str], language: str, n: int
) -> Iterable[str]:
    seen = 0
    for sample in ds:
        if lang_col and language:
            lang_val = str(sample.get(lang_col, "")).lower()
            if not (lang_val == language or lang_val.startswith(language + "-")):
                continue
        text = sample.get(text_col)
        if not text or not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue
        yield text
        seen += 1
        if seen >= n:
            return


# ── Stats ────────────────────────────────────────────────────────────
def compute_stats(name: str, texts: list[str]) -> dict:
    if not texts:
        return {"name": name, "n_sampled": 0, "error": "no texts loaded"}

    char_lens = [len(t) for t in texts]
    word_lens = [len(WORD_RE.findall(t)) for t in texts]
    vocab = Counter()
    n_with_diacritic = 0
    n_ascii_only = 0

    for t in texts:
        words = [w.lower() for w in WORD_RE.findall(t)]
        vocab.update(words)
        if any(c in POLISH_DIACRITICS for c in t):
            n_with_diacritic += 1
        if t.isascii():
            n_ascii_only += 1

    sorted_by_len = sorted(zip(word_lens, texts))
    shortest = [t for _, t in sorted_by_len[:3]]
    longest = [t for _, t in sorted_by_len[-3:]]

    return {
        "name": name,
        "n_sampled": len(texts),
        "median_chars": int(median(char_lens)),
        "median_words": int(median(word_lens)),
        "p10_words": _percentile(word_lens, 10),
        "p90_words": _percentile(word_lens, 90),
        "vocab_size": len(vocab),
        "type_token_ratio": round(len(vocab) / max(sum(word_lens), 1), 4),
        "diacritic_ratio": round(n_with_diacritic / len(texts), 3),
        "ascii_only_ratio": round(n_ascii_only / len(texts), 3),
        "top_words": [w for w, _ in vocab.most_common(20)],
        "examples_short": shortest,
        "examples_long": [t[:200] + ("…" if len(t) > 200 else "") for t in longest],
    }


def _percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    k = int(round((p / 100) * (len(s) - 1)))
    return s[k]


# ── Report ───────────────────────────────────────────────────────────
def write_markdown_report(stats: list[dict], out_path: Path) -> None:
    lines = ["# Polish text sources — comparison\n"]
    lines.append(
        "| source | n | median words | p10/p90 | vocab | TTR | diacritic% | ascii-only% |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in stats:
        if "error" in s:
            lines.append(f"| {s['name']} | 0 | — | — | — | — | — | — | (load failed) |")
            continue
        lines.append(
            f"| {s['name']} | {s['n_sampled']} | {s['median_words']} "
            f"| {s['p10_words']}/{s['p90_words']} | {s['vocab_size']} "
            f"| {s['type_token_ratio']} | {int(s['diacritic_ratio']*100)}% "
            f"| {int(s['ascii_only_ratio']*100)}% |"
        )

    lines.append("\n**Quick read:**")
    lines.append("- High **diacritic%** (>70%) = text really is Polish.")
    lines.append("- High **ascii-only%** (>20%) = code-switching to English suspected.")
    lines.append("- Very low **TTR** = repetitive vocab; very high TTR = noisy/diverse text.")
    lines.append("")

    for s in stats:
        if "error" in s:
            continue
        lines.append(f"## {s['name']}")
        lines.append(f"\nTop words: `{', '.join(s['top_words'][:15])}`\n")
        lines.append("Shortest examples:")
        for t in s["examples_short"]:
            lines.append(f"- `{t}`")
        lines.append("\nLongest examples:")
        for t in s["examples_long"]:
            lines.append(f"- `{t}`")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", out_path)


# ── CLI ──────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", nargs="+", required=True, help="NAME:ID[:CONFIG][:SPLIT]")
    p.add_argument("--language", default="pl", help="Language to filter (when source has a lang column)")
    p.add_argument("--sample-size", type=int, default=500)
    p.add_argument("--output-dir", default="results/text_sources_inspection")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []
    for spec in args.sources:
        name = spec.split(":")[0]
        try:
            texts = load_source(spec, args.sample_size, args.language)
            stats = compute_stats(name, texts)
            (out_dir / f"{name}.json").write_text(
                json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            (out_dir / f"{name}_samples.txt").write_text(
                "\n".join(texts), encoding="utf-8"
            )
            all_stats.append(stats)
            logger.info(
                "%s: n=%d, diacritic=%.0f%%, ascii_only=%.0f%%",
                name,
                stats["n_sampled"],
                stats["diacritic_ratio"] * 100,
                stats["ascii_only_ratio"] * 100,
            )
        except Exception as e:
            logger.error("%s failed: %s", name, e, exc_info=True)
            all_stats.append({"name": name, "error": str(e)})

    write_markdown_report(all_stats, out_dir / "comparison.md")
    print(f"\nReport: {out_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
