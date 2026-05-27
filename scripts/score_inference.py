#!/usr/bin/env python3
"""Score audio_inference.py output JSONs with WER (Latin) / CER (CJK).

Two modes:

  # single-run mode — score one inference JSON, print per-record + aggregate:
  python scripts/score_inference.py --run path/to/fleurs_pl/<ckpt>_transcribe.json

  # gap mode — two runs over the same test set, prints WER(synth) − WER(real):
  python scripts/score_inference.py \\
      --synth-trained synth-trained/<ckpt_A>_transcribe.json \\
      --real-trained  real-trained/<ckpt_B>_transcribe.json

  # batch mode — many runs at once, summary CSV emitted:
  python scripts/score_inference.py --root results/inference_eval/ \\
      --out results/scores.csv
"""
from __future__ import annotations

import argparse, csv, json, re, sys, unicodedata
from collections import defaultdict
from pathlib import Path

# Repo root for imports
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from audio_tokenization.contracts.prediction import read_inference_run, InferenceRun


# --- Normalization ----------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s一-鿿]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def is_cjk(s: str) -> bool:
    """True if more than half the alpha chars are CJK (zh/ja/ko ranges)."""
    cjk = sum(1 for c in s if "一" <= c <= "鿿")
    alpha = sum(1 for c in s if c.isalpha() or "一" <= c <= "鿿")
    return alpha > 0 and cjk / alpha > 0.5


def normalize_latin(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = _PUNCT_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def normalize_cjk(s: str) -> str:
    """Strip whitespace + punctuation, keep CJK chars (and Latin letters if mixed)."""
    s = unicodedata.normalize("NFKC", s)
    s = _PUNCT_RE.sub("", s)
    s = _WS_RE.sub("", s)
    return s.strip()


# --- Edit distance (Levenshtein) -------------------------------------------

def edit_distance(a: list, b: list) -> int:
    """Standard DP edit distance over arbitrary token sequences."""
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + (ca != cb),  # substitution
            )
        prev = curr
    return prev[-1]


# --- Per-record scoring -----------------------------------------------------

def score_record(reference: str, prediction: str, lang_hint: str | None = None) -> dict:
    """Return {metric, ref_len, errors, score}. Metric: 'wer' or 'cer'."""
    use_cer = (lang_hint or "").startswith("zh") or is_cjk(reference)
    if use_cer:
        ref_norm = normalize_cjk(reference)
        hyp_norm = normalize_cjk(prediction)
        ref_units = list(ref_norm)
        hyp_units = list(hyp_norm)
        metric = "cer"
    else:
        ref_norm = normalize_latin(reference)
        hyp_norm = normalize_latin(prediction)
        ref_units = ref_norm.split()
        hyp_units = hyp_norm.split()
        metric = "wer"

    if not ref_units:
        return {"metric": metric, "ref_len": 0, "errors": len(hyp_units),
                "score": float("nan")}
    errors = edit_distance(ref_units, hyp_units)
    return {"metric": metric, "ref_len": len(ref_units),
            "errors": errors, "score": errors / len(ref_units)}


# --- Per-run aggregation ----------------------------------------------------

def score_run(run: InferenceRun, lang_hint: str | None = None) -> dict:
    """Aggregate WER/CER across all records in a run.

    Returns: {metric, n_records, total_errors, total_ref_len, score}
    where `score` is corpus-level (sum errors / sum ref units), not mean-of-rates.
    Mean-of-rates is also returned as `mean_score` for sanity-check.
    """
    if not run.records:
        return {"metric": "?", "n_records": 0, "total_errors": 0,
                "total_ref_len": 0, "score": float("nan"),
                "mean_score": float("nan")}

    # Decide metric once per run from the first non-empty ref:
    metric = None
    for r in run.records:
        if r.reference_text:
            metric = "cer" if (
                (lang_hint or "").startswith("zh") or is_cjk(r.reference_text)
            ) else "wer"
            break
    metric = metric or "wer"

    total_err = 0
    total_ref = 0
    per_rate = []
    skipped = 0
    for r in run.records:
        if not r.reference_text:
            skipped += 1
            continue
        s = score_record(r.reference_text, r.prediction_text, lang_hint)
        if s["ref_len"] == 0:
            skipped += 1
            continue
        total_err += s["errors"]
        total_ref += s["ref_len"]
        per_rate.append(s["score"])

    corpus = total_err / total_ref if total_ref else float("nan")
    mean = sum(per_rate) / len(per_rate) if per_rate else float("nan")
    return {
        "metric": metric, "n_records": len(per_rate),
        "n_skipped": skipped,
        "total_errors": total_err, "total_ref_len": total_ref,
        "score": corpus, "mean_score": mean,
    }


# --- CLI --------------------------------------------------------------------

def _lang_from_name(name: str) -> str | None:
    """Heuristic: pull language hint from dataset_name like 'fleurs_pl_pl', 'cv25_zh'."""
    n = name.lower()
    if "pl_pl" in n or "_pl" in n or n.endswith("pl") or "polish" in n:
        return "pl"
    if "zh" in n or "cmn" in n or "chinese" in n or "aishell" in n:
        return "zh"
    return None


def _fmt(v): return f"{v*100:.2f}%" if isinstance(v, float) and v == v else "N/A"


def _print_run(path: Path, run: InferenceRun, score: dict):
    print(f"\n== {path} ==")
    print(f"  dataset_name: {run.dataset_name}")
    print(f"  model_path:   {run.model_path}")
    print(f"  task:         {run.task}  backend={run.backend}  temp={run.temperature}")
    print(f"  records:      {len(run.records)} (scored={score['n_records']}, "
          f"skipped={score['n_skipped']})")
    print(f"  {score['metric'].upper()} (corpus): {_fmt(score['score'])}   "
          f"({score['total_errors']}/{score['total_ref_len']})")
    print(f"  {score['metric'].upper()} (mean):   {_fmt(score['mean_score'])}")


def cmd_run(args):
    run = read_inference_run(Path(args.run))
    lang = args.lang or _lang_from_name(run.dataset_name)
    score = score_run(run, lang)
    _print_run(Path(args.run), run, score)


def cmd_gap(args):
    synth = read_inference_run(Path(args.synth_trained))
    real = read_inference_run(Path(args.real_trained))
    if synth.dataset_name != real.dataset_name:
        print(f"WARNING: dataset names differ: synth={synth.dataset_name} "
              f"real={real.dataset_name}", file=sys.stderr)
    lang = args.lang or _lang_from_name(synth.dataset_name)
    s_synth = score_run(synth, lang)
    s_real = score_run(real, lang)
    _print_run(Path(args.synth_trained), synth, s_synth)
    _print_run(Path(args.real_trained), real, s_real)
    metric = s_synth["metric"].upper()
    if s_synth["score"] == s_synth["score"] and s_real["score"] == s_real["score"]:
        gap = s_synth["score"] - s_real["score"]
        print(f"\n== REALITY GAP ==")
        print(f"  {metric}(synth-trained)  = {_fmt(s_synth['score'])}")
        print(f"  {metric}(real-trained)   = {_fmt(s_real['score'])}")
        print(f"  gap = synth - real      = {_fmt(gap)}  "
              f"({'synth worse' if gap > 0 else 'synth better/equal'})")


def cmd_batch(args):
    root = Path(args.root)
    files = sorted(root.rglob("*.json"))
    if not files:
        print(f"No JSON files under {root}", file=sys.stderr)
        sys.exit(2)

    rows = []
    for f in files:
        try:
            run = read_inference_run(f)
        except Exception as e:
            print(f"SKIP {f}: {e}", file=sys.stderr)
            continue
        lang = _lang_from_name(run.dataset_name)
        score = score_run(run, lang)
        rows.append({
            "file": str(f.relative_to(root)),
            "dataset": run.dataset_name,
            "model": Path(run.model_path).name,
            "task": run.task,
            "backend": run.backend,
            "n_records": score["n_records"],
            "n_skipped": score["n_skipped"],
            "metric": score["metric"],
            "score_corpus": score["score"],
            "score_mean": score["mean_score"],
            "total_errors": score["total_errors"],
            "total_ref_len": score["total_ref_len"],
        })

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {len(rows)} rows to {out}")
    else:
        print(f"{'dataset':<30} {'model':<40} {'metric':<6} {'score':>10}  {'n':>5}")
        for r in rows:
            print(f"  {r['dataset']:<30} {r['model']:<40} {r['metric']:<6} "
                  f"{_fmt(r['score_corpus']):>10}  {r['n_records']:>5}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang", choices=["pl", "zh"], default=None,
                   help="Override language inference (default: from dataset name).")
    p.add_argument("--run", help="Single inference JSON to score.")
    p.add_argument("--root", help="Batch mode: walk this dir, score every JSON.")
    p.add_argument("--out", default=None, help="(batch only) CSV output path.")
    p.add_argument("--synth-trained", help="(gap mode) JSON from synth-trained model.")
    p.add_argument("--real-trained", help="(gap mode) JSON from real-trained model.")
    args = p.parse_args()

    n_modes = sum(bool(x) for x in [args.run, args.root,
                                    args.synth_trained or args.real_trained])
    if n_modes != 1:
        p.error("Specify exactly one mode: --run | --root | (--synth-trained AND --real-trained)")
    if bool(args.synth_trained) != bool(args.real_trained):
        p.error("--synth-trained and --real-trained must both be set for gap mode")

    if args.synth_trained:
        cmd_gap(args)
    elif args.run:
        cmd_run(args)
    else:
        cmd_batch(args)


if __name__ == "__main__":
    main()
