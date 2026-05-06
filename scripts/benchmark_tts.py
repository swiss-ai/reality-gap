#!/usr/bin/env python3
"""TTS benchmark harness: compare backends on a fixed sentence set.

Two phases (split so generation and scoring can run on different nodes):

  1. generate: text + reference audio -> WAV per (backend, sentence) + JSON manifest
  2. score:    WAVs -> WER (Whisper), speaker similarity (ECAPA), RTF, optionally UTMOS

Per-language winners are picked from the aggregated comparison table.

Usage:
    # Phase 1 (GPU node, per backend):
    python scripts/benchmark_tts.py generate \\
        --backend xtts --language pl \\
        --sentences-file data/tts_bench/pl_50.json \\
        --reference-audio /capstor/.../fleurs_cache/pl_pl \\
        --output-dir results/tts_bench/

    # Phase 2 (any node with GPU for Whisper):
    python scripts/benchmark_tts.py score \\
        --output-dir results/tts_bench/ --language pl

    # Phase 3 (aggregation):
    python scripts/benchmark_tts.py aggregate --output-dir results/tts_bench/
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torchaudio

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from speech_generation import PolishTextNormalizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Backend registry ──────────────────────────────────────────────────
# Add new backends here. Each entry returns an instantiated TTSBackend.
def _build_xtts(args):
    from speech_generation.backends.xtts_tts import XTTSTTSBackend

    return XTTSTTSBackend(
        checkpoint=args.checkpoint or "tts_models/multilingual/multi-dataset/xtts_v2",
        device=args.device,
        language=args.language,
    )


def _build_cosyvoice2(args):
    from speech_generation.backends.cosyvoice2_tts import CosyVoice2TTSBackend

    return CosyVoice2TTSBackend(
        checkpoint=args.checkpoint or "iic/CosyVoice2-0.5B",
        device=args.device,
        mode="cross_lingual",
    )


def _build_mms_tts(args):
    from speech_generation.backends.mms_tts import MMSTTSBackend

    iso3 = {"pl": "pol", "en": "eng", "de": "deu", "fr": "fra"}.get(args.language, args.language)
    return MMSTTSBackend(language=iso3, device=args.device)


BACKENDS = {
    "xtts": _build_xtts,
    "cosyvoice2": _build_cosyvoice2,
    "mms_tts": _build_mms_tts,
    # TODO: f5_tts, maskgct, kokoro
}


# ── Reference audio loading ───────────────────────────────────────────
def load_reference_audios(paths: list[str]) -> list[tuple[torch.Tensor, int]]:
    refs = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            audio_files = sorted(p.glob("*.wav")) + sorted(p.glob("*.flac"))
            if not audio_files:
                try:
                    from datasets import load_from_disk

                    ds = load_from_disk(str(p))
                    for i, sample in enumerate(ds):
                        if i >= 3:
                            break
                        a = sample["audio"]
                        refs.append(
                            (torch.tensor(a["array"], dtype=torch.float32), a["sampling_rate"])
                        )
                    continue
                except Exception as e:
                    logger.warning("Could not load %s: %s", p, e)
                    continue
            for af in audio_files[:3]:
                audio, sr = torchaudio.load(str(af))
                refs.append((audio.mean(dim=0), sr))
        elif p.is_file():
            audio, sr = torchaudio.load(str(p))
            refs.append((audio.mean(dim=0), sr))
    return refs


# ── Phase 1: generate ─────────────────────────────────────────────────
def cmd_generate(args):
    out_dir = Path(args.output_dir) / args.language / args.backend
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.sentences_file) as f:
        sentences = json.load(f)
    logger.info("Loaded %d sentences", len(sentences))

    normalizer = _normalizer_for(args.language)
    if normalizer:
        for s in sentences:
            s["normalized_text"] = normalizer.normalize(s["text"])
    else:
        for s in sentences:
            s["normalized_text"] = s["text"]

    refs = load_reference_audios(args.reference_audio)
    if not refs:
        logger.error("No reference audio loaded")
        sys.exit(1)
    logger.info("Loaded %d reference clips", len(refs))

    if args.backend not in BACKENDS:
        logger.error("Unknown backend %r. Available: %s", args.backend, list(BACKENDS))
        sys.exit(2)
    backend = BACKENDS[args.backend](args)
    backend.load_model()

    manifest = []
    t_start = time.time()
    for i, s in enumerate(sentences):
        ref_audio, ref_sr = refs[i % len(refs)]
        sid = s["id"]
        wav_path = out_dir / f"{sid}.wav"

        t0 = time.time()
        try:
            output = backend.generate(
                text=s["normalized_text"],
                reference_audio=ref_audio,
                reference_audio_sr=ref_sr,
                render_audio=True,
            )
            elapsed = time.time() - t0

            audio = output.audio
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
            torchaudio.save(str(wav_path), audio.cpu(), output.audio_sample_rate)

            manifest.append(
                {
                    "id": sid,
                    "source_text": s["text"],
                    "normalized_text": s["normalized_text"],
                    "wav_path": str(wav_path.relative_to(args.output_dir)),
                    "speaker_idx": i % len(refs),
                    "duration_seconds": round(output.duration_seconds, 3)
                    if output.token_rate_hz > 0
                    else round(audio.shape[-1] / output.audio_sample_rate, 3),
                    "generation_seconds": round(elapsed, 3),
                    "rtf": round(
                        elapsed / max(audio.shape[-1] / output.audio_sample_rate, 1e-6), 3
                    ),
                    "metadata": output.metadata,
                }
            )
            logger.info("[%d/%d] %s: %.2fs", i + 1, len(sentences), sid, elapsed)
        except Exception as e:
            logger.error("[%d/%d] %s FAILED: %s", i + 1, len(sentences), sid, e, exc_info=True)
            manifest.append({"id": sid, "error": str(e)})

    summary = {
        "backend": args.backend,
        "language": args.language,
        "n_sentences": len(sentences),
        "n_succeeded": sum(1 for m in manifest if "error" not in m),
        "wall_time_seconds": round(time.time() - t_start, 2),
        "samples": manifest,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(
        "Wrote %d/%d to %s", summary["n_succeeded"], summary["n_sentences"], out_dir
    )


def _normalizer_for(language: str):
    if language == "pl":
        return PolishTextNormalizer()
    return None  # TODO: en, de, fr, ...


# ── Phase 2: score ────────────────────────────────────────────────────
def cmd_score(args):
    lang_dir = Path(args.output_dir) / args.language
    backend_dirs = [d for d in lang_dir.iterdir() if d.is_dir()]
    if not backend_dirs:
        logger.error("No backend output found under %s", lang_dir)
        sys.exit(1)

    whisper_model = _load_whisper(args.whisper_model, args.device) if not args.skip_wer else None
    spk_model = _load_speaker_encoder(args.device) if not args.skip_speaker else None

    for bd in backend_dirs:
        manifest_path = bd / "manifest.json"
        if not manifest_path.exists():
            logger.warning("Skipping %s — no manifest", bd)
            continue
        with open(manifest_path) as f:
            manifest = json.load(f)

        for sample in manifest["samples"]:
            if "error" in sample:
                continue
            wav_path = Path(args.output_dir) / sample["wav_path"]
            if not wav_path.exists():
                logger.warning("Missing WAV: %s", wav_path)
                continue

            if whisper_model:
                sample["wer"] = _compute_wer(
                    whisper_model, wav_path, sample["normalized_text"], args.language
                )
            if spk_model:
                ref_idx = sample["speaker_idx"]
                # TODO: cache reference embeddings; for now recompute via wav_path
                # of the speaker (skip until reference paths are wired in)
                sample["speaker_similarity"] = None

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        logger.info("Scored %s", bd.name)


def _load_whisper(model_name: str, device: str):
    import whisper

    return whisper.load_model(model_name, device=device)


def _compute_wer(model, wav_path: Path, reference_text: str, language: str) -> Optional[float]:
    try:
        result = model.transcribe(str(wav_path), language=language, fp16=False)
        hyp = result["text"].strip().lower()
        ref = reference_text.strip().lower()
        return _wer(ref.split(), hyp.split())
    except Exception as e:
        logger.warning("WER failed for %s: %s", wav_path.name, e)
        return None


def _wer(ref: list[str], hyp: list[str]) -> float:
    """Levenshtein-based WER. Trivial implementation; for large eval use jiwer."""
    if not ref:
        return 0.0 if not hyp else 1.0
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return round(dp[n][m] / n, 4)


def _load_speaker_encoder(device: str):
    """ECAPA-TDNN via SpeechBrain. Stubbed — wire up when speaker-sim is needed."""
    # TODO: from speechbrain.pretrained import EncoderClassifier
    return None


# ── Phase 3: aggregate ────────────────────────────────────────────────
def cmd_aggregate(args):
    root = Path(args.output_dir)
    rows = []
    for lang_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for backend_dir in sorted(p for p in lang_dir.iterdir() if p.is_dir()):
            mf = backend_dir / "manifest.json"
            if not mf.exists():
                continue
            with open(mf) as f:
                m = json.load(f)
            ok = [s for s in m["samples"] if "error" not in s]
            wers = [s["wer"] for s in ok if s.get("wer") is not None]
            rtfs = [s["rtf"] for s in ok if "rtf" in s]
            rows.append(
                {
                    "language": lang_dir.name,
                    "backend": backend_dir.name,
                    "n_ok": len(ok),
                    "n_failed": len(m["samples"]) - len(ok),
                    "wer_mean": round(sum(wers) / len(wers), 4) if wers else None,
                    "rtf_mean": round(sum(rtfs) / len(rtfs), 3) if rtfs else None,
                }
            )

    out_path = root / "comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"\n{'language':<8} {'backend':<14} {'n_ok':>6} {'wer':>8} {'rtf':>8}")
    print("-" * 50)
    for r in rows:
        print(
            f"{r['language']:<8} {r['backend']:<14} {r['n_ok']:>6} "
            f"{(r['wer_mean'] if r['wer_mean'] is not None else '—'):>8} "
            f"{(r['rtf_mean'] if r['rtf_mean'] is not None else '—'):>8}"
        )
    print(f"\nWrote {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate WAVs for one backend")
    g.add_argument("--backend", required=True, choices=list(BACKENDS))
    g.add_argument("--language", required=True, help="ISO code, e.g. 'pl'")
    g.add_argument("--sentences-file", required=True, help="JSON list of {id, text}")
    g.add_argument("--reference-audio", nargs="+", required=True)
    g.add_argument("--output-dir", default="results/tts_bench")
    g.add_argument("--checkpoint", default=None, help="Override default checkpoint")
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    s = sub.add_parser("score", help="Compute WER / speaker-sim on generated WAVs")
    s.add_argument("--output-dir", default="results/tts_bench")
    s.add_argument("--language", required=True)
    s.add_argument("--whisper-model", default="large-v3")
    s.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    s.add_argument("--skip-wer", action="store_true")
    s.add_argument("--skip-speaker", action="store_true")

    a = sub.add_parser("aggregate", help="Build comparison table")
    a.add_argument("--output-dir", default="results/tts_bench")

    args = p.parse_args()
    if args.cmd == "generate":
        cmd_generate(args)
    elif args.cmd == "score":
        cmd_score(args)
    elif args.cmd == "aggregate":
        cmd_aggregate(args)


if __name__ == "__main__":
    main()
