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
# Only commercially-usable TTS backends (CC-BY-4.0 no SA/NC, MIT, Apache 2.0)
# are registered for active runs. Backends with non-commercial licenses live in
# REFERENCE_BACKENDS — they can still be invoked explicitly via --backend
# <name> --allow-reference for comparison baselines, but cannot be used to
# produce training data.
def _build_cosyvoice2(args):
    from speech_generation.backends.cosyvoice2_tts import CosyVoice2TTSBackend

    return CosyVoice2TTSBackend(
        checkpoint=args.checkpoint or "iic/CosyVoice2-0.5B",
        device=args.device,
        mode="cross_lingual",
    )


def _build_omnivoice(args):
    from speech_generation.backends.omnivoice_tts import OmniVoiceTTSBackend

    return OmniVoiceTTSBackend(
        checkpoint=args.checkpoint or "k2-fsa/OmniVoice",
        device=args.device,
    )


def _build_voxcpm2(args):
    from speech_generation.backends.voxcpm2_tts import VoxCPM2TTSBackend

    return VoxCPM2TTSBackend(
        checkpoint=args.checkpoint or "openbmb/VoxCPM2",
        device=args.device,
    )


def _build_piper(args):
    from speech_generation.backends.piper_tts import PiperTTSBackend

    return PiperTTSBackend(
        voice_path=args.checkpoint or "voices/pl_PL-gosia-medium.onnx",
        device=args.device,
    )


def _build_parler(args):
    from speech_generation.backends.parler_tts import ParlerTTSBackend

    return ParlerTTSBackend(
        checkpoint=args.checkpoint or "parler-tts/parler-tts-mini-multilingual-v1.1",
        device=args.device,
    )


def _build_f5(args):
    from speech_generation.backends.f5_tts import F5TTSBackend

    # Default to the Sticzu/marek-f5tts-polish Polish fine-tune (MIT, native PL).
    # Pass --checkpoint to override (e.g. "F5TTS_v1_Base" for EN/ZH base).
    return F5TTSBackend(
        checkpoint=args.checkpoint or "Sticzu/marek-f5tts-polish",
        device=args.device,
    )


def _build_qwen_omni(args):
    from speech_generation.backends.qwen_omni_tts import QwenOmniTTSBackend

    # Qwen2.5-Omni uses built-in voices, not zero-shot cloning.
    # speaker can be set via --checkpoint as "<model>:<voice>" if needed
    # (rare); default voice is per-language ("Chelsie" for zh).
    return QwenOmniTTSBackend(
        checkpoint=args.checkpoint or "Qwen/Qwen2.5-Omni-7B",
        device=args.device,
    )


def _build_indextts(args):
    from speech_generation.backends.indextts_tts import IndexTTSBackend

    return IndexTTSBackend(
        checkpoint=args.checkpoint or "IndexTeam/IndexTTS-1.5",
        device=args.device,
    )


def _build_melotts(args):
    from speech_generation.backends.melotts_tts import MeloTTSBackend

    # MeloTTS picks language from args, defaults handle pl/zh/en.
    return MeloTTSBackend(
        language=args.language,
        device=args.device,
    )


# Non-commercial backends, kept for reference comparisons only.
def _build_xtts(args):
    from speech_generation.backends.xtts_tts import XTTSTTSBackend

    return XTTSTTSBackend(
        checkpoint=args.checkpoint or "tts_models/multilingual/multi-dataset/xtts_v2",
        device=args.device,
        language=args.language,
    )


def _build_mms_tts(args):
    from speech_generation.backends.mms_tts import MMSTTSBackend

    iso3 = {"pl": "pol", "en": "eng", "de": "deu", "fr": "fra"}.get(args.language, args.language)
    return MMSTTSBackend(language=iso3, device=args.device)


BACKENDS = {
    "cosyvoice2": _build_cosyvoice2,   # Apache 2.0 — bad Polish quality (cross-lingual baseline)
    "omnivoice":  _build_omnivoice,    # Apache 2.0 — multilingual + voice cloning
    "voxcpm2":    _build_voxcpm2,      # Apache 2.0 — supervisor's pick, vLLM-servable
    "piper":      _build_piper,        # MIT — Polish-native, fixed-voice, lightweight
    "parler":     _build_parler,       # Apache 2.0 — text-description-controlled
    "f5":         _build_f5,           # MIT (code + Sticzu Polish fine-tune weights)
    "qwen_omni":  _build_qwen_omni,    # Apache 2.0 — multimodal LLM with built-in zh voices
    "indextts":   _build_indextts,     # Apache 2.0 — BiliBili, zh-native, voice cloning
    "melotts":    _build_melotts,      # MIT — MyShell, multilingual, built-in voices
}

# Available only with --allow-reference. License disqualifies these from
# producing training data, but they're useful as quality baselines.
REFERENCE_BACKENDS = {
    "xtts": _build_xtts,         # Coqui non-commercial license
    "mms_tts": _build_mms_tts,   # CC-BY-NC 4.0  (baseline: WER 0.19 on pl_50)
}

BACKEND_LICENSES = {
    "cosyvoice2": "Apache-2.0",
    "omnivoice":  "Apache-2.0",
    "voxcpm2":    "Apache-2.0",
    "piper":      "MIT",
    "parler":     "Apache-2.0",
    "f5":         "MIT",
    "qwen_omni":  "Apache-2.0",
    "indextts":   "Apache-2.0",
    "melotts":    "MIT",
    "xtts":       "Coqui-Public-Model-License (non-commercial)",
    "mms_tts":    "CC-BY-NC-4.0",
}


# ── Reference audio loading ───────────────────────────────────────────
def load_reference_audios(
    paths: list[str],
) -> list[tuple[torch.Tensor, int, Optional[str]]]:
    """Load reference clips. If a sibling .txt file exists, its contents are
    returned as the reference transcript (needed by OmniVoice; ignored by others)."""

    def _maybe_transcript(audio_path: Path) -> Optional[str]:
        txt = audio_path.with_suffix(".txt")
        if txt.exists():
            return txt.read_text(encoding="utf-8").strip()
        return None

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
                        text = sample.get("transcription") or sample.get("text")
                        refs.append(
                            (
                                torch.tensor(a["array"], dtype=torch.float32),
                                a["sampling_rate"],
                                text,
                            )
                        )
                    continue
                except Exception as e:
                    logger.warning("Could not load %s: %s", p, e)
                    continue
            for af in audio_files[:3]:
                audio, sr = torchaudio.load(str(af))
                refs.append((audio.mean(dim=0), sr, _maybe_transcript(af)))
        elif p.is_file():
            audio, sr = torchaudio.load(str(p))
            refs.append((audio.mean(dim=0), sr, _maybe_transcript(p)))
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

    if args.backend in BACKENDS:
        build_fn = BACKENDS[args.backend]
    elif args.backend in REFERENCE_BACKENDS:
        if not getattr(args, "allow_reference", False):
            logger.error(
                "Backend %r has a non-commercial license — pass --allow-reference "
                "to run it as a comparison baseline (cannot be used for training data).",
                args.backend,
            )
            sys.exit(2)
        logger.warning(
            "Running reference-only backend %r (non-commercial license). "
            "Output is for comparison only, not training data.",
            args.backend,
        )
        build_fn = REFERENCE_BACKENDS[args.backend]
    else:
        logger.error(
            "Unknown backend %r. Available: %s   (reference-only: %s)",
            args.backend, list(BACKENDS), list(REFERENCE_BACKENDS),
        )
        sys.exit(2)
    backend = build_fn(args)
    backend.load_model()

    manifest = []
    t_start = time.time()
    for i, s in enumerate(sentences):
        ref_audio, ref_sr, ref_text = refs[i % len(refs)]
        sid = s["id"]
        wav_path = out_dir / f"{sid}.wav"

        t0 = time.time()
        try:
            output = backend.generate(
                text=s["normalized_text"],
                reference_audio=ref_audio,
                reference_audio_sr=ref_sr,
                render_audio=True,
                ref_text=ref_text,
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

    ok_samples = [m for m in manifest if "error" not in m]
    total_audio = sum(m.get("duration_seconds", 0.0) for m in ok_samples)
    total_gen = sum(m.get("generation_seconds", 0.0) for m in ok_samples)

    summary = {
        "backend": args.backend,
        "language": args.language,
        "license": BACKEND_LICENSES.get(args.backend, "Unknown"),
        "commercial_usable": args.backend in BACKENDS,
        "n_sentences": len(sentences),
        "n_succeeded": len(ok_samples),
        "wall_time_seconds": round(time.time() - t_start, 2),
        "total_audio_seconds": round(total_audio, 2),
        "total_generation_seconds": round(total_gen, 2),
        # aggregate RTF: GPU-seconds per audio-second across the whole run
        # (less noisy than averaging per-sample RTFs; what matters for cost projection)
        "aggregate_rtf": round(total_gen / total_audio, 4) if total_audio > 0 else None,
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
                metrics = _compute_metrics(
                    whisper_model, wav_path, sample["normalized_text"], args.language
                )
                sample["wer"] = metrics["wer"]
                sample["cer"] = metrics["cer"]
                sample["per"] = metrics["per"]
                sample["hyp"] = metrics["hyp"]
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


def _edit_distance_norm(ref: list, hyp: list) -> float:
    """Levenshtein distance / len(ref). Works for any token sequence: words
    (WER), chars (CER), or phonemes (PER)."""
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


# Backwards-compat alias; existing callers (if any) keep working.
_wer = _edit_distance_norm


_EPITRAN_PL = None


def _g2p_polish(text: str) -> list[str]:
    """Polish grapheme→phoneme via epitran (pol-Latn). Returns IPA chars.

    Each Unicode codepoint of the IPA output is a token. Affricates like
    "t͡ʂ" (cz) get split across multiple tokens but the edit distance is
    still meaningful for measuring pronunciation correctness — which is
    what we care about when "się" comes out as "sze".
    """
    global _EPITRAN_PL
    if _EPITRAN_PL is None:
        import epitran  # pip install epitran (pure-python, no system deps)
        _EPITRAN_PL = epitran.Epitran("pol-Latn")
    return list(_EPITRAN_PL.transliterate(text))


def _compute_metrics(
    model, wav_path: Path, reference_text: str, language: str
) -> dict:
    """Transcribe with Whisper + compute WER/CER/PER vs reference.

    Returns dict with wer (word-level), cer (char-level, whitespace-stripped),
    per (IPA phoneme-level via Polish G2P), and hyp (the transcript).
    PER is None for non-Polish languages or if epitran isn't installed.
    """
    import re

    try:
        audio, sr = torchaudio.load(str(wav_path))
        if audio.ndim > 1:
            audio = audio.mean(dim=0)
        if sr != 16000:
            audio = torchaudio.transforms.Resample(sr, 16000)(audio)
        audio_np = audio.numpy().astype("float32")
        result = model.transcribe(audio_np, language=language, fp16=False)
        hyp = result["text"].strip().lower()
        ref = reference_text.strip().lower()

        wer = _edit_distance_norm(ref.split(), hyp.split())

        # CER on whitespace-stripped char sequences. "się" vs "sie" is 1 char
        # diff (CER 0.33 on a 3-char ref), but WER counts it as a whole word
        # wrong (WER 1.0).
        ref_chars = list(re.sub(r"\s+", "", ref))
        hyp_chars = list(re.sub(r"\s+", "", hyp))
        cer = _edit_distance_norm(ref_chars, hyp_chars)

        # PER via Polish G2P — directly measures pronunciation independent of
        # orthography. Catches "sze" instead of "się" which CER underweights.
        per: Optional[float] = None
        if language == "pl":
            try:
                per = _edit_distance_norm(_g2p_polish(ref), _g2p_polish(hyp))
            except Exception as e:
                logger.debug("PER skipped for %s: %s", wav_path.name, e)

        return {"wer": wer, "cer": cer, "per": per, "hyp": hyp}
    except Exception as e:
        logger.warning("metrics failed for %s: %s", wav_path.name, e)
        return {"wer": None, "cer": None, "per": None, "hyp": None}


def _compute_wer(model, wav_path: Path, reference_text: str, language: str) -> Optional[float]:
    """Backwards-compat shim around _compute_metrics — returns just WER."""
    return _compute_metrics(model, wav_path, reference_text, language)["wer"]


def _load_speaker_encoder(device: str):
    """ECAPA-TDNN via SpeechBrain. Stubbed — wire up when speaker-sim is needed."""
    # TODO: from speechbrain.pretrained import EncoderClassifier
    return None


# ── Phase 3: aggregate ────────────────────────────────────────────────
def cmd_aggregate(args):
    root = Path(args.output_dir)
    rows = []
    breakdowns = []  # per (lang, backend, category) WER

    # Try to load category labels from sentence files for per-category breakdown
    sentence_categories: dict[str, dict[str, str]] = {}
    sentences_dir = Path("data/tts_bench")
    if sentences_dir.exists():
        for f in sentences_dir.glob("*.json"):
            try:
                items = json.loads(f.read_text(encoding="utf-8"))
                lang = f.stem.split("_")[0]
                sentence_categories.setdefault(lang, {}).update(
                    {s["id"]: s.get("category", "uncategorized") for s in items}
                )
            except Exception:
                pass

    for lang_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for backend_dir in sorted(p for p in lang_dir.iterdir() if p.is_dir()):
            mf = backend_dir / "manifest.json"
            if not mf.exists():
                continue
            with open(mf) as f:
                m = json.load(f)
            ok = [s for s in m["samples"] if "error" not in s]
            wers = [s["wer"] for s in ok if s.get("wer") is not None]
            cers = [s["cer"] for s in ok if s.get("cer") is not None]
            pers = [s["per"] for s in ok if s.get("per") is not None]
            rtfs = [s["rtf"] for s in ok if "rtf" in s]
            rows.append(
                {
                    "language": lang_dir.name,
                    "backend": backend_dir.name,
                    "license": m.get("license", BACKEND_LICENSES.get(backend_dir.name, "Unknown")),
                    "commercial_usable": m.get("commercial_usable", backend_dir.name in BACKENDS),
                    "n_ok": len(ok),
                    "n_failed": len(m["samples"]) - len(ok),
                    "wer_mean": round(sum(wers) / len(wers), 4) if wers else None,
                    "cer_mean": round(sum(cers) / len(cers), 4) if cers else None,
                    "per_mean": round(sum(pers) / len(pers), 4) if pers else None,
                    "rtf_mean": round(sum(rtfs) / len(rtfs), 3) if rtfs else None,
                    "aggregate_rtf": m.get("aggregate_rtf"),
                    "total_audio_seconds": m.get("total_audio_seconds"),
                    "total_generation_seconds": m.get("total_generation_seconds"),
                }
            )

            cats = sentence_categories.get(lang_dir.name, {})
            if cats:
                from collections import defaultdict

                w_buckets: dict[str, list[float]] = defaultdict(list)
                c_buckets: dict[str, list[float]] = defaultdict(list)
                p_buckets: dict[str, list[float]] = defaultdict(list)
                for s in ok:
                    cat = cats.get(s["id"], "uncategorized")
                    if s.get("wer") is not None:
                        w_buckets[cat].append(s["wer"])
                    if s.get("cer") is not None:
                        c_buckets[cat].append(s["cer"])
                    if s.get("per") is not None:
                        p_buckets[cat].append(s["per"])
                all_cats = set(w_buckets) | set(c_buckets) | set(p_buckets)
                for c in sorted(all_cats):
                    ws, cs, ps = w_buckets[c], c_buckets[c], p_buckets[c]
                    breakdowns.append(
                        {
                            "language": lang_dir.name,
                            "backend": backend_dir.name,
                            "category": c,
                            "n": max(len(ws), len(cs), len(ps)),
                            "wer_mean": round(sum(ws) / len(ws), 4) if ws else None,
                            "cer_mean": round(sum(cs) / len(cs), 4) if cs else None,
                            "per_mean": round(sum(ps) / len(ps), 4) if ps else None,
                        }
                    )

    out_path = root / "comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": rows, "breakdown": breakdowns}, f, indent=2, ensure_ascii=False)

    print(
        f"\n{'lang':<5} {'backend':<14} {'license':<18} {'n_ok':>5} "
        f"{'wer':>7} {'cer':>7} {'per':>7} {'rtf':>6} {'agg_rtf':>8}"
    )
    print("-" * 88)
    for r in rows:
        wer = r["wer_mean"] if r["wer_mean"] is not None else "—"
        cer = r["cer_mean"] if r["cer_mean"] is not None else "—"
        per = r["per_mean"] if r["per_mean"] is not None else "—"
        rtf = r["rtf_mean"] if r["rtf_mean"] is not None else "—"
        agg = r.get("aggregate_rtf") if r.get("aggregate_rtf") is not None else "—"
        print(
            f"{r['language']:<5} {r['backend']:<14} "
            f"{(r.get('license') or 'Unknown')[:18]:<18} "
            f"{r['n_ok']:>5} {wer:>7} {cer:>7} {per:>7} {rtf:>6} {agg:>8}"
        )

    if breakdowns:
        print(f"\nPer-category WER/CER:")
        print(f"{'lang':<5} {'backend':<14} {'category':<16} {'n':>4} "
              f"{'wer':>7} {'cer':>7} {'per':>7}")
        print("-" * 68)
        for b in breakdowns:
            cer = b.get("cer_mean") if b.get("cer_mean") is not None else "—"
            per = b.get("per_mean") if b.get("per_mean") is not None else "—"
            print(
                f"{b['language']:<5} {b['backend']:<14} {b['category']:<16} "
                f"{b['n']:>4} {b['wer_mean']:>7} {cer:>7} {per:>7}"
            )

    print(f"\nWrote {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate WAVs for one backend")
    g.add_argument(
        "--backend", required=True,
        choices=list(BACKENDS) + list(REFERENCE_BACKENDS),
        help="Commercially-usable: " + ", ".join(BACKENDS) + ". "
             "Reference-only (non-commercial): " + ", ".join(REFERENCE_BACKENDS),
    )
    g.add_argument("--language", required=True, help="ISO code, e.g. 'pl'")
    g.add_argument("--sentences-file", required=True, help="JSON list of {id, text}")
    g.add_argument("--reference-audio", nargs="+", required=True)
    g.add_argument("--output-dir", default="results/tts_bench")
    g.add_argument("--checkpoint", default=None, help="Override default checkpoint")
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument(
        "--allow-reference", action="store_true",
        help="Allow running a non-commercial reference backend as a baseline. "
             "Output is not eligible for training data.",
    )

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
