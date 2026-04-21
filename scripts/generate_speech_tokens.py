#!/usr/bin/env python3
"""Generate discrete speech tokens from Polish text using CosyVoice2 TTS.

Proof-of-concept: processes ~10 Polish sentences, saves speech tokens (.pt),
sidecar metadata (.json), and optionally rendered audio (.wav) for listening.

Usage:
    # Minimal (cluster, with reference audio from FLEURS):
    python scripts/generate_speech_tokens.py \
        --reference-audio /path/to/polish_speaker.wav

    # Full options:
    python scripts/generate_speech_tokens.py \
        --reference-audio /path/to/polish_speaker.wav \
        --output-dir outputs/speech_tokens \
        --render-audio \
        --device cuda \
        --checkpoint iic/CosyVoice2-0.5B

    # Multiple reference speakers (round-robin):
    python scripts/generate_speech_tokens.py \
        --reference-audio speaker1.wav speaker2.wav speaker3.wav \
        --render-audio
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torchaudio

# Add src/ to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from speech_generation import PolishTextNormalizer, save_sample
from speech_generation.backends.cosyvoice2_tts import CosyVoice2TTSBackend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Default Polish sentences ──────────────────────────────────────────
# Cover: everyday speech, numbers, dates, times, abbreviations, formal register
DEFAULT_SENTENCES = [
    {
        "id": "pl_001",
        "text": "Dzień dobry, witam serdecznie.",
    },
    {
        "id": "pl_002",
        "text": "W roku 2026 Polska liczy około 38 milionów mieszkańców.",
    },
    {
        "id": "pl_003",
        "text": "Proszę skręcić w ulicę Marszałkowską pod numer 15.",
    },
    {
        "id": "pl_004",
        "text": "Temperatura wynosi minus 3 stopnie Celsjusza.",
    },
    {
        "id": "pl_005",
        "text": "Spotkanie odbędzie się 20 kwietnia o godzinie 14:30.",
    },
    {
        "id": "pl_006",
        "text": "Doktor Kowalski przyjmuje pacjentów od poniedziałku do piątku.",
    },
    {
        "id": "pl_007",
        "text": "Cena biletu wynosi 25 złotych i 50 groszy.",
    },
    {
        "id": "pl_008",
        "text": "Czy mógłbyś mi pomóc znaleźć drogę na dworzec kolejowy?",
    },
    {
        "id": "pl_009",
        "text": "Profesor Nowak wygłosi wykład na temat sztucznej inteligencji.",
    },
    {
        "id": "pl_010",
        "text": "Na lotnisku Chopina wylądowało 150 samolotów w ciągu jednego dnia.",
    },
]

# Default reference audio on cluster (FLEURS Polish split, first sample)
DEFAULT_REFERENCE = (
    "/capstor/store/cscs/swissai/infra01/audio-datasets/benchmark/fleurs_cache/pl_pl"
)


def load_reference_audios(paths: list[str]) -> list[tuple[torch.Tensor, int]]:
    """Load reference audio files, returning list of (tensor, sr) pairs."""
    refs = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            # Take first few .wav/.flac files from a dataset directory
            audio_files = sorted(p.glob("*.wav")) + sorted(p.glob("*.flac"))
            if not audio_files:
                # Try loading as HF dataset with audio column
                logger.info("No WAV/FLAC in %s, trying as HF dataset...", p)
                try:
                    from datasets import load_from_disk

                    ds = load_from_disk(str(p))
                    for i, sample in enumerate(ds):
                        if i >= 3:
                            break
                        audio_data = sample.get("audio", {})
                        arr = torch.tensor(audio_data["array"], dtype=torch.float32)
                        sr = audio_data["sampling_rate"]
                        refs.append((arr, sr))
                    continue
                except Exception as e:
                    logger.warning("Could not load %s as dataset: %s", p, e)
                    continue
            for af in audio_files[:3]:
                audio, sr = torchaudio.load(str(af))
                refs.append((audio.mean(dim=0), sr))
        elif p.is_file():
            audio, sr = torchaudio.load(str(p))
            refs.append((audio.mean(dim=0), sr))
        else:
            logger.warning("Reference audio not found: %s", p)
    return refs


def main():
    parser = argparse.ArgumentParser(
        description="Generate speech tokens from Polish text via CosyVoice2"
    )
    parser.add_argument(
        "--reference-audio",
        nargs="+",
        default=[DEFAULT_REFERENCE],
        help="Reference audio file(s) or directory for voice cloning. "
        "Multiple files enable round-robin speaker diversity.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/speech_tokens",
        help="Output directory (default: outputs/speech_tokens)",
    )
    parser.add_argument(
        "--render-audio",
        action="store_true",
        help="Also save rendered WAV files for listening",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (default: cuda if available)",
    )
    parser.add_argument(
        "--checkpoint",
        default="iic/CosyVoice2-0.5B",
        help="CosyVoice2 model checkpoint (ModelScope ID or local path)",
    )
    parser.add_argument(
        "--mode",
        default="cross_lingual",
        choices=["cross_lingual", "zero_shot"],
        help="Inference mode (default: cross_lingual for Polish)",
    )
    parser.add_argument(
        "--sentences-file",
        help="JSON file with sentences (list of {id, text} dicts). "
        "If not provided, uses 10 built-in Polish sentences.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load sentences ────────────────────────────────────────────────
    if args.sentences_file:
        with open(args.sentences_file) as f:
            sentences = json.load(f)
        logger.info("Loaded %d sentences from %s", len(sentences), args.sentences_file)
    else:
        sentences = DEFAULT_SENTENCES
        logger.info("Using %d built-in Polish sentences", len(sentences))

    # ── Normalize text ────────────────────────────────────────────────
    normalizer = PolishTextNormalizer()
    for s in sentences:
        s["normalized_text"] = normalizer.normalize(s["text"])
        logger.info("  [%s] %s", s["id"], s["text"])
        logger.info("       -> %s", s["normalized_text"])

    # ── Load reference audio ──────────────────────────────────────────
    logger.info("Loading reference audio from: %s", args.reference_audio)
    refs = load_reference_audios(args.reference_audio)
    if not refs:
        logger.error(
            "No reference audio loaded. Provide --reference-audio with valid "
            "WAV/FLAC files or a FLEURS cache directory."
        )
        sys.exit(1)
    logger.info("Loaded %d reference audio clip(s) for speaker diversity", len(refs))

    # ── Load TTS backend ──────────────────────────────────────────────
    logger.info("Loading CosyVoice2 backend (checkpoint=%s)...", args.checkpoint)
    backend = CosyVoice2TTSBackend(
        checkpoint=args.checkpoint,
        device=args.device,
        mode=args.mode,
    )
    backend.load_model()
    logger.info("Backend ready on %s", args.device)

    # ── Generate ──────────────────────────────────────────────────────
    results = []
    total_tokens = 0
    total_duration = 0.0
    t0 = time.time()

    for i, sample in enumerate(sentences):
        ref_audio, ref_sr = refs[i % len(refs)]
        sid = sample["id"]
        logger.info(
            "[%d/%d] Generating tokens for %s (speaker %d)...",
            i + 1,
            len(sentences),
            sid,
            i % len(refs),
        )

        try:
            output = backend.generate(
                text=sample["normalized_text"],
                reference_audio=ref_audio,
                reference_audio_sr=ref_sr,
                render_audio=args.render_audio,
            )

            metadata = save_sample(
                sample_id=sid,
                output=output,
                source_text=sample["text"],
                normalized_text=sample["normalized_text"],
                output_dir=output_dir,
                save_audio=args.render_audio,
            )

            results.append(metadata)
            total_tokens += output.num_tokens
            total_duration += output.duration_seconds
            logger.info(
                "  -> %d tokens (%.1fs audio), saved to %s/",
                output.num_tokens,
                output.duration_seconds,
                output_dir,
            )

        except Exception as e:
            logger.error("  FAILED for %s: %s", sid, e, exc_info=True)
            results.append({"id": sid, "error": str(e)})

    elapsed = time.time() - t0

    # ── Summary ───────────────────────────────────────────────────────
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    summary = {
        "total_samples": len(sentences),
        "successful": len(successful),
        "failed": len(failed),
        "total_tokens": total_tokens,
        "total_duration_seconds": round(total_duration, 2),
        "elapsed_seconds": round(elapsed, 2),
        "backend": "cosyvoice2",
        "checkpoint": args.checkpoint,
        "mode": args.mode,
        "device": args.device,
        "output_dir": str(output_dir),
    }

    # Save summary
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Generation complete: {len(successful)}/{len(sentences)} succeeded")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Total audio duration: {total_duration:.1f}s")
    print(f"  Wall time: {elapsed:.1f}s")
    print(f"  Output: {output_dir}/")
    if failed:
        print(f"  Failed: {[r['id'] for r in failed]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
