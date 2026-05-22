# Multi-speaker synthesis methodology — PL & ZH (2026-05-22)

## Motivation

Initial synthesis (K=1, single reference voice per language) was flagged as
sub-optimal by supervisor. Literature review confirmed:

- **Chu et al. 2024 (CosyVoice ablation)**: text diversity > speaker diversity;
  saturation around 50–100 speakers.
- **Ogun, Colotte, Vincent 2025**: monotonic improvement with speaker count up
  to a plateau in the low thousands; K=2,457 → 8K shows diminishing returns
  past ~4K but a clear benefit over K=1.
- **VC-based augmentation can fail entirely** if added speakers lack phonetic
  diversity — quality of the reference pool matters more than raw count.
- **Quality metrics don't predict downstream ASR usefulness** (NISQA MOS,
  synthetic WER are weak proxies). Don't filter the reference pool by
  synthesis WER.

Decision: rebuild PL + ZH synthesis with **K=50 reference speakers** per
language, using a defensible selection methodology rather than naive random
sampling.

## Selection methodology

Per language, applied to the corpus selected as ref source:

1. **Quality filter** — drop speakers with < 10 min total speech (insufficient
   to estimate cadence; also dominated by low-engagement contributors).
2. **Cadence prefilter** — keep only speakers in the slowest X percentile by
   chars-per-second (cps). Lower cps → longer synthesized output per source
   text → preserves (or expands) total dataset hours after synthesis.
   **Critical for PL**: K=1 baseline already at ×0.85 ratio (15% hour loss);
   random K=50 would worsen this; cadence-aware selection reverses it.
3. **Gender stratify** within the slow pool: target K/2 male + K/2 female,
   random sample with fixed `seed=42`. Anchor (the K=1 baseline ref voice)
   force-included for direct comparability.
4. **Reference clip selection** per chosen speaker: pick the clip whose
   duration is closest to the speaker's median (avoids outlier-length clips
   that ramble or got mid-sentence cuts).
5. **Round-robin assignment** to manifest items via `hash(item.id) % K` —
   deterministic, reproducible, every speaker gets ~equal coverage.

## Source corpus per language

### ZH: AISHELL-3 (full train Shar)
- 218 speakers nominal, 174 in extracted manifest, 142 pass quality filter
- Studio recordings, gender labels in `spk-info.txt`, professional
- **Selected: cps70 percentile** (relaxed cutoff; ZH hour-loss not a concern)
- Why: ZH K=1 baseline already at ×1.09 (gain 9%); any K=50 from slow pool
  retains the expansion. Widening the percentile fixes the gender skew
  that strict pools showed.

### PL: Common Voice pl validated
- 3,314 unique speakers (massive pool vs MLS pl's 11)
- Crowd-recorded amateur, gender often unspecified in metadata
- **Selected: cps30 percentile** (stricter cutoff; PL hour-loss is real)
- Why: PL K=1 baseline at ×0.85 (lose 15%); strict cadence filter inverts
  this to ×1.10 (gain 10%) — net +29% hours vs K=1.
- Quality filter handles CV's amateur-recording variance by dropping
  low-volume contributors (143/3314 survive 10-min floor).

### Why CV pl instead of MLS pl
MLS pl Shar has only 11 unique speakers (LibriVox single-reader-per-book).
Insufficient for K=50. CV pl is licensed for synth→STT use cases (Mozilla's
stated intent for the corpus, per their policy); not used for deployable
voice-cloning products.

## Concrete pool stats

| Pool | K | mean cps | M/F balance | Predicted ratio | vs K=1 baseline |
|---|---|---|---|---|---|
| ZH (AISHELL-3, cps70) | 50 | 3.125 | 17 / 33 | ×1.16 | +7% hours |
| PL (CV pl, cps30) | 43 | 9.484 | 27 / 5 (+ 11 ?) | ×1.10 | +29% hours |

**Notes:**
- PL K=43 because slow pool only had 43 speakers — backfilled all of them.
- PL gender skew (5F) reflects CV pl reality: female contributors are less
  represented overall, AND slow-speaking female contributors are very rare
  in this pool. Worth flagging as a methodological caveat in the report.
- ZH gender skew (33F vs 17M) is the inverse: AISHELL-3 slow speakers are
  predominantly female (likely a real cadence-vs-gender correlation in
  Mandarin).

## Implementation artefacts

- `scripts/build_zh_ref_pool.py` — ZH pool builder (manifest + spk-info → JSON)
- `scripts/build_pl_ref_pool.py` — PL pool builder (CV TSV → JSON)
- `scripts/extract_pl_refs.py` — extract MP3s from CV .tar.zst → 16 kHz WAV
- `scripts/augment_manifest_with_refs.py` — hash-based round-robin assignment
- `scripts/synthesize_to_shar.py` — patched to support per-item refs
  (pre-caches all distinct refs at start; sorts items by ref before batching;
  splits batches at ref boundaries so batched RTF stays intact)

## Ethics statement

Reference voice selection considered consent expectations per source:
- **AISHELL-3** (ZH ref pool): purpose-built TTS dataset; no concerns.
- **CV pl** (PL ref pool): used for synth→STT training only, in line with
  Mozilla's stated intent. Not used to build deployable voice-cloning models.
- **Anchor refs** (MLS spk1636 / AISHELL-3 SSB0668): LibriVox / standard TTS
  releases; consent unambiguous.
- **Excluded as ref sources**: YODAS, YODAS2 (YouTube content; creators did
  not consent to AI training); VoxPopuli (where labeled speaker metadata
  was missing — would require additional pipeline work).

## Open items

1. PL gender imbalance (5F): could widen to cps50 percentile (~71 speakers in
   pool) to recover more female speakers, at cost of mean_cps ~10.3 → ratio
   ~×1.01. Hour preservation still better than K=1 baseline.
2. Smoke test of the synth patch was queued at write time — once it
   completes cleanly we fan out the full PL + ZH redo.
3. VoxPopuli pl extension (~700 MEPs) is deferred future work. Would require
   downloading VoxPopuli's labeled metadata from HF (~36 MB TSV) and an
   ffmpeg-based clip extractor from raw .ogg sessions.
