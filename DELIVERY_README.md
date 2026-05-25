# Synthetic Polish + Chinese Speech Datasets — voxcpm2

Delivery from M. Kwapniewska, 2026-05-24.

Synthesis was done with **VoxCPM2** (Apache 2.0, OpenBMB) using
**nano-vllm-voxcpm** for throughput. Final parquets live under
`/capstor/store/cscs/swissai/infra01/audio-datasets/synthesized/voxcpm2/`.

Per-set tokenized outputs (.bin/.idx, Megatron format) live under
`/capstor/store/cscs/swissai/infra01/audio-datasets/reality_gap/tokenized/voxcpm2/`.

**Real-audio source mapping** (for ablation training on the same utterances): see `_real_audio_manifests/MANIFEST.json` inside the delivery dir. For each synth set it lists the original real-audio source path on cluster + per-set `<set>_ids.txt` (one synth id per line). See [Real-audio source mapping](#real-audio-source-mapping) below.

---

## Synthesis modes

Each set is delivered in **one or both** of two synthesis modes:

| Suffix | Mode | Reference voice                                                                                                                              |
|---|---|----------------------------------------------------------------------------------------------------------------------------------------------|
| `_spk1636` (PL) / `_spk0668` (ZH) | **K=1** single reference | PL: MLS speaker `1636` (slow audiobook narrator). ZH: AISHELL-3 speaker `SSB0668` (slow narrator, ~4 cps, clear tonal pronunciation).        |
| `_K50` | **K=50** multi-speaker pool | 50-speaker reference pool, seed=42, balanced gender × cadence × quality. Each utterance synth'd against a different ref drawn from the pool. |

**Selection rule applied to this delivery:**

- **Audio-paired sources** (MLS, VoxPopuli, AISHELL-1/3/4, WenetSpeech, Emilia, YODAS2, Common Voice, Granary YTC): both K=1 and K=50 delivered where both were synthesized — they share the same source text but differ in speaker conditioning, useful as a single-speaker vs multi-speaker ablation handle.
- **Text-only sources** (FineWeb-2): K=50 only. No real-audio counterpart means there's no single "canonical speaker" to anchor to. Not yet on the shared cluster, final synthesis still being run.

---

## K=1 ↔ K=50 set mapping

Quick reference for which K=1 set corresponds to which K=50 set per source corpus.

### Polish

| Source corpus | K=1 set | K=50 set |
|---|---|---|
| Common Voice 25 pl | `pl_cv25_spk1636` | `pl_cv25_K50` |
| MLS pl | `pl_mls_spk1636` | `pl_mls_v2_K50` |
| VoxPopuli pl ASR — 1kh slice | `pl_voxpopuli_1kh_spk1636` *(Granary text)* | `pl_voxpopuli_full_K50` *(raw VoxPopuli text)* ↓ |
| VoxPopuli pl ASR — rest | `pl_voxpopuli_rest_spk1636` *(Granary text)* | `pl_voxpopuli_full_K50` ↑ |
| YODAS2 pl | `pl_yodas2_spk1636` | `pl_yodas2_K50` |
| Granary YODAS pl | `pl_granary_yodas_spk1636` | — (none, ~7 h slice, K=1 only) |
| Granary YTC pl | `pl_granary_ytc_spk1636` | `pl_granary_ytc_K50` |
| FineWeb-2 pl (text-only) | — | `pl_fineweb2_10kh_K50` *(synth in flight)* |

**Note on VoxPopuli pl:** the K=1 deliveries (`pl_voxpopuli_1kh_spk1636` + `pl_voxpopuli_rest_spk1636`) use **Granary's curated text normalization** of the VoxPopuli pl ASR audio. The K=50 delivery (`pl_voxpopuli_full_K50`) uses **raw VoxPopuli ASR text**. Same underlying audio source corpus, two different text-cleanup variants — pair these as an additional text-normalization ablation handle if useful. (Naming note: these two sets were originally `pl_granary_1kh_spk1636` / `pl_granary_rest_spk1636`; renamed for consistency with the K=50 sibling.)

### Chinese

| Source corpus | K=1 set | K=50 set |
|---|---|---|
| AISHELL-1 | `zh_aishell1_spk0668` | `zh_aishell1_K50` |
| AISHELL-3 | `zh_aishell3_spk0668` | `zh_aishell3_K50` |
| AISHELL-4 (long) | `zh_aishell4_L_spk0668` | `zh_aishell4_L_K50` |
| AISHELL-4 (medium) | `zh_aishell4_M_spk0668` | `zh_aishell4_M_K50` |
| AISHELL-4 (short) | `zh_aishell4_S_spk0668` | `zh_aishell4_S_K50` |
| Common Voice 25 zh-CN | `zh_cv25_spk0668` | `zh_cv25_K50` |
| Emilia-YODAS ZH (full extract) | `zh_emilia_full_spk0668` | `zh_emilia_full_K50` |
| Emilia-YODAS ZH (cluster subset, 41 h) | `zh_emilia_spk0668` | — (subset, K=1 only) |
| WenetSpeech — 1kh slice | `zh_wenetspeech_1kh_spk0668` | `zh_wenetspeech_full_K50` ↓ |
| WenetSpeech — rest (>1kh) | `zh_wenetspeech_rest_spk0668` | `zh_wenetspeech_full_K50` ↑ |
| YODAS2 zh | `zh_yodas2_spk0668` | `zh_yodas2_K50` |
| FineWeb-2 zh (text-only) | — | `zh_fineweb2_10kh_K50` *(synth in flight)* |

---

## Polish inventory

Synth-hour totals computed from `duration` column over delivered parquets, 2026-05-24.

| Set | Source dataset | License | K-mode | Synth h | Rows | Status |
|---|---|---|---|---|---|---|
| `pl_cv25_K50` | Common Voice 25 pl | CC0-1.0 | K=50 | 142.0 | 121,857 | ✓ |
| `pl_cv25_spk1636` | Common Voice 25 pl | CC0-1.0 | K=1 | 111.1 | 91,393 | ✓ |
| `pl_fineweb2_10kh_K50` | FineWeb-2 pl (text-only) | ODC-By | K=50 | _pending_ | _pending_ | ⏳ synth in flight |
| `pl_granary_yodas_spk1636` | Granary YODAS pl | CC-BY 3.0 | K=1 | 6.6 | 3,239 | ✓ |
| `pl_granary_ytc_K50` | Granary YTC pl | CC-BY 3.0 | K=50 | 10.2 | 2,922 | ✓ |
| `pl_granary_ytc_spk1636` | Granary YTC pl | CC-BY 3.0 | K=1 | 8.9 | 2,647 | ✓ |
| `pl_mls_spk1636` | MLS pl | CC-BY 4.0 | K=1 | 90.6 | 21,913 | ✓ |
| `pl_mls_v2_K50` | MLS pl (v2 manifest) | CC-BY 4.0 | K=50 | 109.5 | 25,043 | ✓ |
| `pl_voxpopuli_1kh_spk1636` | VoxPopuli pl ASR (1kh, Granary text) | CC-BY 3.0 | K=1 | 850.3 | 164,573 | ✓ |
| `pl_voxpopuli_rest_spk1636` | VoxPopuli pl ASR (rest, Granary text) | CC-BY 3.0 | K=1 | 11,932.5 | 2,307,985 | ✓ |
| `pl_voxpopuli_full_K50` | VoxPopuli pl ASR (full, raw text) | CC0-1.0 | K=50 | 11,355.1 | 2,042,340 | ✓ |
| `pl_yodas2_K50` | YODAS2 pl | CC-BY 3.0 | K=50 | 650.9 | 490,986 | ✓ |
| `pl_yodas2_spk1636` | YODAS2 pl | CC-BY 3.0 | K=1 | 613.2 | 490,387 | ✓ |

**Polish totals (excluding pending FineWeb-2 text-only):**
- **K=50 audio: 12,267.7 h** (5 sets)
- **K=1 audio: 13,613.2 h** (7 sets)
- Once `pl_fineweb2_10kh_K50` synth completes, total Polish ≈ **36 kh**.

**Polish source notes:**

- All audio-paired sources have real-audio counterparts (except FineWeb-2 which is text-only).
- ~1% English contamination caught in Granary VoxPopuli pl (MEPs speaking English on the Polish track) — dropped via Polish-character predicate (`ąćęłńóśźż`).
- Compression ratio: VoxCPM2 + slow MLS-1636 reference produces synth audio ~85% of source duration on Polish (close to 1× on read-speech sources like MLS).
- K=50 vs K=1 on the same source: ratio ≈ 1.08× (K=50 pool includes faster + slower speakers around the mean).

---

## Chinese inventory

| Set | Source dataset | License | K-mode | Synth h | Rows | Status |
|---|---|---|---|---|---|---|
| `zh_aishell1_K50` | AISHELL-1 | Apache 2.0 | K=50 | 133.6 | 120,098 | ✓ |
| `zh_aishell1_spk0668` | AISHELL-1 | Apache 2.0 | K=1 | 124.6 | 119,887 | ✓ |
| `zh_aishell3_K50` | AISHELL-3 | Apache 2.0 | K=50 | 58.3 | 63,262 | ✓ |
| `zh_aishell3_spk0668` | AISHELL-3 | Apache 2.0 | K=1 | 55.0 | 63,262 | ✓ |
| `zh_aishell4_L_K50` | AISHELL-4 (long) | Apache 2.0 | K=50 | 16.8 | 10,658 | ✓ |
| `zh_aishell4_L_spk0668` | AISHELL-4 (long) | Apache 2.0 | K=1 | 15.3 | 8,614 | ✓ |
| `zh_aishell4_M_K50` | AISHELL-4 (medium) | Apache 2.0 | K=50 | 59.9 | 42,328 | ✓ |
| `zh_aishell4_M_spk0668` | AISHELL-4 (medium) | Apache 2.0 | K=1 | 53.9 | 32,474 | ✓ |
| `zh_aishell4_S_K50` | AISHELL-4 (short) | Apache 2.0 | K=50 | 31.3 | 22,003 | ✓ |
| `zh_aishell4_S_spk0668` | AISHELL-4 (short) | Apache 2.0 | K=1 | 28.4 | 17,024 | ✓ |
| `zh_cv25_K50` | Common Voice 25 zh-CN | CC0-1.0 | K=50 | 199.9 | 167,017 | ✓ |
| `zh_cv25_spk0668` | Common Voice 25 zh-CN | CC0-1.0 | K=1 | 188.4 | 167,017 | ✓ |
| `zh_emilia_full_K50` | Emilia-YODAS ZH (full extract) | CC-BY 4.0 | K=50 | 318.2 | 110,156 | ✓ |
| `zh_emilia_full_spk0668` | Emilia-YODAS ZH (full extract) | CC-BY 4.0 | K=1 | 310.8 | 110,156 | ✓ |
| `zh_emilia_spk0668` | Emilia-YODAS ZH (cluster subset) | CC-BY 4.0 | K=1 | 45.6 | 19,064 | ✓ |
| `zh_fineweb2_10kh_K50` | FineWeb-2 zh (text-only) | ODC-By | K=50 | _pending_ | _pending_ | ⏳ synth in flight |
| `zh_wenetspeech_1kh_spk0668` | WenetSpeech 1kh slice | Apache 2.0 (CC-BY-derived) | K=1 | 1,082.6 | 1,014,667 | ✓ |
| `zh_wenetspeech_full_K50` | WenetSpeech full | Apache 2.0 (CC-BY-derived) | K=50 | 11,685.6 | 13,306,267 | ✓ |
| `zh_wenetspeech_rest_spk0668` | WenetSpeech rest (>1kh) | Apache 2.0 (CC-BY-derived) | K=1 | 6,760.7 | 6,341,128 | ✓ |
| `zh_yodas2_K50` | YODAS2 zh | CC-BY 3.0 | K=50 | 271.8 | 250,239 | ✓ |
| `zh_yodas2_spk0668` | YODAS2 zh | CC-BY 3.0 | K=1 | 256.9 | 250,239 | ✓ |

**Chinese totals (excluding pending FineWeb-2 text-only):**
- **K=50 audio: 12,775.4 h** (9 sets)
- **K=1 audio: 8,922.2 h** (11 sets — note K=1 wenetspeech is 1kh + rest = 7,843 h vs K=50 full = 11,686 h; different filtering / scoping between manifests)
- Once `zh_fineweb2_10kh_K50` completes, total Chinese ≈ **32 kh**.

**Chinese source notes:**

- Compression ratio varies by source: WenetSpeech ~0.94×, AISHELL-1 ~0.83-0.89×, Emilia ~1.11×.
- AISHELL-4 size labels (L/M/S) refer to **segment-length partitioning** of the same meeting corpus — `M` is the largest by hours (60 h) because mid-length segments dominate the corpus. Not "Large > Medium > Small" by hours.
- AISHELL-3 IS included as a synth source despite providing the K=1 reference voice — K=50 draws from a different multi-speaker pool, and the spk0668 audio of AISHELL-3 utterances is distinct from the originals.
- WenetSpeech is split into `1kh` (first 1k h) + `rest` (remaining ~6.8k h) for K=1, unified as `full` (11.7k h) for K=50 — same underlying source, different shard groupings.

---

## Parquet schema

```
id              : string                                      (source utterance id)
text            : string                                      (raw Polish/Chinese text)
duration        : float64                                     (seconds)
audio           : struct<bytes: binary, sampling_rate: int64> (PCM 16-bit WAV @ 24 kHz)
language        : string                                      ("pl" or "zh")
```

Matches `audio_tokenization/prepare/prepare_parquet_to_shar.py` expectations on the `mkwapniewska/reality_gap` branch. Layout under each set:

```
voxcpm2/<set_name>/data/train-<shard>-<cuts>.parquet
```

Notes:
- `audio.bytes` are 24 kHz native (no resampling — matches VoxCPM2 output).
- No `text_tokens` column — supervisor's pipeline tokenizes from `text` downstream.
- Each row is one independent clip (5-30s); supervisor's interleaver weaves rows at chunk boundaries.

---

## Real-audio source mapping

For each synth set, the **real-audio source corpus** on cluster is documented in `_real_audio_manifests/MANIFEST.json` alongside the synth parquets. This lets you train a comparison ASR/codec model on the same utterances using their original recordings.

**Layout under `_real_audio_manifests/`:**

```
_real_audio_manifests/
├── MANIFEST.json                — top-level map: set → {source_path, source_format, license, n_ids, ids_file}
├── pl_cv25_K50_ids.txt          — one synth id per line (the utterances synthesized for this set)
├── pl_voxpopuli_full_K50_ids.txt
├── pl_mls_v2_K50_ids.txt
├── …                            — one *_ids.txt per delivered synth set
```

**`MANIFEST.json` entry shape:**

```json
{
  "pl_voxpopuli_full_K50": {
    "source_path": "/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/voxpopuli_asr/pl",
    "source_format": "lhotse_shar",
    "license": "CC0-1.0",
    "notes": "VoxPopuli pl ASR raw. synth id == recording_id.",
    "n_ids": 2042340,
    "ids_file": "_real_audio_manifests/pl_voxpopuli_full_K50_ids.txt"
  },
  …
}
```

**`source_format` values + how to load:**

| Format | Storage | How to look up an id |
|---|---|---|
| `lhotse_shar` | `cuts.NNNNNN.jsonl.gz` + `recording.NNNNNN.tar` | Standard Lhotse `SharReader`; synth id matches Lhotse `cut.id` (or its prefix — see VoxPopuli note below). |
| `common_voice` | Parquet (`processed/commonvoice25/<lang>/`) + tsv | Match synth id (e.g. `common_voice_pl_20867142`) against the `path`/clip-id column. |
| `hf_raw` | HF dataset cache (tars + jsonl manifests) | Synth id matches the HF row's `id` field (Emilia: `ZH_<vid>_W<seq>`; YODAS2: `<videoID>-<shardN>-<startms>-<endms>`). |
| `granary_raw` | Granary's curated parquet/json layout under `raw/granary/nvidia-granary/<lang>/` | Synth id matches Granary's per-utterance id (parseable: `pl000_<row>_<videoID>_<…>` for YODAS subset, `<date>-<time>-<committee>_pl_<seg>` for VoxPopuli subset). |
| `text_only` | n/a | FineWeb-2 — no real-audio counterpart; `source_path` is `null`. |

---

## Language filtering

Source manifests run through a per-language character predicate before synthesis:

- **Polish:** drop items with no Polish-specific characters (`ąćęłńóśźż` + uppercase).
- **Chinese:** drop items with no CJK Unified Ideographs (U+4E00-U+9FFF).
- **min-chars:** pl=20, zh=5 (drops near-empty rows).

Edge cases (short Polish phrases without diacritics like "Tak", "Nie") get dropped — negligible at scale.

