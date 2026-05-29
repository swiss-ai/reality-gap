# Scripts overview

What each script does and where it sits in the pipeline. Authored by `mkwapniewska` unless noted (supervisor `xyixuan` / `Alvorecer721`, contributor `melanierieff`).

The pipeline runs in 5 phases:

```
[1] TTS backend selection  →  [2] Ref-pool building  →  [3] Synthesis (K=50/K=1)
                                                              ↓
                              [5] Eval (this branch's contribution)
                                                              ↑
                              [4] Convert → tokenize  (chained per set)
```

---

## [1] TTS backend selection (one-time, frozen)

How VoxCPM2 won over Piper / CosyVoice2 / XTTS / MMS-TTS / IndexTTS / Qwen2.5-Omni.

| Script | Role |
|---|---|
| `benchmark_tts.py` | Generate test set → score WER/CER via Whisper → aggregate per-backend tables. Test sets in `data/tts_bench/`. |
| `generate_speech_tokens.py` | Polish PoC: text → CosyVoice2 → speech tokens. Early-pipeline scaffolding for the Polish PoC. |
| `generate_samples.py` | Save original + reconstructed WAV pairs from a tokenizer round-trip (listening-test artifact). |
| `normalize.py` | Standalone CLI wrapper around `src/speech_generation/PolishTextNormalizer` (abbreviations, numbers, dates, currency → spoken Polish). Used by `benchmark_tts.py` and `generate_speech_tokens.py`. |

## [2] Ref-pool building (K=50 methodology)

How the 50 speakers per language were chosen.

| Script | Role |
|---|---|
| `build_pl_ref_pool.py` | Read CV pl validated set → quality filter (≥10 min/spk) → cps30 percentile (slowest 30%) → gender stratify → 43 effective PL speakers + per-speaker median-duration ref clip. |
| `build_zh_ref_pool.py` | Same idea for AISHELL-3 train: ≥10 min/spk → cps70 percentile → 33 F / 17 M stratified = 50 speakers. |
| `augment_manifest_with_refs.py` | Take a synth manifest + a ref-pool JSON → write per-utterance ref assignment via `hash(item.id) % K`. Deterministic with seed=42. |
| `slurm/extract_pl_refs.slurm` | SLURM driver wrapping the PL ref-pool build (single GPU node, runs the analysis end-to-end). |
| `slurm/voxpopuli_vad_scan.slurm` | Voice-activity detection for VoxPopuli PL (long parliamentary audio needs splitting before synth). |

## [3] Synthesis → Lhotse Shar

Bulk text → speech via VoxCPM2 + nano-vllm-voxcpm.

| Script | Role |
|---|---|
| `synthesize_to_shar.py` | The synth worker: reads a manifest shard, runs nano-vllm-voxcpm with the assigned ref voice per utt, writes Lhotse Shar (cuts.NNNNNN.jsonl.gz + recording.NNNNNN.tar). Per-task; sbatch-array fans it out. |
| `slurm/pl_k50_orchestrator.slurm` | PL K=50 orchestrator: submits one synth array per dataset, all 18+ sets in flight. |
| `slurm/pl_k50_orchestrator_extra.slurm` | Same for any post-fact additions (FineWeb-2 batches, etc.). |
| `slurm/zh_fw2_k50_orchestrator.slurm` | ZH FineWeb-2 K=50 orchestrator. |
| `auto_refire_failed.sh` | After a SLURM array completes, re-submit only the shards that failed/timed-out (idempotent — checks for existing parquets). |
| `slurm/rsync_k50_to_delivery.slurm` | Bulk rsync of completed Shar dirs to the delivery store path. |

## [4] Convert (Shar → parquet → Megatron .bin/.idx)

| Script | Role |
|---|---|
| `shar_to_parquet.py` | Lhotse Shar → HuggingFace-style parquet (`id`, `text`, `duration`, `audio: struct<bytes, sampling_rate>`, `language`). Handles both `shard_NNNN/` (synth) and `worker_NN/` (MLS) layouts. **Required for real-audio re-tokenize of MLS PL.** |
| `extract_yodas2_to_parquet.py` | Custom extractor for YODAS2 raw tar.zst → per-segment parquet matching delivery schema. Slices video-level WAVs by start_cs/end_cs in segment_id. **Required for real-audio re-tokenize of YODAS2 PL + ZH.** |
| `extract_emilia_to_parquet.py` | Custom extractor for Emilia WDS tars (flat `<key>.mp3` + `<key>.json` pairs) → parquet. **Required for real-audio re-tokenize of Emilia ZH.** |
| `slurm/mkwapniewska_pl.slurm` | Generic launcher that runs Hydra `audio_tokenization run dataset=mkwapniewska/<subset>` with `stage=convert` or `stage=tokenize`. Language-agnostic despite the name. |
| `slurm/mkwapniewska_pl_chain.sh` | Submits `convert` then `tokenize` as an `afterok` dependent chain for one subset. Convert: 1 task × 288 cpus. Tokenize: 4 tasks × 4 GPUs. |
| `merge_indexed_datasets.py` | Merge per-rank Megatron `.bin/.idx` outputs into one (rarely needed — supervisor's pipeline already shards per rank). |
| `merge_all_s1.sh` *(supervisor)* | Stage-1 merge wrapper. |

The per-set Hydra configs live in `audio_tokenization/configs/pipeline/dataset/mkwapniewska/*.yaml`:
- `*_K50.yaml` — K=50 multi-speaker synth (production)
- `*_spk1636.yaml` (PL) / `*_spk0668.yaml` (ZH) — K=1 single-speaker baseline (kept for the ablation)
- `*_real.yaml` — Real-audio re-tokenize (CV25 PL, MLS PL, YODAS2 PL+ZH, Emilia ZH) — language-pure tokenized counterparts for the reality-gap inputs.

## [5] Eval (this branch — reality-gap measurement)

Run when supervisor publishes the two trained checkpoints. Both checkpoints get the same 4 PL/ZH test sets; reality_gap = WER(synth-trained) − WER(real-trained) per set.

| Script | Role |
|---|---|
| `audio_inference.py` *(from upstream/batch_tok)* | Encode test audio → WavTokenizer40 → build prompt `[bos, audio_start, codes+262344, audio_end, stt_transcribe]` → vLLM generate → decode → write `InferenceRun` JSON. Three input modes: `--audio-dir` (HF Arrow), `--parquet-dir` (parquet), `--wav-dir` (raw wav + metadata.tsv). `--manifest` runs many sets on one model load. |
| `prep_eval_test_sets.sh` | One-shot staging: symlinks CV25 PL + ZH-CN test.parquet into clean per-set dirs so the loader's `*.parquet` glob doesn't pick up train/dev/etc. |
| `slurm/eval_reality_gap.slurm` | SLURM driver: takes `<model_path> <output_subdir>`, runs the whole manifest (`configs/eval/reality_gap.json`) on vLLM. ~30-60 min wall per checkpoint. |
| `score_inference.py` | WER (Latin) / CER (CJK) scorer over inference JSON. Three modes: `--run` (single), `--root` (batch CSV), `--synth-trained X --real-trained Y` (gap mode — prints reality-gap directly). Corpus-level + mean-of-rates both emitted. |
| `run_all_inference.sh`, `run_all_inference_v2.sh` *(from upstream/batch_tok)* | Upstream's eval drivers (kept for reference — single-dataset CLI mode, used by supervisor). Our `eval_reality_gap.slurm` does the same with `--manifest`. |

## Analysis / report figures

| Script | Role |
|---|---|
| `plot_f0_diversity.py` | F0 distribution width K=50 vs K=1 (matched-id pairs). Publication-ready PDF output (no title, 13pt fonts, vector). ZH numbers: K=50 σ=61.6 Hz vs K=1 σ=25.3 Hz. PL numbers anomalous — see report caveat. |
| `analyze_tokenizers.py` | Aggregate per-tokenizer per-language metrics → plots → CSV. Reads `metrics/` outputs. |
| `tokenizer_evaluation.py` | Main tokenizer eval loop: encode→decode→compute MSE/SNR/SDR/PESQ/STOI/ESTOI + tokens/sec + compression ratio. Writes JSON to `metrics/`. |
| `generate_tables.py` | Performance comparison tables (LaTeX-ready). |
| `plot_xcodec_vs_wavtokenizer.py` *(supervisor)* | XCodec vs WavTokenizer comparison plots. |

## Dataset downloaders

| Script | Role |
|---|---|
| `download_emilia_zh.py` | HF Emilia ZH config (WDS tars). |
| `download_eurospeech.py` | HF Eurospeech 22-lang cache. |
| `download_fleurs.py` / `download_fleurs.sh` | HF FLEURS 40-lang cache. |
| `download_gtzan.py` | HF GTZAN music dataset. |
| `download_naturelm.py` | HF NatureLM environmental audio. |

## Submission / log helpers

| Script | Role |
|---|---|
| `submit_missing_jobs.py` | Find missing tokenizer × language combinations and fan out SLURM jobs. Supports `--dry-run`, `--group-by`, `--validate-metrics`. |
| `delete_successful_logs.py` | Clean up `logs/` after a successful run (keeps FAILED logs untouched). |
| `auto_refire_failed.sh` | (see Phase 3) re-fires only the shards that failed. |

## Supervisor / contributor scripts (kept for reference)

| Script | Author | Role |
|---|---|---|
| `audio_continuation_demo.py` | supervisor | Audio-continuation demo from batch_tok. |
| `check_silence_tokens.py` | supervisor | Silence-token sanity probe during tokenization. |
| `profile_batch_size.py` | supervisor | Batch-size profiling for the synth pipeline. |
| `test_audioset_pad_trim.py`, `test_padding_effects.py` | supervisor | Audio padding behaviour tests (informed the `trim_last_tokens` config decision). |
| `test_minimal.py`, `verify_datasets.py` | contributor (Melanie) | Minimal smoke + dataset verification. |
| `check_voxpopuli.py` | contributor (Melanie) | VoxPopuli probe. |
| `build_hf_dataset.py` | supervisor | HF dataset export helper. |
| `merge_all_s1.sh` | supervisor | Stage-1 merge wrapper. |
| `plot_xcodec_vs_wavtokenizer.py` | contributor (Melanie) | Tokenizer comparison plots. |
