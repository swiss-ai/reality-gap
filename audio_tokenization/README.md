# Audio Tokenization

> Part of the `benchmark-audio-tokenizer` repository

This module provides a pipeline for tokenizing audio datasets for language model training. It **maximally reuses** code from `vision_tokenization` (located at `../src/repos/benchmark-image-tokenzier/vision_tokenization/`) and adapts it for audio modality.

## Architecture: Maximizing Code Reuse

**Strategy:** Import and reuse as much code as possible from `vision_tokenization`, adapt minimally where needed, and only write new code for audio-specific functionality.

### Direct Imports (No Changes Required)
- `pipelines/indexed_dataset_megatron.py` - Megatron dataset writer (`.bin/.idx` format) - imported from `vision_tokenization`
- `vokenizers/base.py` - `BaseTokenizer` abstract class - imported from `vision_tokenization`

### Minimal Adaptations
- `pipelines/base.py` - Adapted `BasePipeline`, `BaseTokenizerWorker`, `ProgressActor` for audio (removed image-specific logic)
- `pipelines/hf/pipeline.py` - Changed `image_field` → `audio_field`, removed image-specific filtering and pixel parameters
- `pipelines/hf/workers.py` - Changed image extraction → audio extraction, removed resolution filtering
- `utils/omni_tokenizer/create_audio_base.py` - Adapted from `create_base.py` for audio tokens

### Audio-Specific (New Code)
- `vokenizers/audio/audio_only.py` - `AudioOnlyTokenizer` class (analog to `EMUImageOnlyTokenizer`)
- `tokenize.py` - Main entry point (adapted from `vision_tokenization/tokenize.py`)

> **Note**: This module uses tokenizer implementations from `../src/audio_tokenizers/` (e.g., WavTokenizer) and integrates them into the unified pipeline architecture.

## Setup & Installation

This guide assumes that `benchmark-audio-tokenizer` is cloned in your home directory (`$HOME/benchmark-audio-tokenizer`).

> **Note**: This module uses `posttraining-data` as a git submodule (located at `src/repos/posttraining-data/`) for downloading datasets via the shared HF downloader. Make sure submodules are initialized when cloning:
> ```bash
> git submodule update --init --recursive
> ```

### Starting Interactive VSCode on Clariden

From the Clariden login node, navigate to the `benchmark-audio-tokenizer` directory and start an interactive VSCode session:

```bash
cd benchmark-audio-tokenizer

srun --time=08:00:00 \
  --environment=./ngc-24.11.toml \
  --account=infra01 \
  --partition=normal \
  --container-mounts="$HOME/vscode-cli-$(arch):/code" \
  --pty /code/code tunnel --accept-server-license-terms \
  --name="$CLUSTER_NAME-tunnel"
```

For detailed VSCode setup instructions, see: [https://github.com/swiss-ai/documentation/blob/main/pages/setup_vscode.md](https://github.com/swiss-ai/documentation/blob/main/pages/setup_vscode.md)

### Creating the Virtual Environment

Once in the `audio_tokenization` directory, create the virtual environment:

```bash
cd audio_tokenization
make venv
```

This creates a `.venv` directory with all required dependencies. Activate it with:

```bash
source .venv/bin/activate
```

The `Makefile` handles dependency compilation and ensures compatibility with the NGC 24.11 PyTorch environment.

## Repository Layout

- `data/raw` – raw dataset downloads e.g. from huggingface.
- `data/standardized` – schema-compliant HF datasets (audio resampled, metadata normalized).
- `data/tokenized` – final `.bin/.idx` files ready for Megatron/NeMo.
- `scripts/` – dataset-specific scripts (`download_librispeech.sh`, `standardize_librispeech.py`, …).
- `vokenizers/` – Audio tokenizer classes (`audio/audio_only.py`) - **target implementation**.
- `legacy_tokenizers/` – Legacy PoC tokenizer (`audio_tokenizer.py`) - **deprecated, will be replaced by `vokenizers/`**.
- `pipelines/` – Pipeline infrastructure (adapted from `vision_tokenization`):
  - `base.py` – Base pipeline classes (`BasePipeline`, `BaseTokenizerWorker`, `ProgressActor`)
  - `hf/pipeline.py` – HuggingFace dataset pipeline
  - `hf/workers.py` – Ray workers for distributed tokenization
- `utils/omni_tokenizer/` – Omni-tokenizer creation scripts (adapted from vision).
- `tokenize.py` – Main entry point for audio tokenization pipeline.

> **Note**: This module is part of `benchmark-audio-tokenizer`. Tokenizer implementations are located in `../src/audio_tokenizers/`, and pipeline infrastructure is imported from `../src/repos/benchmark-image-tokenzier/vision_tokenization/`.
> 
> **Important**: The `legacy_tokenizers/` directory (previously `tokenizers/`) contains proof-of-concept code and will be replaced by `vokenizers/audio/audio_only.py` which uses the Omni-Tokenizer system. It was renamed to avoid conflicts with HuggingFace's `tokenizers` package.

## Dataset Schema

All standardized datasets follow this unified schema:

| Field | Type | Required? | Description |
| :--- | :--- | :--- | :--- |
| `example_id` | string | **Required** | Unique stable identifier (dataset_name + original_id) |
| `dataset` | string | **Required** | Name: `"voxpopuli"`, `"gigaspeech"`, `"librispeech"`, etc. |
| `audio` | HF Audio object | **Required** | `{array, sampling_rate}` after resampling |
| `audio_path` | string | Optional | Path from HF dataset (not always available in streaming) |
| `sampling_rate` | int | **Required** | Standardize to **16_000** for all datasets |
| `duration` | float | **Required** | Duration in seconds (compute yourself) |
| `language` | string | Optional | ISO code if known (`"en"`, `"hr"`, `"de"`, `"multilingual"`) |
| `speaker_id` | string | Optional | Stringified speaker ID |
| `gender` | string | Optional | `"male"`, `"female"`, `"unknown"` |
| `accent` | string | Optional | Free text or `"unknown"` |
| `transcript` | string | Optional | Normalized text or raw text; empty string if unlabeled |
| `transcript_type` | string | Optional | `"gold"`, `"auto"`, `"none"` |
| `source_url` | string | Optional | For YouTube-derived corpora (GigaSpeech) |
| `metadata` | dict | Optional | Dataset-specific extras (timestamps, categories, etc.) |

## Next Steps: Stage 1 Speech Continuation MVP

**Status:** 🚧 In Progress

### Milestones

- [x] Decide special tokens (based on SpeechGPT/VioLA/Kimi-Audio literature review)
- [x] Create Audio Omni-Tokenizer (`utils/omni_tokenizer/create_audio_base.py`)
- [x] Implement `AudioOnlyTokenizer` class (`vokenizers/audio/audio_only.py`)
- [x] Adapt pipeline infrastructure (`pipelines/base.py`, `pipelines/hf/pipeline.py`, `pipelines/hf/workers.py`)
- [x] Create main `tokenize.py` entry point
- [x] Test end-to-end tokenization pipeline
- [ ] Adapt training pipeline for Stage 1 Speech Continuation
- [ ] Test end-to-end training run

### Special Tokens Decision

Based on literature review (SpeechGPT, VioLA, Kimi-Audio, SpiritLM), we use explicit block tokens to separate audio from text:

- `<|audio_start|>` - Start of audio block
- `<|audio_token_start|>` - Start of codec tokens (allows metadata insertion between start and token_start)
- `<|audio_end|>` - End of audio block

These tokens will be integrated into the Llama vocabulary via the Omni-Tokenizer system (similar to how vision tokens `<|visual token XXX|>` are handled).

**Current Status:** ✅ `AudioOnlyTokenizer` (`vokenizers/audio/audio_only.py`) is implemented and tested. It uses the Omni-Tokenizer system with dynamic token IDs. The legacy `AudioTokenizer` (`legacy_tokenizers/audio_tokenizer.py`) with hardcoded token IDs (0-6) is deprecated and will be replaced in the pipeline migration.

## Tokenization Steps

> **Note**: Currently, this repository processes only the LibriSpeech clean dataset as a proof of concept. Support for additional datasets (e.g., VoxPopuli, GigaSpeech) is planned and will be added in future updates.

This section describes the complete pipeline for processing LibriSpeech clean data from download to tokenized Megatron format.

### 1. Download

```bash
scripts/download_librispeech.sh
```

Uses the shared HF downloader to fetch the LibriSpeech *clean* config.

### 2. Standardize

```bash
python scripts/standardize_librispeech.py \
  --output-root data/standardized/librispeech_clean \
  --stage-dir $SCRATCH/standardized_tmp \
  --skip-existing \
  --num-proc 8 \
  --batch-size 256 \
  --inputs \
    data/raw/librispeech_asr/train.100 \
    data/raw/librispeech_asr/validation \
    data/raw/librispeech_asr/test
```

**Key options:**
- Pass any number of `--inputs`; each must point to a Hugging Face `save_to_disk` directory.
- Optional resampling via `--resample` (off by default because LibriSpeech is already 16 kHz).
- Use `--skip-existing` to avoid reprocessing completed splits and `--max-examples N` for smoke tests.
- `--stage-dir` can point to `$SCRATCH` or node-local storage; after completion, the split is moved to `--output-root`.
- Increase `--num-proc` to parallelize `datasets.map` across processes and use `--batch-size N` (>1) to amortize disk writes per worker.
- Produces the schema-compliant dataset; optionally emits Parquet with `--write-parquet`.

### 3. Tokenize

**Current (Proof of Concept):**
```bash
python scripts/tokenize_librispeech.py \
  --standardized-root data/standardized/librispeech_clean \
  --tokenized-root data/tokenized/librispeech_clean \
  --stage-dir $SCRATCH/tokenized_tmp \
  --log-dir $SCRATCH/token_logs \
  --num-shards 1 \
  --max-examples 2048  # optional smoke test
```

**Target (After Pipeline Migration):**
```bash
python -m audio_tokenization.tokenize hf \
  --mode audio_only \
  --dataset-name librispeech_asr \  # or --dataset-path /path/to/standardized hf dataset
  --dataset-split train.100 \
  --tokenizer-path /path/to/audio_omni_tokenizer \  # e.g., /iopsstor/scratch/.../audio_omni_tokenizer
  --output-dir ./data/tokenized/librispeech_clean \
  --num-gpus 4 \  # set per hardware
  --num-shards 100 \  # set per dataset size/checkpointing needs
  --device cuda \
  --audio-field audio
```

**Key differences:**
- Uses unified `tokenize.py` entry point (same as vision pipeline)
- Requires omni-tokenizer with audio tokens (created via `create_audio_base.py`)
- Uses `AudioOnlyTokenizer` class (analog to `EMUImageOnlyTokenizer`)
- Distributed processing via Ray workers (same infrastructure as vision)

See the **Special Tokens & Layout** section below for details on the token structure.

## Speech Continuation (Stage 1)

**Goal:** Audio-only (or audio-dominant) pretraining for speech continuation.

A minimal speech continuation model is planned for the near future. This script will load tokenized data, split each sequence into prompt/target segments (e.g., 50/50), and run a basic model to verify end-to-end setup. Results (loss, example IDs) will be logged. The goal is to quickly check the full Megatron-first speech pipeline, similar to the vision workflow.

**Architecture:**
- Base model: Llama 3B
- Input: Audio tokens wrapped with special structure tokens
- Output: Next audio token prediction
- Training: Standard causal language modeling objective

## Special Tokens & Layout

### Current Implementation (Proof of Concept) - **DEPRECATED**

`AudioTokenizer` (`legacy_tokenizers/audio_tokenizer.py`) wraps the WavTokenizer codes with hardcoded token IDs. This is legacy code and will be replaced by `AudioOnlyTokenizer`.

| Name | ID | Meaning |
| --- | --- | --- |
| `bos` | 0 | Sequence start |
| `eos` | 1 | Sequence end |
| `audio_start` | 2 | Start marker for audio media |
| `audio_end` | 3 | End marker for audio media |
| `meta_start` | 4 | Optional metadata block start |
| `meta_end` | 5 | Optional metadata block end |
| `token_start` | 6 | Beginning of codec frames |

All WavTokenizer frames (4096 codes) are shifted by `audio_token_offset = 7`. Example: codec ID `42` → stored ID `49`.

### Target Implementation (Omni-Tokenizer Based) - **✅ IMPLEMENTED**

`AudioOnlyTokenizer` (`vokenizers/audio/audio_only.py`) uses the Omni-Tokenizer system:

**Special Tokens:**
- `<|audio_start|>` - Start of audio block
- `<|audio_token_start|>` - Start of codec tokens (allows metadata insertion)
- `<|audio_end|>` - End of audio block

**Audio Codec Tokens:**
- `<|audio token 000000|>` through `<|audio token 004095|>` (WavTokenizer codebook)
- Automatically offset in the Omni-Tokenizer vocabulary (after text tokens)

**Typical Sequence Structure:**
```
[BOS]                              → Sequence start (from text tokenizer)
<|audio_start|>                    → Start of audio block
<|audio_token_start|>              → Start of codec tokens
<|audio token 000042|>              → Audio codec token (from WavTokenizer)
<|audio token 001234|>              → Audio codec token
...                                 → All audio tokens
<|audio_end|>                       → End of audio block
[EOS]                              → Sequence end (from text tokenizer)
```

**Key Differences:**
- Token IDs are dynamic (from Omni-Tokenizer vocabulary) instead of hardcoded
- Audio tokens are integrated into unified vocabulary (no manual offset needed)
- Compatible with Llama base model (tokens are part of model's embedding matrix)
- Same structure as vision tokens (`<|visual token XXX|>`)

**Omni-Tokenizer Creation:**
```bash
python utils/omni_tokenizer/create_audio_base.py \
  --text-tokenizer-path meta-llama/Llama-3.2-3B \
  --audio-tokenizer-path /path/to/wavtokenizer \
  --output-path ./llama3_audio_omni_tokenizer
```

This creates a tokenizer that extends Llama's vocabulary with:
1. Structure tokens: `<|audio_start|>`, `<|audio_token_start|>`, `<|audio_end|>`
2. Audio codec tokens: `<|audio token 000000|>` through `<|audio token 004095|>`
3. Reserved tokens: `<|RESERVED_OMNI_XXX|>` for future use

## Smoke Tests (Compute Node Session)

### Prerequisites
- You are already on a compute node in interactive `srun` mode (as described above)
- Virtualenv is available in `audio_tokenization/.venv`. (as described above)
- Standardized data is present:
  - LibriSpeech: `audio_tokenization/data/standardized/librispeech_clean/*` (from `scripts/standardize_librispeech.py`)
  - VoxPopuli (when ready): per-language HF datasets under `data/standardized/voxpopuli/<lang>/` (convert from raw via `scripts/standardize_voxpopuli.py` + submit script)

### One-time: build audio omni-tokenizer
```bash
cd ~/benchmark-audio-tokenizer
export PYTHONPATH=~/benchmark-audio-tokenizer:$PYTHONPATH
source audio_tokenization/.venv/bin/activate

python audio_tokenization/utils/omni_tokenizer/create_audio_base.py \
  --text-tokenizer-path meta-llama/Llama-3.2-3B \
  --output-path $SCRATCH/audio_omni_tokenizer
```
- Result: folder `$SCRATCH/audio_omni_tokenizer` containing the merged text+audio tokenizer. Reuse this path in all runs.

### Tokenization (LibriSpeech, 50 samples)
```bash
cd ~/benchmark-audio-tokenizer
export PYTHONPATH=~/benchmark-audio-tokenizer:$PYTHONPATH
source audio_tokenization/.venv/bin/activate

python -m audio_tokenization.tokenize hf \
  --tokenizer-path /iopsstor/scratch/cscs/lmantel/audio_omni_tokenizer \  # existing omni-tokenizer folder
  --output-dir $SCRATCH/tokenized_librispeech_smoke \                  # new smoke output
  --dataset-path audio_tokenization/data/standardized/librispeech_clean/train.100 \  # HF save_to_disk
  --dataset-name librispeech \
  --dataset-split train.100 \
  --mode audio_only \
  --num-gpus 1 \
  --num-shards 1 \
  --device cuda \
  --max-samples 50
```

### Expected outcome
- Run finishes without exceptions.
- Duration: a few seconds for 50 samples (Ray metric warnings are cosmetic).
- Output directory (e.g., `$SCRATCH/tokenized_librispeech_smoke_v2`) contains `.bin/.idx` and `dataset_info.json`.
- Logs show `Pipeline completed successfully` and `Total samples processed: 50`.
