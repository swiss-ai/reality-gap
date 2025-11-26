# Audio Tokenization

> Part of the `benchmark-audio-tokenizer` repository

This module provides a pipeline for tokenizing audio datasets for language model training. It uses tokenizer implementations from `../src/audio_tokenizers/` and reuses code from `../src/repos/benchmark-image-tokenzier/` (particularly Megatron utilities) to generate Megatron-compatible datasets.

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
- `data/tokenized` – token sequences plus auxiliary info for debugging/auditing.
- `data/megatron` – final `.bin/.idx` files ready for Megatron/NeMo.
- `scripts/` – driver scripts (`download_librispeech.sh`, `standardize_librispeech.py`, `tokenize_librispeech.py`, …).
- `tokenizers/` – AudioTokenizer adapter layer plus Megatron writer helper.

> **Note**: This module is part of `benchmark-audio-tokenizer`. Tokenizer implementations are located in `../src/audio_tokenizers/`, and Megatron utilities are imported from `../src/repos/benchmark-image-tokenzier/vision_tokenization/pipelines/`.

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

```bash
python scripts/tokenize_librispeech.py \
  --standardized-root data/standardized/librispeech_clean \
  --megatron-root data/megatron/librispeech_clean \
  --stage-dir $SCRATCH/tokenized_tmp \
  --log-dir $SCRATCH/token_logs \
  --num-shards 1 \
  --max-examples 2048  # optional smoke test
```

**Key options:**
- Uses `AudioTokenizer` (WavTokenizer backend plus BOS/AUDIO/META wrapper).
- Megatron-first: each split writes `data/megatron/<split>/tokens/{tokens.bin,tokens.idx,tokens.meta.json}` plus `logs/tokenization_stats.json`.
- Pass `--write-tokenized` if you still need HF `save_to_disk` datasets for debugging (written under `--tokenized-root`).
- `--stage-dir` can be pointed at `$SCRATCH` to keep intermediate `.bin/.idx` off Lustre until the split finishes.
- Support for `--num-shards/--shard-id` lets you process subsets of a split (useful for distributed jobs).

See the **Special Tokens & Layout** section below for details on the token structure and offsets.

## Speech Continuation
A minimal speech continuation model is planned for the near future. This script will load tokenized data, split each sequence into prompt/target segments (e.g., 50/50), and run a basic model to verify end-to-end setup. Results (loss, example IDs) will be logged. The goal is to quickly check the full Megatron-first speech pipeline, similar to the vision workflow.

## Special Tokens & Layout

`AudioTokenizer` (`tokenizers/audio_tokenizer.py`) wraps the WavTokenizer codes with additional markers so downstream models can distinguish sections. The default tokens are:

| Name | ID | Meaning |
| --- | --- | --- |
| `bos` | 0 | Sequence start |
| `eos` | 1 | Sequence end |
| `audio_start` | 2 | Start marker for audio media |
| `audio_end` | 3 | End marker for audio media |
| `meta_start` | 4 | Optional metadata block start |
| `meta_end` | 5 | Optional metadata block end |
| `token_start` | 6 | Beginning of codec frames |

All WavTokenizer frames (4096 codes) are shifted by `audio_token_offset = 7`. Example: codec ID `42` → stored ID `49`. A typical sequence structure is:

```
[BOS] (ID: 0)                    → Sequence start
<|audio_start|> (ID: 2)          → Start of audio block
<|meta_start|> (ID: 4)           → Start of optional metadata
<|meta_end|> (ID: 5)             → End of metadata
<|token_start|> (ID: 6)          → Start of codec tokens
<shifted_codec_tokens...>        → Audio codes (offset +7)
<|audio_end|> (ID: 3)            → End of audio block
[EOS] (ID: 1)                    → Sequence end
```

JSON metadata (`tokenizer_info`) preserves counts, sampling rate, tokens/s, etc. without lengthening the sequence.
