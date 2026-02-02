# Audio Tokenization Pipeline

Distributed audio tokenization pipeline for creating Megatron-LM compatible datasets.

## Features

- **Distributed Processing**: Ray-based parallel tokenization across multiple GPUs
- **WavTokenizer Support**: Primary support for WavTokenizer-40 (40 tokens/second)
- **HuggingFace Integration**: Direct processing of HuggingFace audio datasets
- **Bucket Filtering**: Pre-filter samples by audio length for efficient batch processing
- **Special Token Encapsulation**: Wraps audio tokens with `<|audio_start|>` and `<|audio_end|>` tokens
- **Megatron Output Format**: Binary `.bin` + `.idx` format for efficient training
- **Resume Support**: Checkpoint-based resume for interrupted runs

## Shard Assignment Modes

Two shard assignment strategies are available:

| Mode | Description | Best For |
|------|-------------|----------|
| `shared` | Dynamic queue-based distribution | Variable-length audio (load balancing) |
| `static` | Pre-assigned shards per worker | Uniform-length bucketed data (best throughput) |

### Why CPU Workers Matter

Audio files in datasets often have different sampling rates (e.g., 16kHz, 44.1kHz, 48kHz), but batch
processing requires uniform sample rates. The pipeline uses HuggingFace's `Audio` feature to lazily
decode and resample audio to the target rate (24kHz for WavTokenizer):

```python
dataset = dataset.cast_column(audio_field, Audio(sampling_rate=target_sample_rate))
```

Audio decoding and resampling are CPU-bound, single-threaded operations. The `decode_workers_per_gpu`
parameter controls how many parallel processes handle this in PyTorch's DataLoader, allowing multiple
audio files to be decoded/resampled simultaneously while the GPU processes the current batch.

### Benchmark: AudioSet bucket_240000 (17,288 samples, 4 GPUs, batch_size=256)

```
┌────────┬─────────┬────────┬─────────────┬────────────┬─────────┐
│  Mode  │ CPU/GPU │  Time  │ Samples/sec │ Tokens/sec │ Speedup │
├────────┼─────────┼────────┼─────────────┼────────────┼─────────┤
│ Shared │ 16      │ 58.25s │ 296.8       │ 119,907    │ 1.0x    │
├────────┼─────────┼────────┼─────────────┼────────────┼─────────┤
│ Static │ 16      │ 32.10s │ 538.5       │ 217,562    │ 1.81x   │
├────────┼─────────┼────────┼─────────────┼────────────┼─────────┤
│ Shared │ 48      │ 46.57s │ 371.3       │ 149,986    │ 1.25x   │
├────────┼─────────┼────────┼─────────────┼────────────┼─────────┤
│ Static │ 48      │ 29.04s │ 595.4       │ 240,522    │ 2.01x   │
└────────┴─────────┴────────┴─────────────┴────────────┴─────────┘
```

**Prefetch factor comparison (Static mode, 48 CPU/GPU):**
```
┌──────────┬────────┬─────────────┬────────────┬─────────┐
│ Prefetch │  Time  │ Samples/sec │ Tokens/sec │ Speedup │
├──────────┼────────┼─────────────┼────────────┼─────────┤
│ 4        │ 31.79s │ 543.9       │ 219,724    │ 1.0x    │
├──────────┼────────┼─────────────┼────────────┼─────────┤
│ 16       │ 30.38s │ 569.1       │ 229,906    │ 1.05x   │
└──────────┴────────┴─────────────┴────────────┴─────────┘
```

**Recommendation**: Use `static` mode with bucket filtering for uniform-length audio clips.
Higher prefetch factors provide minimal benefit when CPU workers are already saturated.

## Directory Structure

```
audio_tokenization/
├── tokenize.py              # Main CLI entry point (Hydra-based)
├── configs/                 # Hydra configuration files
│   ├── config.yaml          # Main config with pipeline settings
│   └── dataset/             # Dataset-specific configs
│       ├── audioset.yaml
│       └── audioset_bucketed.yaml
├── vokenizers/              # Audio tokenizer wrappers
│   ├── base.py              # Base classes and encapsulation
│   └── wavtokenizer/        # WavTokenizer implementations
└── pipelines/               # Processing pipelines
    ├── base.py              # Base pipeline classes
    └── hf/                  # HuggingFace pipeline
        ├── pipeline.py      # Main HF pipeline
        ├── bucket_pipeline.py   # Bucketed pipeline (pre-filtered by length)
        ├── bucket_index.py  # Bucket metadata loader
        └── workers/         # Ray distributed workers
            ├── base.py      # Worker class
            ├── static_runner.py   # Static shard assignment
            └── shared_runner.py   # Shared queue assignment
```

## Quick Start

### Basic Usage (Hydra CLI)

```bash
# Tokenize AudioSet with default settings
python -m audio_tokenization.tokenize dataset=audioset

# With bucket filtering (uniform-length samples for efficient batching)
python -m audio_tokenization.tokenize dataset=audioset_bucketed \
    dataset.bucket.target_bucket=240000

# Override settings via CLI
python -m audio_tokenization.tokenize dataset=audioset_bucketed \
    num_gpus=4 \
    num_shards=16 \
    batch_size=256 \
    shard_assignment=static

# Resume interrupted run
python -m audio_tokenization.tokenize dataset=audioset_bucketed resume=true
```

## Configuration

Configuration uses Hydra. Example `config.yaml`:

```yaml
defaults:
  - dataset: audioset_bucketed
  - _self_

# Pipeline settings
output_dir: ./data/tokenized
num_gpus: 4
num_shards: 16
batch_size: 256
shard_assignment: static              # shared or static
decode_workers_per_gpu: 16            # multiprocessing decode/resample per GPU worker

# Tokenizer settings
tokenizer:
  path: /path/to/omni_tokenizer       # Must have audio tokens added
  sampling_rate: 24000

# Filtering
min_duration: 1.0
max_duration: 30.0

# W&B logging
wandb:
  enabled: true
  project: audio-tokenization
```

Example dataset config (`dataset/audioset_bucketed.yaml`):

```yaml
dataset_name: agkphysics/AudioSet
config_name: full
dataset_split: bal_train
audio_field: audio
cache_dir: /path/to/cache

bucket:
  enabled: true
  metadata_dir: /path/to/length_buckets
  target_bucket: 240000    # 10-second clips at 24kHz
  shuffle_seed: 42
```

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `tokenizer_name` | Audio tokenizer to use | `wavtokenizer-40` |
| `omni_tokenizer_path` | Path to omni_tokenizer for special tokens | `null` |
| `audio_token_offset` | Offset to add to audio token IDs | `0` |
| `dataset_name` | HuggingFace dataset name | Required |
| `config_name` | Dataset configuration | `null` |
| `dataset_split` | Dataset split to process | `train` |
| `output_dir` | Output directory | `./output` |
| `num_gpus` | Number of GPUs | `1` |
| `num_shards` | Number of shards for processing | `num_gpus * 4` |
| `vocab_size` | Vocabulary size (for optimal dtype) | `4096` |
| `audio_field` | Audio field name in dataset | `audio` |
| `min_duration` | Minimum audio duration (seconds) | `null` |
| `max_duration` | Maximum audio duration (seconds) | `null` |
| `max_samples` | Maximum samples to process | `null` |

## Special Tokens

Audio tokens are encapsulated with special tokens:

```
[BOS] [audio_start] [audio_token_0] ... [audio_token_N] [audio_end] [EOS]
```

The special tokens use the reserved OMNI tokens from the omni_tokenizer:
- `<|RESERVED_OMNI_008|>` → `<|audio_start|>`
- `<|RESERVED_OMNI_009|>` → `<|audio_end|>`

Audio token IDs are offset by 131,272 in the omni_tokenizer vocabulary.

## Output Format

The pipeline produces Megatron-LM compatible binary datasets:

- `rank_X_shard_Y_Z.bin` - Binary token data
- `rank_X_shard_Y_Z.idx` - Index file with offsets
- `dataset_info.json` - Statistics and metadata

The optimal data type is automatically selected based on vocabulary size:
- vocab < 256: `uint8`
- vocab < 65536: `uint16`
- otherwise: `int32`

## Tokenizer Details

### WavTokenizer-40

- **Frame Rate**: 40 tokens/second
- **Sample Rate**: 24 kHz
- **Codebook Size**: 4,096 tokens
- **Single Codebook**: Yes

## API Usage

```python
from audio_tokenization.pipelines.hf import BucketedHFDatasetPipeline

pipeline = BucketedHFDatasetPipeline(
    tokenizer_path="/path/to/omni_tokenizer",
    output_dir="./data/tokenized",
    dataset_name="agkphysics/AudioSet",
    dataset_split="bal_train",
    config_name="full",
    mode="audio_only",
    num_gpus=4,
    num_shards=16,
    device="cuda",
    batch_size=256,
    shard_assignment="static",
    bucket_metadata_dir="/path/to/length_buckets",
    target_bucket=240000,  # 10-second clips at 24kHz
)

result = pipeline.run()
print(f"Processed: {result['total_samples']} samples, {result['total_tokens']} tokens")
```

## Requirements

- Python 3.8+
- PyTorch
- Ray
- HuggingFace Datasets
- torchaudio
- transformers (for special token handling)

### Installing torchaudio

```bash
uv pip install --system --break-system-packages --no-deps --no-build-isolation git+https://github.com/pytorch/audio.git@release/2.9
```
