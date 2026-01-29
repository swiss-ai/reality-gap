# Audio Tokenization Pipeline

Distributed audio tokenization pipeline for creating Megatron-LM compatible datasets.

## Features

- **Distributed Processing**: Ray-based parallel tokenization across multiple GPUs
- **WavTokenizer Support**: Primary support for WavTokenizer-40 (40 tokens/second)
- **HuggingFace Integration**: Direct processing of HuggingFace audio datasets
- **Special Token Encapsulation**: Optional wrapping with `<|audio_start|>` and `<|audio_end|>` tokens
- **Megatron Output Format**: Binary `.bin` + `.idx` format for efficient training
- **Resume Support**: Checkpoint-based resume for interrupted runs

## Directory Structure

```
audio_tokenization/
├── tokenize.py              # Main CLI entry point
├── configs/                 # Example configuration files
│   ├── example_librispeech.json
│   ├── example_common_voice.json
│   └── example_with_encapsulation.json
├── vokenizers/              # Audio tokenizer wrappers
│   ├── base.py              # Base classes and encapsulation
│   └── audio_tokenizer.py   # WavTokenizer wrapper
└── pipelines/               # Processing pipelines
    ├── base.py              # Base pipeline classes
    ├── indexed_dataset_megatron.py  # Megatron output format
    └── hf/                  # HuggingFace pipeline
        ├── pipeline.py      # Main HF pipeline
        └── workers.py       # Ray distributed workers
```

## Quick Start

### Basic Usage

```bash
# Tokenize LibriSpeech with WavTokenizer
python -m audio_tokenization.tokenize hf --config configs/example_librispeech.json

# With more GPUs
python -m audio_tokenization.tokenize hf --config configs/example_librispeech.json --num-gpus 4

# Resume interrupted run
python -m audio_tokenization.tokenize hf --config configs/example_librispeech.json --resume
```

### List Available Tokenizers

```bash
python -m audio_tokenization.tokenize list
```

## Configuration

Example configuration file:

```json
{
    "tokenizer_name": "wavtokenizer-40",
    "omni_tokenizer_path": null,
    "audio_token_offset": 0,
    "dataset_name": "openslr/librispeech_asr",
    "config_name": "clean",
    "dataset_split": "train.100",
    "output_dir": "./data/tokenized",
    "num_gpus": 4,
    "num_shards": 16,
    "vocab_size": 4096,
    "audio_field": "audio",
    "min_duration": 1.0,
    "max_duration": 30.0
}
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

Audio tokens can be encapsulated with special tokens when `omni_tokenizer_path` is provided:

```
[audio_start_id] [audio_token_0] ... [audio_token_N] [audio_end_id]
```

The special tokens use the reserved OMNI tokens:
- `<|RESERVED_OMNI_008|>` → `<|audio_start|>`
- `<|RESERVED_OMNI_009|>` → `<|audio_end|>`

For reference, vision tokens use:
- `<|RESERVED_OMNI_001|>` through `<|RESERVED_OMNI_007|>` for image structure

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
from audio_tokenization.pipelines.hf import HFDatasetPipeline

pipeline = HFDatasetPipeline(
    tokenizer_name="wavtokenizer-40",
    dataset_name="openslr/librispeech_asr",
    config_name="clean",
    dataset_split="train.100",
    output_dir="./output",
    num_gpus=4,
)

result = pipeline.run()
print(result["statistics"])
```

## Requirements

- Python 3.8+
- PyTorch
- Ray
- HuggingFace Datasets
- torchaudio
- transformers (for special token handling)
