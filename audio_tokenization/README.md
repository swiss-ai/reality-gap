# Audio Tokenization Pipeline

Distributed audio tokenization pipeline for creating Megatron-LM compatible datasets.

## Supported Pipelines

- `dataset_type: wds`
  - Ray-based WebDataset (`.tar`) tokenization.
  - Good when your data is already in shard archives.
- `dataset_type: lhotse`
  - DDP-based tokenization from pre-built Lhotse Shar manifests.
  - Best for large-scale dynamic bucketing and checkpointed long runs.

## Features

- Distributed processing across multiple GPUs/nodes
- WavTokenizer support with omni-tokenizer special-token wrapping
- Megatron output format (`.bin` + `.idx`)
- Resume support
- Optional W&B logging

## Directory Structure

```text
audio_tokenization/
├── tokenize.py
├── configs/
│   ├── config.yaml
│   └── dataset/
│       ├── peoples_speech_wds.yaml
│       ├── audioset_lhotse.yaml
│       └── unsupervised_peoples_speech_lhotse.yaml
├── vokenizers/
│   └── wavtokenizer/
└── pipelines/
    ├── base.py
    ├── shard_assignment.py
    ├── shard_io.py
    ├── wds/
    └── lhotse/
```

## Quick Start

```bash
# WDS pipeline
python -m audio_tokenization.tokenize dataset=peoples_speech_wds num_gpus=4

# Lhotse pipeline
python -m audio_tokenization.tokenize dataset=audioset_lhotse num_gpus=4

# Resume
python -m audio_tokenization.tokenize dataset=audioset_lhotse resume=true num_gpus=4
```

## Configuration

Main config: `audio_tokenization/configs/config.yaml`

Dataset configs:
- `audio_tokenization/configs/dataset/peoples_speech_wds.yaml`
- `audio_tokenization/configs/dataset/audioset_lhotse.yaml`
- `audio_tokenization/configs/dataset/unsupervised_peoples_speech_lhotse.yaml`

Tokenizer config:

```yaml
tokenizer:
  path: /path/to/omni_tokenizer
  sampling_rate: 24000
  torch_compile: false
  trim_last_tokens: 5
```

## Output Format

Generated files:
- `rank_<id>_shard_<shard>_<total>.bin`
- `rank_<id>_shard_<shard>_<total>.idx`
- `dataset_info.json`

Lhotse pipeline also writes rank checkpoints for resume.

## Notes

- `shards_per_gpu` is no longer used.
- WDS uses explicit tar shard lists/globs from dataset config.
- Lhotse expects pre-built Shar data (`shar_dir` + `shar_index.json`).

## Requirements

- Python 3.8+
- PyTorch
- Ray (WDS pipeline)
- Lhotse (Lhotse pipeline)
- torchaudio
- transformers
