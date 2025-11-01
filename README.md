# benchmark-audio-tokenizers

A comprehensive benchmarking framework for evaluating discrete audio tokenizer performance across different models and datasets.

## Overview

This repository provides tools and scripts for systematically evaluating audio tokenizers on speech data. Currently, we benchmark tokenizers on the EuroSpeech dataset, with support for NeuCodec as our first evaluated model.

## Project Structure

```
.
├── examples/                      # (not currently in use)
├── jobs/                          # Job submission files for cluster computing
├── logs/                          # Execution logs (.out and .err files per language)
├── metrics/                       # Evaluation results and metrics (JSON output)
├── scripts/                       # All Python scripts and shell scripts
│   ├── download_eurospeech.py     # Download of EuroSpeech
│   ├── download_fleurs.py         # Download of FLEURS
│   ├── neucodec_evaluation.py     # NeuCodec evaluation script
│   ├── test_minimal.py            # Minimal testing script
│   ├── verify_eurospeech.py       # Dataset verification script
│   └── run_eurospeech.sh          # Shell script for compute node execution
├── src/
│   ├── audio_tokenizers/          # Tokenizer implementations and wrappers
│   └── repos/                     # External repository dependencies
├── venv/                          # Virtual environment for download_[dataset] scripts
├── neucodec-venv/                 # Virtual environment for NeuCodec evaluation
├── requirements_venv.txt          # Dependencies for login node environment
└── requirements_neucodec-venv.txt # Dependencies for NeuCodec environment
```

## Current Status

**Datasets:**
- ✅ EuroSpeech 
  - 22 languages: bosnia-herzegovina, bulgaria, croatia, denmark, estonia, finland, france, germany, greece, iceland, italy, latvia, lithuania, malta, norway, portugal, serbia, slovakia, slovenia, sweden, uk, ukraine
  - 100 samples per language
  - Cached in `$SCRATCH/benchmark-audio-tokenizer/datasets/eurospeech_cache`

- ✅ FLEURS 
  - 102 languages available
  - 100 samples per language
  - Cached in `$SCRATCH/benchmark-audio-tokenizer/datasets/fleurs_cache`  

**Tokenizers Evaluated:**
- ✅ NeuCodec

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- Access to compute cluster with scratch storage

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Set up the login node environment (for dataset download and verification):
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements_venv.txt
```

3. Set up the NeuCodec evaluation environment (for compute nodes):
```bash
python -m venv neucodec-venv
source neucodec-venv/bin/activate
pip install -r requirements_neucodec-venv.txt
```

### Downloading the Datasets

#### EuroSpeech

The EuroSpeech dataset is automatically downloaded to your scratch directory:

```bash
source venv/bin/activate
python scripts/download_eurospeech.py
```

This will download 100 samples for each of the 22 supported languages to:
```
$SCRATCH/benchmark-audio-tokenizer/datasets/eurospeech_cache/
```

Supported languages: bosnia-herzegovina, bulgaria, croatia, denmark, estonia, finland, france, germany, greece, iceland, italy, latvia, lithuania, malta, norway, portugal, serbia, slovakia, slovenia, sweden, uk, ukraine.


#### FLEURS
Edit scripts/download_fleurs.py to select languages.
```bash
python scripts/download_fleurs.py
```

### Running Evaluations

#### On Compute Nodes (recommended)

Use the provided shell script to run all evaluations on compute nodes:

```bash
sbatch scripts/run_eurospeech.sh
```

#### Manual Evaluation

Evaluate a specific language:
```bash
source neucodec-venv/bin/activate
python scripts/neucodec_evaluation.py --language germany
```

Evaluate multiple languages:
```bash
python scripts/neucodec_evaluation.py --languages germany france italy
```

Evaluate all languages:
```bash
python scripts/neucodec_evaluation.py
```

Results are saved to the `metrics/` directory as JSON files.

Execution logs (`.out` and `.err` files) for each language evaluation are saved in the `logs/` directory.

## Evaluation Metrics

The framework computes comprehensive reconstruction quality metrics:

- **MSE** (Mean Squared Error): Measures reconstruction error
- **SNR** (Signal-to-Noise Ratio): Overall signal quality in dB
- **SDR** (Signal-to-Distortion Ratio): Distortion measurement in dB
- **PESQ**: Perceptual Evaluation of Speech Quality (1.0-4.5 scale)
- **STOI**: Short-Time Objective Intelligibility (0-1 scale)
- **ESTOI**: Extended STOI for improved accuracy

Additionally tracks:
- Tokens per second
- Compression ratio


## Contributing

This project is part of the Data Science Lab course at ETH Zurich, autumn semester 2025 (Leonard Mantel, Melanie Rieff).

## License


## Acknowledgments

- Eurospeech dataset
- NeuCodec implementation