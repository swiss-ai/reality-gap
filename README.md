# benchmark-audio-tokenizers

A comprehensive benchmarking framework for evaluating discrete audio tokenizer performance across different models and datasets.

## Overview

This repository provides tools and scripts for systematically evaluating audio tokenizers on multilingual speech data. We support multiple datasets (EuroSpeech, FLEURS) with automatic dataset detection and unified evaluation pipeline.

## Project Structure

```
.
├── examples/                      # (not currently in use)
├── jobs/                          # Job submission files for cluster computing
├── logs/                          # Execution logs (.out and .err files per language)
├── metrics/                       # Evaluation results and metrics (JSON output)
├── scripts/                       # All Python scripts and shell scripts
│   ├── download_eurospeech.py     # Download EuroSpeech dataset
│   ├── download_fleurs.py         # Download FLEURS dataset
│   ├── download_all_fleurs.sh     # SLURM batch download for FLEURS
│   ├── neucodec_evaluation.py     # Multi-dataset evaluation script
│   ├── run_eurospeech.sh          # SLURM batch evaluation for EuroSpeech
│   ├── test_minimal.py            # Minimal testing script
│   └── verify_eurospeech.py       # Dataset verification script
├── src/
│   ├── audio_tokenizers/          # Tokenizer implementations and wrappers
│   └── repos/                     # External repository dependencies
├── venv/                          # Virtual environment for downloads (also: fleurs-venv)
├── neucodec-venv/                 # Virtual environment for evaluations
├── requirements_venv.txt          # Dependencies for download environment
└── requirements_neucodec-venv.txt # Dependencies for evaluation environment
```

## Current Status

**Datasets:**
- ✅ **EuroSpeech** 
  - 22 languages: bosnia-herzegovina, bulgaria, croatia, denmark, estonia, finland, france, germany, greece, iceland, italy, latvia, lithuania, malta, norway, portugal, serbia, slovakia, slovenia, sweden, uk, ukraine
  - 100 samples per language
  - Cached in `$SCRATCH/benchmark-audio-tokenizer/datasets/eurospeech_cache`

- ✅ **FLEURS**
  - 40 languages configured (102 available): ast_es, ca_es, nl_nl, en_us, gl_es, hu_hu, ga_ie, kea_cv, lb_lu, oc_fr, es_419, cy_gb, hy_am, be_by, cs_cz, ka_ge, mk_mk, pl_pl, ro_ro, ru_ru, cmn_hans_cn, yue_hant_hk, ja_jp, ko_kr, hi_in, bn_in, ta_in, te_in, th_th, vi_vn, id_id, af_za, sw_ke, am_et, yo_ng, ar_eg, tr_tr, he_il, fa_ir
  - 100 samples per language
  - Cached in `$SCRATCH/benchmark-audio-tokenizer/datasets/fleurs_cache`

**Total Coverage (as of November 1st 2025):** 62 languages across 2 datasets

**Tokenizers Evaluated:**
- ✅ NeuCodec

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for evaluation)
- Access to compute cluster with scratch storage
- SLURM workload manager (optional, for parallel processing)

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd benchmark-audio-tokenizer
```

2. Set up the download environment:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements_venv.txt
```

3. Set up the evaluation environment:
```bash
python -m venv neucodec-venv
source neucodec-venv/bin/activate
pip install -r requirements_neucodec-venv.txt
```

## Downloading Datasets

### EuroSpeech (Sequential Download)

```bash
source venv/bin/activate
python scripts/download_eurospeech.py
```

Downloads 100 samples for each of 22 languages to:
```
$SCRATCH/benchmark-audio-tokenizer/datasets/eurospeech_cache/
```

### FLEURS (Parallel Download via SLURM - Recommended)

```bash
# Make script executable
chmod +x scripts/download_fleurs.sh

# Submit 40 parallel download jobs
bash scripts/download_fleurs.sh
```

**Alternative (Sequential):**
```bash
source venv/bin/activate

# Download all 40 languages 
python scripts/download_fleurs.py

# Or download specific languages
python scripts/download_fleurs.py --language en_us
python scripts/download_fleurs.py --languages en_us ja_jp ko_kr
```

Downloads to: `$SCRATCH/benchmark-audio-tokenizer/datasets/fleurs_cache/`


## Running Evaluations

### Multi-Dataset Evaluation Features

The evaluation script automatically:
- Detects which dataset a language belongs to
- Loads from the correct cache directory
- Tracks dataset origin in results
- Supports mixed-dataset evaluation

### Evaluation Commands

#### Single Language (Auto-detects dataset)
```bash
source neucodec-venv/bin/activate

# EuroSpeech language
python scripts/neucodec_evaluation.py --language germany

# FLEURS language
python scripts/neucodec_evaluation.py --language en_us
```

#### Multiple Languages (Mixed datasets OK!)
```bash
# Mix of EuroSpeech and FLEURS
python scripts/neucodec_evaluation.py --languages germany en_us ja_jp france
```

#### All Languages from One Dataset
```bash
# All EuroSpeech languages (22)
python scripts/neucodec_evaluation.py --dataset eurospeech

# All FLEURS languages (40)
python scripts/neucodec_evaluation.py --dataset fleurs
```

#### All Languages from All Datasets
```bash
# All 62 languages
python scripts/neucodec_evaluation.py --dataset all

# Or simply (same as above)
python scripts/neucodec_evaluation.py
```

### SLURM Batch Evaluation

#### EuroSpeech (22 parallel jobs)
```bash
bash scripts/run_eurospeech.sh
```

#### FLEURS (40 parallel jobs)
Create and run a similar batch script for FLEURS languages, or submit individual jobs:
```bash
sbatch -J neucodec_en_us scripts/run_single_eval.sh en_us
```

### Results Organization

Results are automatically organized by dataset:

```
metrics/
├── neucodec_eurospeech_germany_results.json      # Per-language results
├── neucodec_fleurs_en_us_results.json            # Per-language results
├── neucodec_eurospeech_summary.json              # All EuroSpeech results
├── neucodec_fleurs_summary.json                  # All FLEURS results
└── neucodec_all_results.json                     # Combined results
```

Each result includes:
- Language name
- Dataset origin
- All metrics (MSE, SNR, SDR, PESQ, STOI, ESTOI)
- Tokenization statistics

Execution logs are saved in `logs/` directory.

## Evaluation Metrics

The framework computes comprehensive reconstruction quality metrics:

### Reconstruction Quality
- **MSE** (Mean Squared Error): Measures reconstruction error
- **SNR** (Signal-to-Noise Ratio): Overall signal quality in dB
- **SDR** (Signal-to-Distortion Ratio): Distortion measurement in dB
- **PESQ**: Perceptual Evaluation of Speech Quality (1.0-4.5 scale)
- **STOI**: Short-Time Objective Intelligibility (0-1 scale)
- **ESTOI**: Extended STOI for improved accuracy

### Tokenization Efficiency
- **Tokens per second**: Tokenization rate
- **Compression ratio**: Original size / Token size

All metrics include: mean, standard deviation, min, max, and median values.

## Adding New Datasets

To add a new dataset (e.g., LibriSpeech):

1. **Create download script** following `download_fleurs.py` pattern
2. **Ensure data structure compatibility**:
   ```python
   {
       'audio': {'array': ..., 'sampling_rate': ..., 'path': ...},
       'text': ...,
       'language': ...,
       'sample_id': ...
   }
   ```
3. **Update evaluation script** - Add to DATASETS dictionary:
   ```python
   DATASETS = {
       'eurospeech': {...},
       'fleurs': {...},
       'librispeech': {
           'cache_dir': os.path.join(SCRATCH_DIR, "librispeech_cache"),
           'languages': ['en_train', 'en_test', 'en_dev']
       }
   }
   ```

The evaluation script will automatically support the new dataset!

## Dataset Details

### Unified Structure

Both datasets use identical structure for compatibility:
```python
{
    "audio": {
        "array": np.ndarray,      # Audio waveform
        "sampling_rate": int,     # Sample rate (Hz)
        "path": str              # Original file path
    },
    "text": str,                 # Transcription
    "language": str,             # Language code/name
    "sample_id": int            # Sample index
}
```

### Language Codes

**EuroSpeech:** Full country names (e.g., `germany`, `france`, `uk`)

**FLEURS:** ISO codes (e.g., `en_us`, `de_de`, `cmn_hans_cn`)

See `scripts/download_fleurs.py` for complete FLEURS language list.

## Troubleshooting

### Dataset Download Issues
- **FLEURS "BuilderConfig not found"**: Check language codes in error message
- **FLEURS "trust_remote_code required"**: Already handled in download script
- **EuroSpeech validation split**: Slovakia uses validation split automatically

### Evaluation Issues
- **"Language not found"**: Verify language code spelling and download status
- **"Dataset not found at PATH"**: Check `$SCRATCH` variable and cache directories
- **PESQ errors**: Automatically resampled to 16kHz, check audio quality
- **Memory issues**: Reduce batch size or use SLURM with more memory

### SLURM Issues
- **Jobs queued**: Cluster is busy, jobs will run when resources available
- **Cancel jobs**: `scancel -u $USER -n download_fleurs` or `scancel -u $USER -n neucodec`

## Contributing

This project is part of the Data Science Lab course at ETH Zurich, autumn semester 2025.

**Contributors:**
- Leonard Mantel
- Melanie Rieff

## Citation

If you use this benchmarking framework, please cite the relevant datasets:

**EuroSpeech:**
```bibtex
@dataset{eurospeech,
  title={EuroSpeech Dataset},
  author={Disco-eth},
  year={2024},
  url={https://huggingface.co/datasets/disco-eth/EuroSpeech}
}
```

**FLEURS:**
```bibtex
@article{fleurs2022arxiv,
  title={FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech},
  author={Conneau, Alexis and Ma, Min and Khanuja, Simran and Zhang, Yu and Axelrod, Vera and Dalmia, Siddharth and Riesa, Jason and Rivera, Clara and Bapna, Ankur},
  journal={arXiv preprint arXiv:2205.12446},
  year={2022},
  url={https://arxiv.org/abs/2205.12446}
}
```

## License
