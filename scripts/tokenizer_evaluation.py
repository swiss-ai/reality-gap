"""
Evaluate registered audio tokenizer backends on multilingual speech and sound datasets
and export metric summaries to `metrics/`.

Usage examples:
    python scripts/tokenizer_evaluation.py --tokenizer xcodec2 --language germany
    python scripts/tokenizer_evaluation.py --tokenizer cosyvoice2 --language germany
"""
import os
import sys
import torch
import numpy as np
from datasets import load_from_disk
import json
from pathlib import Path
from tqdm import tqdm
import scipy.signal as signal
import argparse
import static_ffmpeg
static_ffmpeg.add_paths()
import stempeg

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'src'))
from audio_tokenizers import get_tokenizer

from pesq import pesq
from pystoi import stoi
import museval

# Base directories
DATASET_DIR = Path("/capstor/store/cscs/swissai/infra01/audio-datasets")
RESULTS_DIR = project_root / "metrics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset configurations
DATASETS = {
    'eurospeech': {
        'cache_dir': DATASET_DIR / "eurospeech_cache",
        'languages': [
            "bosnia-herzegovina", "bulgaria", "croatia", "denmark", "estonia", "finland",
            "france", "germany", "greece", "iceland", "italy", "latvia", "lithuania",
            "malta", "norway", "portugal", "serbia", "slovakia", "slovenia",
            "sweden", "uk", "ukraine"
        ]
    },
    'fleurs': {
        'cache_dir': DATASET_DIR / "fleurs_cache",
        'languages': [
            # Western Europe
            "ast_es", "ca_es", "nl_nl", "en_us", "gl_es", "hu_hu", "ga_ie", 
            "kea_cv", "lb_lu", "oc_fr", "es_419", "cy_gb",
            # Eastern Europe
            "hy_am", "be_by", "cs_cz", "ka_ge", "mk_mk", "pl_pl", "ro_ro", "ru_ru",
            # Asia - CJK
            "cmn_hans_cn", "yue_hant_hk", "ja_jp", "ko_kr",
            # Asia - South Asia
            "hi_in", "bn_in", "ta_in", "te_in",
            # Asia - Southeast Asia
            "th_th", "vi_vn", "id_id",
            # Africa
            "af_za", "sw_ke", "am_et", "yo_ng",
            # Middle East
            "ar_eg", "tr_tr", "he_il", "fa_ir"
        ]
    },
    "naturelm": {
        "cache_dir": DATASET_DIR / "naturelm_cache",
        "languages": [
            "Xeno-canto", "WavCaps", "NatureLM", "Watkins",
            "iNaturalist", "Animal Sound Archive"
        ],
        "source_field": "source_dataset",  # Use this field to filter by source
    },
    "gtzan": {
        "cache_dir": DATASET_DIR / "gtzan_cache",
        "languages": [
            "blues", "classical", "country", "disco", "hiphop",
            "jazz", "metal", "pop", "reggae", "rock"
        ],
        "source_field": "genre_name",  # Use this field to filter by genre
    },
    }

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {device}")

def get_dataset_for_language(language):
    """
    Determine which dataset a language belongs to and return dataset name and cache dir.
    Returns: (dataset_name, cache_dir, language) or (None, None, None) if not found
    """
    for dataset_name, config in DATASETS.items():
        if language in config['languages']:
            return dataset_name, config['cache_dir'], language
    
    # Language not found in any dataset
    return None, None, None

def resample_audio(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio

    duration = len(audio) / orig_sr
    num_samples = int(duration * target_sr)
    
    resampled = signal.resample(audio, num_samples)
    return resampled
    

def calculate_sdr(reference, estimated):
    try:
        min_len = min(len(reference), len(estimated))
        reference = reference[:min_len]
        estimated = estimated[:min_len]
        
        reference = reference.reshape(1, -1)
        estimated = estimated.reshape(1, -1)
        
        sdr, isr, sir, sar = museval.evaluate(reference, estimated)
        
        if isinstance(sdr, np.ndarray):
            return float(np.mean(sdr))
        else:
            return float(sdr)
    except Exception as e:
        print(f"SDR calculation error: {e}")
        return None

def calculate_reconstruction_metrics(original, reconstructed, original_sr, reconstructed_sr):
    metrics = {}
    
    if reconstructed_sr != original_sr:
        reconstructed_resampled = resample_audio(reconstructed, reconstructed_sr, original_sr)
    else:
        reconstructed_resampled = reconstructed

    min_len = min(len(original), len(reconstructed_resampled))
    original_aligned = original[:min_len]
    reconstructed_aligned = reconstructed_resampled[:min_len]
    
    original_normalized = original_aligned / (np.abs(original_aligned).max() + 1e-8)
    reconstructed_normalized = reconstructed_aligned / (np.abs(reconstructed_aligned).max() + 1e-8)
    
    # 1. MSE 
    mse = np.mean((original_normalized - reconstructed_normalized) ** 2)
    metrics['mse'] = float(mse)
    
    # 2. SNR (Signal-to-Noise Ratio) in dB
    signal_power = np.mean(original_normalized ** 2)
    noise_power = np.mean((original_normalized - reconstructed_normalized) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    metrics['snr_db'] = float(snr)
    
    # 3. SDR (Signal-to-Distortion Ratio)
    sdr = calculate_sdr(original_normalized, reconstructed_normalized)
    if sdr is not None:
        metrics['sdr_db'] = sdr
    
    # 4. PESQ (Perceptual Evaluation of Speech Quality)
    try:
        if original_sr != 16000:
            original_16k = resample_audio(original, original_sr, 16000)
            reconstructed_16k = resample_audio(reconstructed, reconstructed_sr, 16000)
        else:
            original_16k = original
            reconstructed_16k = resample_audio(reconstructed, reconstructed_sr, 16000)
        
        min_len_16k = min(len(original_16k), len(reconstructed_16k))
        original_16k = original_16k[:min_len_16k]
        reconstructed_16k = reconstructed_16k[:min_len_16k]

        original_16k_norm = original_16k / (np.abs(original_16k).max() + 1e-8)
        reconstructed_16k_norm = reconstructed_16k / (np.abs(reconstructed_16k).max() + 1e-8)
        
        pesq_score = pesq(16000, original_16k_norm, reconstructed_16k_norm, 'wb')
        metrics['pesq'] = float(pesq_score)
    except Exception as e:
        print(f"PESQ calculation error: {e}")
        metrics['pesq'] = None
    
    # 5. STOI (Short-Time Objective Intelligibility)
    try:
        stoi_score = stoi(original_normalized, reconstructed_normalized, original_sr, extended=False)
        metrics['stoi'] = float(stoi_score)
        
        estoi_score = stoi(original_normalized, reconstructed_normalized, original_sr, extended=True)
        metrics['estoi'] = float(estoi_score)
    except Exception as e:
        print(f"STOI calculation error: {e}")
        metrics['stoi'] = None
        metrics['estoi'] = None
    
    return metrics

def process_language(language, tokenizer, dataset_name, cache_dir):
    """
    Process all samples for a given language and return summary statistics.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {language} (dataset: {dataset_name})")
    print(f"{'='*60}")
    
    # Special handling for NatureLM and GTZAN: load from single directory and filter by field
    if dataset_name == "naturelm":
        lang_dir = cache_dir / "naturelm"
    elif dataset_name == "gtzan":
        lang_dir = cache_dir / "gtzan"
    else:
        lang_dir = cache_dir / language
    
    if not lang_dir.exists():
        print(f"Dataset not found for {language} at {lang_dir}")
        return None
    
    # Load dataset
    try:
        ds = load_from_disk(lang_dir)
        
        # For NatureLM and GTZAN, filter by source field
        if dataset_name in ["naturelm", "gtzan"]:
            source_field = DATASETS[dataset_name].get("source_field", "source_dataset" if dataset_name == "naturelm" else "genre_name")
            # Filter dataset to only samples from this source/genre
            filtered_indices = [i for i in range(len(ds)) if ds[i][source_field] == language]
            if not filtered_indices:
                print(f"No samples found for {language}")
                return None
            # Create a filtered view
            ds = ds.select(filtered_indices)
        
        print(f"Loaded {len(ds)} samples for {language}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    all_metrics = []
    all_tokens_per_second = []
    all_compression_ratios = []
    successful = 0
    failed = 0
    
    for idx in tqdm(range(len(ds)), desc=f"Processing {language}"):
        try:
            sample = ds[idx]
            
            audio_array = sample['audio']['array']
            sr = sample['audio']['sampling_rate']
            duration = len(audio_array) / sr
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_array).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
            
            # Encode
            try:
                # Note: Don't move to device here - let the tokenizer handle device placement
                # Some tokenizers (like CosyVoice2 with ONNX) need CPU input and handle conversion internally
                tokens, encode_info = tokenizer.encode(audio_tensor, sr=sr)
            except Exception as e:
                print(f"\nError encoding sample {idx}: {e}")
                failed += 1
                continue
            
            # Decode
            try:
                reconstructed, decode_info = tokenizer.decode(tokens)
            except Exception as e:
                print(f"\nError decoding sample {idx}: {e}")
                failed += 1
                continue
            
            # Get output sample rate with fallback
            recon_sr = decode_info.get("output_sample_rate", tokenizer.output_sample_rate)
            if recon_sr is None:
                recon_sr = sr  # Fallback to input sample rate
            
            try:
                reconstructed_array = reconstructed.squeeze().cpu().numpy()
            except Exception as e:
                print(f"\nError extracting reconstructed audio for sample {idx}: {e}")
                failed += 1
                continue
            
            # Calculate metrics
            try:
                metrics = calculate_reconstruction_metrics(
                    audio_array,
                    reconstructed_array,
                    sr,
                    recon_sr
                )
            except Exception as e:
                print(f"\nError calculating metrics for sample {idx}: {e}")
                failed += 1
                continue
            
            # Calculate statistics
            try:
                num_tokens = tokens.numel()
                tokens_per_sec = num_tokens / duration
                
                original_size = len(audio_array) * 2  # 16-bit audio
                token_size = num_tokens * 2  # 16-bit tokens
                compression_ratio = original_size / token_size
            except Exception as e:
                print(f"\nError calculating statistics for sample {idx}: {e}")
                failed += 1
                continue
            
            all_metrics.append(metrics)
            all_tokens_per_second.append(tokens_per_sec)
            all_compression_ratios.append(compression_ratio)
            successful += 1
            
        except KeyboardInterrupt:
            print(f"\n\nInterrupted by user at sample {idx}")
            break
        except Exception as e:
            import traceback
            print(f"\nUnexpected error processing sample {idx}: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            failed += 1
            continue
                
    if not all_metrics:
        print(f"No samples processed successfully for {language}")
        if failed > 0:
            print(f"Failed: {failed} samples")
        return None
    
    if failed > 0:
        print(f"Successfully processed: {successful} samples, Failed: {failed} samples")
    
    def compute_stats(metric_name):
        values = [m[metric_name] for m in all_metrics if m.get(metric_name) is not None]
        if not values:
            return None
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values))
        }
    
    summary = {
        'language': language,
        'dataset': dataset_name,
        'num_samples': len(all_metrics),
        'num_failed': failed,
        'metrics': {
            'mse': compute_stats('mse'),
            'snr_db': compute_stats('snr_db'),
            'sdr_db': compute_stats('sdr_db'),
            'pesq': compute_stats('pesq'),
            'stoi': compute_stats('stoi'),
            'estoi': compute_stats('estoi')
        },
        'tokens_per_second': {
            'mean': float(np.mean(all_tokens_per_second)),
            'std': float(np.std(all_tokens_per_second))
        },
        'compression_ratio': {
            'mean': float(np.mean(all_compression_ratios)),
            'std': float(np.std(all_compression_ratios))
        }
    }
    
    return summary

def print_summary(summary):
    print(f"\n{'='*60}")
    print(f"Summary for {summary['language']} ({summary['dataset']})")
    print(f"{'='*60}")
    print(f"Samples processed: {summary['num_samples']}")
    print(f"\nReconstruction Metrics:")
    
    metrics = summary['metrics']
    
    if metrics['mse']:
        print(f"  MSE:         {metrics['mse']['mean']:.6f} ± {metrics['mse']['std']:.6f}")
    
    if metrics['snr_db']:
        print(f"  SNR (dB):    {metrics['snr_db']['mean']:.2f} ± {metrics['snr_db']['std']:.2f} "
              f"(min: {metrics['snr_db']['min']:.2f}, max: {metrics['snr_db']['max']:.2f})")
    
    if metrics['sdr_db']:
        print(f"  SDR (dB):    {metrics['sdr_db']['mean']:.2f} ± {metrics['sdr_db']['std']:.2f} "
              f"(min: {metrics['sdr_db']['min']:.2f}, max: {metrics['sdr_db']['max']:.2f})")
    
    if metrics['pesq']:
        print(f"  PESQ:        {metrics['pesq']['mean']:.3f} ± {metrics['pesq']['std']:.3f} "
              f"(min: {metrics['pesq']['min']:.3f}, max: {metrics['pesq']['max']:.3f})")
    
    if metrics['stoi']:
        print(f"  STOI:        {metrics['stoi']['mean']:.4f} ± {metrics['stoi']['std']:.4f} "
              f"(min: {metrics['stoi']['min']:.4f}, max: {metrics['stoi']['max']:.4f})")
    
    if metrics['estoi']:
        print(f"  ESTOI:       {metrics['estoi']['mean']:.4f} ± {metrics['estoi']['std']:.4f} "
              f"(min: {metrics['estoi']['min']:.4f}, max: {metrics['estoi']['max']:.4f})")
    
    print(f"\nTokenization Statistics:")
    print(f"  Tokens/sec:        {summary['tokens_per_second']['mean']:.1f} ± {summary['tokens_per_second']['std']:.1f}")
    print(f"  Compression ratio: {summary['compression_ratio']['mean']:.1f}x ± {summary['compression_ratio']['std']:.1f}x")

def main():
    parser = argparse.ArgumentParser(description='Evaluate audio tokenizer on multiple datasets')
    parser.add_argument('--tokenizer', type=str, default='neucodec',
                        help='Tokenizer name registered in audio_tokenizers (e.g., neucodec, xcodec2)')
    parser.add_argument('--dataset', type=str, choices=['eurospeech', 'fleurs', 'naturelm', 'gtzan', 'all'], default=None,
                        help='Dataset to evaluate: eurospeech, fleurs, naturelm, gtzan, or all. If not specified with --language, evaluates all datasets.')
    parser.add_argument('--language', type=str, default=None,
                        help='Specific language to evaluate (e.g., germany, en_us). Auto-detects dataset.')
    parser.add_argument('--languages', nargs='+', default=None,
                        help='Multiple languages to evaluate (e.g., --languages germany en_us). Auto-detects datasets.')
    args = parser.parse_args()
    
    languages_to_evaluate = []
    
    if args.language:
        languages_to_evaluate = [args.language]
    elif args.languages:
        languages_to_evaluate = args.languages
    elif args.dataset:
        if args.dataset == 'all':
            for dataset_config in DATASETS.values():
                languages_to_evaluate.extend(dataset_config['languages'])
        else:
            languages_to_evaluate = DATASETS[args.dataset]['languages']
    else:
        for dataset_config in DATASETS.values():
            languages_to_evaluate.extend(dataset_config['languages'])
    
    print("="*60)
    print(f"Tokenizer Evaluation: {args.tokenizer}")
    print("="*60)
    print(f"Total languages to evaluate: {len(languages_to_evaluate)}")
    if len(languages_to_evaluate) <= 10:
        print(f"Languages: {', '.join(languages_to_evaluate)}")
    else:
        print(f"Languages: {', '.join(languages_to_evaluate[:10])}... (+{len(languages_to_evaluate)-10} more)")
    
    print("\nLoading tokenizer ...")
    try:
        tokenizer = get_tokenizer(args.tokenizer, device=device)
        print(f"✓ Tokenizer loaded successfully")
        if hasattr(tokenizer, 'sample_rate'):
            print(f"  Input SR: {tokenizer.sample_rate} Hz")
        print(f"  Output SR: {tokenizer.output_sample_rate} Hz")
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        return
    
    all_results = []
    failed_languages = []
    total_failed_samples = 0
    
    for language in languages_to_evaluate:
        dataset_name, cache_dir, _ = get_dataset_for_language(language)
        
        if dataset_name is None:
            print(f"Warning: Language '{language}' not found in any dataset. Skipping.")
            failed_languages.append(language)
            continue
        
        summary = process_language(language, tokenizer, dataset_name, cache_dir)
        if summary:
            all_results.append(summary)
            total_failed_samples += summary.get('num_failed', 0)
            print_summary(summary)
            
            output_file = RESULTS_DIR / f"{args.tokenizer}_{dataset_name}_{language}_results.json"
            with output_file.open('w') as f:
                json.dump(summary, f, indent=2)
            print(f"\nResults saved to: {output_file}")
        else:
            failed_languages.append(language)
    
    if all_results:
        print(f"\n{'='*60}")
        print(f"Successfully evaluated: {len(all_results)}/{len(languages_to_evaluate)} languages")
        if failed_languages:
            print(f"Failed languages: {', '.join(failed_languages)}")
        if total_failed_samples > 0:
            print(f"Total failed samples: {total_failed_samples}")
        print(f"Evaluation complete!")
    else:
        print("No languages were successfully evaluated.")
    
    # Exit with non-zero code if there were any failed samples
    if total_failed_samples > 0 or len(failed_languages) > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()