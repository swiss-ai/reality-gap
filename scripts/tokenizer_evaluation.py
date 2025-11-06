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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from audio_tokenizers import get_tokenizer

from pesq import pesq
from pystoi import stoi
import museval

# Base directories
SCRATCH_DIR = os.path.expandvars("$SCRATCH/benchmark-audio-tokenizer/datasets")
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
RESULTS_DIR = os.path.join(project_root, "metrics")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Dataset configurations
DATASETS = {
    'eurospeech': {
        'cache_dir': os.path.join(SCRATCH_DIR, "eurospeech_cache"),
        'languages': [
            "bosnia-herzegovina", "bulgaria", "croatia", "denmark", "estonia", "finland",
            "france", "germany", "greece", "iceland", "italy", "latvia", "lithuania",
            "malta", "norway", "portugal", "serbia", "slovakia", "slovenia",
            "sweden", "uk", "ukraine"
        ]
    },
    'fleurs': {
        'cache_dir': os.path.join(SCRATCH_DIR, "fleurs_cache"),
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
    }
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
    
    lang_dir = os.path.join(cache_dir, language)
    if not os.path.exists(lang_dir):
        print(f"Dataset not found for {language} at {lang_dir}")
        return None
    
    ds = load_from_disk(lang_dir)
    print(f"Loaded {len(ds)} samples for {language}")
    
    all_metrics = []
    all_tokens_per_second = []
    all_compression_ratios = []
    
    for idx in tqdm(range(len(ds)), desc=f"Processing {language}"):
        sample = ds[idx]
        
        audio_array = sample['audio']['array']
        sr = sample['audio']['sampling_rate']
        duration = len(audio_array) / sr
        
        audio_tensor = torch.from_numpy(audio_array).float()
        
        try:
            tokens, encode_info = tokenizer.encode(audio_tensor, sr=sr)
            reconstructed, decode_info = tokenizer.decode(tokens)
            
            reconstructed_array = reconstructed.squeeze().cpu().numpy()
            token_values = tokens.squeeze().cpu().numpy()
            
            metrics = calculate_reconstruction_metrics(
                audio_array,
                reconstructed_array,
                sr,
                decode_info['output_sample_rate']
            )
            
            num_tokens = tokens.numel()
            tokens_per_sec = num_tokens / duration
            
            original_size = len(audio_array) * 2  # 16-bit audio
            token_size = num_tokens * 2  # 16-bit tokens
            compression_ratio = original_size / token_size
            
            all_metrics.append(metrics)
            all_tokens_per_second.append(tokens_per_sec)
            all_compression_ratios.append(compression_ratio)
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    if not all_metrics:
        print(f"No samples processed successfully for {language}")
        return None
    
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
    parser.add_argument('--dataset', type=str, choices=['eurospeech', 'fleurs', 'all'], default=None,
                        help='Dataset to evaluate: eurospeech, fleurs, or all. If not specified with --language, evaluates all datasets.')
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
    tokenizer = get_tokenizer(args.tokenizer, device=device)
    print(f"Tokenizer loaded: {tokenizer}")
    
    all_results = []
    failed_languages = []
    
    for language in languages_to_evaluate:
        dataset_name, cache_dir, _ = get_dataset_for_language(language)
        
        if dataset_name is None:
            print(f"Warning: Language '{language}' not found in any dataset. Skipping.")
            failed_languages.append(language)
            continue
        
        summary = process_language(language, tokenizer, dataset_name, cache_dir)
        if summary:
            all_results.append(summary)
            print_summary(summary)
            
            output_file = os.path.join(RESULTS_DIR, f"{args.tokenizer}_{dataset_name}_{language}_results.json")
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\nResults saved to: {output_file}")
        else:
            failed_languages.append(language)
    
    if all_results:
        combined_output = os.path.join(RESULTS_DIR, f"{args.tokenizer}_all_results.json")
        with open(combined_output, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        for dataset_name in DATASETS.keys():
            dataset_results = [r for r in all_results if r['dataset'] == dataset_name]
            if dataset_results:
                dataset_output = os.path.join(RESULTS_DIR, f"{args.tokenizer}_{dataset_name}_summary.json")
                with open(dataset_output, 'w') as f:
                    json.dump(dataset_results, f, indent=2)
                print(f"\n{dataset_name.upper()} results saved to: {dataset_output}")
        
        print(f"\n{'='*60}")
        print(f"Combined results saved to: {combined_output}")
        print(f"Successfully evaluated: {len(all_results)}/{len(languages_to_evaluate)} languages")
        if failed_languages:
            print(f"Failed languages: {', '.join(failed_languages)}")
        print(f"Evaluation complete!")
    else:
        print("No languages were successfully evaluated.")

if __name__ == "__main__":
    main()