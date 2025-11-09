import os
from datasets import load_dataset, Dataset, Audio, Features, Value
import json
import time
import argparse

# scratch_dir = os.path.expandvars("$SCRATCH/benchmark-audio-tokenizer/datasets")
# cache_dir = os.path.join(scratch_dir, "fleurs_cache")
# Modified to use capstor for xyixuan
cache_dir = "/capstor/store/cscs/swissai/infra01/audio-datasets/fleurs_cache"
os.makedirs(cache_dir, exist_ok=True)

# FLEURS language codes - manually curate this list as needed
# Full list of 102 languages available at: https://huggingface.co/datasets/google/fleurs
# Format: language_country (e.g., "af_za" for Afrikaans-South Africa)

languages = [
    # Western Europe
     "ast_es",      # Asturian (Spain)
     "ca_es",       # Catalan (Spain)
     "nl_nl",       # Dutch (Netherlands)
     "en_us",       # English (US)
     "gl_es",       # Galician (Spain)
     "hu_hu",       # Hungarian (Hungary)
     "ga_ie",       # Irish (Ireland)
     "kea_cv",      # Kabuverdianu (Cape Verde)
     "lb_lu",       # Luxembourgish (Luxembourg)
     "oc_fr",       # Occitan (France)
     "es_419",      # Spanish (Latin America)
     "cy_gb",       # Welsh (UK)
    
    # Eastern Europe
     "hy_am",       # Armenian (Armenia)
     "be_by",       # Belarusian (Belarus)
     "cs_cz",       # Czech (Czechia)
     "ka_ge",       # Georgian (Georgia)
     "mk_mk",       # Macedonian (North Macedonia)
     "pl_pl",       # Polish (Poland)
     "ro_ro",       # Romanian (Romania)
     "ru_ru",       # Russian (Russia)

    # Asia - CJK
    "cmn_hans_cn", # Chinese Mandarin Simplified (China) 
    "yue_hant_hk", # Chinese Cantonese Traditional (Hong Kong) 
    "ja_jp",       # Japanese (Japan)
    "ko_kr",       # Korean (South Korea)
    
    # Asia - South Asia
    "hi_in",       # Hindi (India)
    "bn_in",       # Bengali (India)
    "ta_in",       # Tamil (India)
    "te_in",       # Telugu (India)
    
    # Asia - Southeast Asia
    "th_th",       # Thai (Thailand)
    "vi_vn",       # Vietnamese (Vietnam)
    "id_id",       # Indonesian (Indonesia)
    
    # Africa
    "af_za",       # Afrikaans (South Africa)
    "sw_ke",       # Swahili (Kenya)
    "am_et",       # Amharic (Ethiopia)
    "yo_ng",       # Yoruba (Nigeria)
    
    # Middle East
    "ar_eg",       # Arabic (Egypt)
    "tr_tr",       # Turkish (Turkey)
    "he_il",       # Hebrew (Israel)
    "fa_ir",       # Persian/Farsi (Iran)
]

'''
# languages = [
     # Western Europe
     "ast_es",      # Asturian (Spain)
     "bs_ba",       # Bosnian (Bosnia)
     "ca_es",       # Catalan (Spain)
     "hr_hr",       # Croatian (Croatia)
     "da_dk",       # Danish (Denmark)
     "nl_nl",       # Dutch (Netherlands)
     "en_us",       # English (US)
     "fi_fi",       # Finnish (Finland)
     "fr_fr",       # French (France)
     "gl_es",       # Galician (Spain)
     "de_de",       # German (Germany)
     "el_gr",       # Greek (Greece)
     "hu_hu",       # Hungarian (Hungary)
     "is_is",       # Icelandic (Iceland)
     "ga_ie",       # Irish (Ireland)
     "it_it",       # Italian (Italy)
     "kea_cv",      # Kabuverdianu (Cape Verde)
     "lb_lu",       # Luxembourgish (Luxembourg)
     "mt_mt",       # Maltese (Malta)
     "nb_no",       # Norwegian Bokmål (Norway)
     "oc_fr",       # Occitan (France)
     "pt_br",       # Portuguese (Brazil)
     "es_419",      # Spanish (Latin America)
     "sv_se",       # Swedish (Sweden)
     "cy_gb",       # Welsh (UK)
     
     # Eastern Europe
     "hy_am",       # Armenian (Armenia)
     "be_by",       # Belarusian (Belarus)
     "bg_bg",       # Bulgarian (Bulgaria)
     "cs_cz",       # Czech (Czechia)
     "et_ee",       # Estonian (Estonia)
     "ka_ge",       # Georgian (Georgia)
     "lv_lv",       # Latvian (Latvia)
     "lt_lt",       # Lithuanian (Lithuania)
     "mk_mk",       # Macedonian (North Macedonia)
     "pl_pl",       # Polish (Poland)
     "ro_ro",       # Romanian (Romania)
     "ru_ru",       # Russian (Russia)
     "sr_rs",       # Serbian (Serbia)
     "sk_sk",       # Slovak (Slovakia)
     "sl_si",       # Slovenian (Slovenia)
     "uk_ua",       # Ukrainian (Ukraine)
     
     # Central Asia/Middle East/North Africa
     "ar_eg",       # Arabic (Egypt)
     "az_az",       # Azerbaijani (Azerbaijan)
     "he_il",       # Hebrew (Israel)
     "kk_kz",       # Kazakh (Kazakhstan)
     "ky_kg",       # Kyrgyz (Kyrgyzstan)
     "mn_mn",       # Mongolian (Mongolia)
     "ps_af",       # Pashto (Afghanistan)
     "fa_ir",       # Persian/Farsi (Iran)
     "ckb_iq",      # Sorani Kurdish (Iraq)
     "tg_tj",       # Tajik (Tajikistan)
     "tr_tr",       # Turkish (Turkey)
     "uz_uz",       # Uzbek (Uzbekistan)
     
     # Sub-Saharan Africa
     "af_za",       # Afrikaans (South Africa)
     "am_et",       # Amharic (Ethiopia)
     "ff_sn",       # Fula (Senegal)
     "lg_ug",       # Ganda/Luganda (Uganda)
     "ha_ng",       # Hausa (Nigeria)
     "ig_ng",       # Igbo (Nigeria)
     "kam_ke",      # Kamba (Kenya)
     "ln_cd",       # Lingala (Congo)
     "luo_ke",      # Luo (Kenya)
     "nso_za",      # Northern Sotho (South Africa)
     "ny_mw",       # Nyanja/Chichewa (Malawi)
     "om_et",       # Oromo (Ethiopia)
     "sn_zw",       # Shona (Zimbabwe)
     "so_so",       # Somali (Somalia)
     "sw_ke",       # Swahili (Kenya)
     "umb_ao",      # Umbundu (Angola)
     "wo_sn",       # Wolof (Senegal)
     "xh_za",       # Xhosa (South Africa)
     "yo_ng",       # Yoruba (Nigeria)
     "zu_za",       # Zulu (South Africa)
     
     # South Asia
     "as_in",       # Assamese (India)
     "bn_in",       # Bengali (India)
     "gu_in",       # Gujarati (India)
     "hi_in",       # Hindi (India)
     "kn_in",       # Kannada (India)
     "ml_in",       # Malayalam (India)
     "mr_in",       # Marathi (India)
     "ne_np",       # Nepali (Nepal)
     "or_in",       # Oriya (India)
     "pa_in",       # Punjabi (India)
     "sd_in",       # Sindhi (India)
     "ta_in",       # Tamil (India)
     "te_in",       # Telugu (India)
     "ur_pk",       # Urdu (Pakistan)
     
     # South-East Asia
     "my_mm",       # Burmese (Myanmar)
     "ceb_ph",      # Cebuano (Philippines)
     "fil_ph",      # Filipino (Philippines)
     "id_id",       # Indonesian (Indonesia)
     "jv_id",       # Javanese (Indonesia)
     "km_kh",       # Khmer (Cambodia)
     "lo_la",       # Lao (Laos)
     "ms_my",       # Malay (Malaysia)
     "mi_nz",       # Maori (New Zealand)
     "th_th",       # Thai (Thailand)
     "vi_vn",       # Vietnamese (Vietnam)
     
     # CJK
     "yue_hant_hk", # Cantonese Traditional (Hong Kong)
     "cmn_hans_cn", # Mandarin Simplified (China)
     "ja_jp",       # Japanese (Japan)
     "ko_kr",       # Korean (South Korea)
 ]
'''




def download_language(lang, n=100, split="train"):
    """
    Download a specific language from FLEURS dataset.
    
    Args:
        lang: Language code (e.g., "af_za", "hi_in")
        n: Maximum number of samples to download (default: 100)
        split: Dataset split to use - "train", "validation", or "test" (default: "train")
    
    Note: FLEURS has approximately:
        - train: ~1000 samples
        - validation: ~400 samples  
        - test: ~400 samples
    """
    print(f"\n{'='*60}")
    print(f"Downloading {lang} ({split}): max {n} samples")
    print(f"{'='*60}")
    
    out_dir = os.path.join(cache_dir, lang)
    if os.path.exists(os.path.join(out_dir, "summary.json")):
        print(f"Already exists, skipping...")
        return 0
    
    try:
        # Load without streaming to avoid torchcodec issues and remove deprecated trust_remote_code
        # Use num_proc for parallel downloading
        # Use split slicing to only load n samples (e.g., "train[:100]")
        dataset = load_dataset("google/fleurs", lang, split=f"{split}[:{n}]", num_proc=32)

        # Use map with batch processing for better performance
        def process_batch(examples, indices):
            """Process a batch of examples efficiently"""
            batch_size = len(examples["audio"])
            processed = {
                "audio": examples["audio"],
                "text": examples.get("transcription", [""] * batch_size),
                "raw_text": examples.get("raw_transcription", [""] * batch_size),
                "language": [lang] * batch_size,
                "lang_id": examples.get("lang_id", [-1] * batch_size),
                "lang_group_id": examples.get("lang_group_id", [-1] * batch_size),
                "gender": examples.get("gender", [-1] * batch_size),
                "sample_id": indices
            }
            return processed

        # Apply batch processing with multiple workers
        dataset = dataset.map(
            process_batch,
            batched=True,
            batch_size=100,
            num_proc=4,  # Use 4 processes for mapping
            with_indices=True,
            desc=f"Processing {lang}"
        )

        # Convert to list format for compatibility
        samples = []
        for i, sample in enumerate(dataset):
            audio_data = sample["audio"]
            samples.append({
                "audio": {
                    "array": audio_data["array"],
                    "sampling_rate": audio_data["sampling_rate"],
                    "path": audio_data.get("path", f"{lang}_{i}")
                },
                "text": sample["text"],
                "raw_text": sample["raw_text"],
                "language": sample["language"],
                "lang_id": sample["lang_id"],
                "lang_group_id": sample["lang_group_id"],
                "gender": sample["gender"],
                "sample_id": sample["sample_id"]
            })
        
        if not samples:
            print(f"No samples downloaded for {lang}")
            return 0
        
        print(f"Saving {len(samples)} samples...")
        
        features = Features({
            'audio': Audio(sampling_rate=samples[0]['audio']['sampling_rate']),
            'text': Value('string'),
            'raw_text': Value('string'),
            'language': Value('string'),
            'lang_id': Value('int64'),
            'lang_group_id': Value('int64'),
            'gender': Value('int64'),
            'sample_id': Value('int64')
        })
        
        ds = Dataset.from_list(samples, features=features)
        ds.save_to_disk(out_dir)
        
        total_dur = sum(len(s["audio"]["array"]) / s["audio"]["sampling_rate"] for s in samples)
        summary = {
            "language": lang,
            "num_samples": len(samples),
            "sampling_rate": samples[0]["audio"]["sampling_rate"],
            "total_duration_sec": total_dur,
            "total_duration_min": total_dur / 60,
            "split": split
        }
        
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"Saved {len(samples)} samples ({total_dur/60:.1f} min)")
        return len(samples)
    
    except Exception as e:
        print(f"Error downloading {lang}: {e}")
        return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download FLEURS dataset languages')
    parser.add_argument('--language', type=str, default=None,
                        help='Specific language to download (e.g., en_us). If not specified, downloads all languages in the list.')
    parser.add_argument('--languages', nargs='+', default=None,
                        help='Multiple languages to download (e.g., --languages en_us de_de fr_fr)')
    args = parser.parse_args()
    
    # Determine which languages to download
    if args.language:
        languages_to_download = [args.language]
    elif args.languages:
        languages_to_download = args.languages
    else:
        languages_to_download = languages
    
    print("="*60)
    print("FLEURS Dataset Download Script")
    print("="*60)
    print(f"Languages to download: {len(languages_to_download)}")
    print(f"Max samples per language: 100")
    if len(languages_to_download) <= 5:
        print(f"Languages: {', '.join(languages_to_download)}")
    else:
        print(f"Languages: {', '.join(languages_to_download[:5])}... (+{len(languages_to_download)-5} more)")
    print("="*60)
    
    total = 0
    successful = 0
    failed = []
    
    for lang in languages_to_download:
        start_time = time.time()
        result = download_language(lang, n=100, split="train")
        overall_elapsed = time.time() - start_time
        print(f"\nExecution time: {overall_elapsed/60:.2f} min ({overall_elapsed:.1f} sec)")
        total += result
        if result > 0:
            successful += 1
        else:
            failed.append(lang)
    
    print(f"\n{'='*60}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"Total samples downloaded: {total}")
    print(f"Successful languages: {successful}/{len(languages_to_download)}")
    if failed:
        print(f"Failed languages: {', '.join(failed)}")
    print(f"Data saved to: {cache_dir}")
    print(f"{'='*60}")
    
 