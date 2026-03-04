#!/usr/bin/env python3
"""
Audio continuation demo: long clips (60-120s), 40s prompt.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

import torch
import torchaudio
import soundfile as sf

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from audio_tokenizers.implementations.wavtokenizer import WavTokenizer40
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_continuation(
    model, input_ids, max_new_tokens, temperature, top_p, device,
    audio_end_id, audio_token_offset,
):
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        output = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
    generated_ids = output[0, len(input_ids):].tolist()
    audio_range = range(audio_token_offset, audio_token_offset + 4096)
    gen_audio_ids = []
    for tid in generated_ids:
        if tid == audio_end_id:
            break
        if tid in audio_range:
            gen_audio_ids.append(tid - audio_token_offset)
    return gen_audio_ids


def main():
    model_path = "/capstor/store/cscs/swissai/infra01/MLLM/apertus-8b/HF_CKPT_96000_nemo"
    tokenizer_path = "/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok"
    manifest_path = "/tmp/long_en_librilight/manifest.json"
    device = "cuda"
    max_new_tokens = 800   # 20s worth
    temperature = 0.5
    top_p = 0.95
    output_dir = f"/capstor/store/cscs/swissai/infra01/audio-datasets/audio_completion_samples/long_en_librilight_t{temperature}_p{top_p}"
    prompt_tokens_count = 1600  # 40s at 40 tok/s
    prompt_seconds = 40.0

    os.makedirs(output_dir, exist_ok=True)

    samples = json.load(open(manifest_path))
    print(f"Loaded {len(samples)} samples")

    print("Loading WavTokenizer...")
    wav_tok = WavTokenizer40(device=device, torch_compile=False)

    print("Loading omni tokenizer...")
    omni_tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    audio_start_id = omni_tok.convert_tokens_to_ids("<|audio_start|>")
    audio_end_id = omni_tok.convert_tokens_to_ids("<|audio_end|>")
    audio_token_offset = omni_tok.convert_tokens_to_ids("<|audio token 0|>")
    bos_id = omni_tok.bos_token_id

    print(f"Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    results = []

    for i, sample in enumerate(samples):
        audio_path = sample["path"]
        cid = sample["id"]
        safe_name = cid.replace("/", "_").replace(" ", "_").replace(".", "_")

        if not os.path.exists(audio_path):
            print(f"  [{i+1}/{len(samples)}] SKIP: {audio_path} not found")
            continue

        audio, sr = sf.read(audio_path)
        total_dur = len(audio) / sr
        print(f"\n  [{i+1}/{len(samples)}] {cid}: {total_dur:.1f}s (sr={sr})", flush=True)

        # Resample to 24kHz if needed (WavTokenizer expects 24kHz)
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        if sr != 24000:
            resampler = torchaudio.transforms.Resample(sr, 24000)
            audio_tensor = resampler(audio_tensor)
            print(f"    Resampled {sr} -> 24000", flush=True)

        audio_tensor = audio_tensor.to(device)
        tokens_full = wav_tok.encode_audio(audio_tensor)
        print(f"    Full tokens: {tokens_full.shape[1]}", flush=True)

        # Mode A: token prompt (first 1600 tokens from full encoding)
        tokens_a = tokens_full[0, :prompt_tokens_count]
        prompt_ids_a = (tokens_a + audio_token_offset).tolist()
        input_ids_a = [bos_id, audio_start_id] + prompt_ids_a
        gen_a = generate_continuation(
            model, input_ids_a, max_new_tokens, temperature, top_p, device,
            audio_end_id, audio_token_offset,
        )
        a_dur = len(gen_a) / 40

        # Mode B: audio prompt (truncate to 40s in original sr, then resample + encode)
        prompt_samples = int(prompt_seconds * sr)
        audio_prompt = audio[:prompt_samples]
        audio_tensor_prompt = torch.from_numpy(audio_prompt).float().unsqueeze(0)
        if sr != 24000:
            audio_tensor_prompt = resampler(audio_tensor_prompt)
        audio_tensor_prompt = audio_tensor_prompt.to(device)
        tokens_b = wav_tok.encode_audio(audio_tensor_prompt)
        prompt_ids_b = (tokens_b.squeeze(0) + audio_token_offset).tolist()
        input_ids_b = [bos_id, audio_start_id] + prompt_ids_b
        gen_b = generate_continuation(
            model, input_ids_b, max_new_tokens, temperature, top_p, device,
            audio_end_id, audio_token_offset,
        )
        b_dur = len(gen_b) / 40

        # Save WAVs
        sample_dir = f"{output_dir}/{safe_name}"
        os.makedirs(sample_dir, exist_ok=True)
        sf.write(f"{sample_dir}/input_full.wav", audio, sr)

        if gen_a:
            prompt_audio_a, _ = wav_tok.decode(tokens_a.unsqueeze(0))
            prompt_audio_a_np = prompt_audio_a.squeeze().cpu().numpy()
            gen_audio_a, _ = wav_tok.decode(torch.tensor([gen_a], dtype=torch.long, device=device))
            gen_audio_a_np = gen_audio_a.squeeze().cpu().numpy()
            sf.write(f"{sample_dir}/A_generated.wav", gen_audio_a_np, 24000)
            sf.write(f"{sample_dir}/A_combined.wav", np.concatenate([prompt_audio_a_np, gen_audio_a_np]), 24000)

        if gen_b:
            prompt_decoded_b, _ = wav_tok.decode(tokens_b)
            prompt_decoded_b_np = prompt_decoded_b.squeeze().cpu().numpy()
            gen_audio_b, _ = wav_tok.decode(torch.tensor([gen_b], dtype=torch.long, device=device))
            gen_audio_b_np = gen_audio_b.squeeze().cpu().numpy()
            sf.write(f"{sample_dir}/B_generated.wav", gen_audio_b_np, 24000)
            sf.write(f"{sample_dir}/B_combined.wav", np.concatenate([prompt_decoded_b_np, gen_audio_b_np]), 24000)

        result = {
            "id": cid, "total_dur": round(total_dur, 2),
            "full_tokens": tokens_full.shape[1],
            "A_generated_tokens": len(gen_a), "A_generated_dur": round(a_dur, 2),
            "B_generated_tokens": len(gen_b), "B_generated_dur": round(b_dur, 2),
        }
        results.append(result)
        print(f"    A={a_dur:.1f}s, B={b_dur:.1f}s", flush=True)

    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)

    a_durs = [r["A_generated_dur"] for r in results]
    b_durs = [r["B_generated_dur"] for r in results]
    print(f"\n{'='*60}")
    print(f"Results ({len(results)} samples):")
    print(f"  Mode A: {sum(1 for d in a_durs if d>0)}/{len(results)} generated, avg={np.mean(a_durs):.1f}s")
    print(f"  Mode B: {sum(1 for d in b_durs if d>0)}/{len(results)} generated, avg={np.mean(b_durs):.1f}s")
    print(f"  Output: {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
