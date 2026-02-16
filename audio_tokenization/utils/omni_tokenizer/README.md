# Audio Omni-Tokenizer

Add audio tokens to omni-tokenizers with automatic codebook size detection.

## Quick Start

```bash
python -m audio_tokenization.utils.omni_tokenizer.add_audio_tokens \
    --input-tokenizer /path/to/vision_omni_tokenizer \
    --output-tokenizer /path/to/audio_omni_tokenizer \
    --audio-tokenizer-type wavtokenizer
```

## Design

The audio omni-tokenizer extends existing omni-tokenizers (with vision tokens) by adding audio capabilities while maintaining the reserved token structure.

**Token Layout (example with Emu3.5 + WavTokenizer):**

```
ID Range        | Tokens
----------------|------------------------------------------
0-131071        | Base text vocab (131,072)
131072          | RESERVED_OMNI_000 (boundary marker)
131073-131079   | Vision structure (img_start, img_end, etc.)
131080-131081   | Audio structure (audio_start, audio_end)
131082-131271   | RESERVED_OMNI_010-199 (future modalities)
131272-262343   | Vision content tokens (131,072)
262344-266439   | Audio content tokens (4,096)
```

**Audio Structure Tokens (renamed from RESERVED_OMNI):**
- `<|RESERVED_OMNI_008|>` → `<|audio_start|>` - Audio sequence start
- `<|RESERVED_OMNI_009|>` → `<|audio_end|>` - Audio sequence end

**Audio Content Tokens:**
- Format: `<|audio token N|>` where N is 0 to codebook_size-1
- WavTokenizer: 4,096 tokens (`<|audio token 0|>` to `<|audio token 4095|>`)

## Supported Audio Tokenizers

| Tokenizer | Codebook Size | Tokens/Second |
|-----------|---------------|---------------|
| wavtokenizer (wavtokenizer-40) | 4,096 | 40 |

## Token ID Conversion

Audio content tokens are contiguous, so conversion is simple:

```python
# Audio tokenizer produces indices 0-4095
# Convert to vocab token IDs:
audio_token_offset = tokenizer.convert_tokens_to_ids("<|audio token 0|>")
token_id = audio_token_offset + audio_index
```

## Multi-Modality Support

This module handles tokenizers with existing modalities:

1. **Detects existing modalities** via mapping files (vision_token_mapping.json, etc.)
2. **Preserves existing tokens** - won't re-add RESERVED_OMNI slots used by other modalities
3. **Copies modality mappings** to output directory

**Reserved Token Allocation:**
- 001-007: Vision structure tokens
- 008-009: Audio structure tokens
- 010-199: Future modalities

## Idempotency

Running `add_audio_tokens` on a tokenizer that already has audio tokens will:
1. Validate the existing audio_vocab_size matches the requested one
2. Skip token addition if already present
3. Raise an error if parameters don't match

## Output Files

- Standard tokenizer files (tokenizer.json, tokenizer_config.json, etc.)
- `audio_token_mapping.json` - Audio token metadata
- Copied modality mappings (e.g., vision_token_mapping.json)

## Python API

```python
from audio_tokenization.utils.omni_tokenizer import add_audio_tokens

tokenizer, stats = add_audio_tokens(
    input_tokenizer_path="/path/to/vision_omni_tokenizer",
    output_path="/path/to/output",
    audio_tokenizer_name="wavtokenizer",
)
```
