"""Audio structure tokens."""

# Audio structure tokens (following vision token convention)
# Vision uses RESERVED_OMNI_001-007, audio uses 008-012
AUDIO_STRUCTURE_TOKENS = [
    ("<|RESERVED_OMNI_008|>", "<|audio_start|>"),
    ("<|RESERVED_OMNI_009|>", "<|audio_end|>"),
    ("<|RESERVED_OMNI_010|>", "<|speech_transcribe|>"),
    ("<|RESERVED_OMNI_011|>", "<|speech_switch|>"),
    ("<|RESERVED_OMNI_012|>", "<|audio_annotate|>"),
]
