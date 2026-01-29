"""Audio structure tokens."""

# Audio structure tokens (following vision token convention)
# Vision uses RESERVED_OMNI_001-007, audio uses 008-009
AUDIO_STRUCTURE_TOKENS = [
    ("<|RESERVED_OMNI_008|>", "<|audio_start|>"),
    ("<|RESERVED_OMNI_009|>", "<|audio_end|>"),
]
