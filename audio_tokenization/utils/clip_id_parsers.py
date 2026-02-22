"""Registry of per-dataset clip ID parsers for extracting (source_id, clip_num).

Used by the audio_text_interleaving pipeline to group clips from the same
source recording and sort them in order.

Examples:
    Emilia:  ``EN_tKvmUvxYZXI_W000006`` -> ``("EN_tKvmUvxYZXI", 6)``
    People's Speech: ``...forum_SLASH_..._00002.flac`` -> ``("..._DOT_mp3", 2)``
"""

import re
from typing import Tuple


def parse_emilia_clip_id(clip_id: str) -> Tuple[str, int]:
    """Parse Emilia-style clip IDs.

    Format: ``{lang}_{youtube_id}_W{clip_num:06d}``
    e.g. ``EN_tKvmUvxYZXI_W000006`` -> ``("EN_tKvmUvxYZXI", 6)``
    """
    match = re.match(r"^(.+)_W(\d+)$", clip_id)
    if match is None:
        raise ValueError(f"Cannot parse Emilia clip ID: {clip_id!r}")
    source_id = match.group(1)
    clip_num = int(match.group(2))
    return source_id, clip_num


def parse_peoples_speech_clip_id(clip_id: str) -> Tuple[str, int]:
    """Parse People's Speech clip IDs.

    Format: ``{source_path_with_SLASH_DOT}_NNNNN.flac``
    e.g. ``forum_SLASH_foo_DOT_mp3_00002.flac`` -> ``("forum_SLASH_foo_DOT_mp3", 2)``
    """
    match = re.match(r"^(.+?)_(\d+)(?:\.\w+)?$", clip_id)
    if match is None:
        raise ValueError(f"Cannot parse People's Speech clip ID: {clip_id!r}")
    source_id = match.group(1)
    clip_num = int(match.group(2))
    return source_id, clip_num


def parse_generic_clip_id(clip_id: str) -> Tuple[str, int]:
    """Fallback parser: treats entire clip ID as source, clip_num=0."""
    return clip_id, 0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_PARSERS = {
    "emilia": parse_emilia_clip_id,
    "peoples_speech": parse_peoples_speech_clip_id,
    "generic": parse_generic_clip_id,
}


def get_clip_id_parser(name: str):
    """Look up a clip ID parser by name.

    Args:
        name: One of ``"emilia"``, ``"peoples_speech"``, ``"generic"``.

    Returns:
        Callable[[str], Tuple[str, int]]
    """
    if name not in _PARSERS:
        raise ValueError(
            f"Unknown clip_id_parser: {name!r}. "
            f"Available: {sorted(_PARSERS.keys())}"
        )
    return _PARSERS[name]
