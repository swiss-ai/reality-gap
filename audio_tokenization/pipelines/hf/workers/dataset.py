"""Dataset loading helpers for HF workers."""

from typing import Any, Dict, Optional

from datasets import Audio, load_dataset


def load_dataset_for_worker(
    dataset_info: Dict[str, Any],
    audio_field: str,
    target_sample_rate: Optional[int],
    logger=None,
):
    """Load and prepare a HF dataset for a worker."""
    split = dataset_info["split"]
    max_samples = dataset_info.get("max_samples")
    if max_samples:
        split = f"{split}[:{max_samples}]"

    dataset = load_dataset(
        dataset_info["name"],
        name=dataset_info.get("config"),
        split=split,
        cache_dir=dataset_info.get("cache_dir"),
    )

    filtered_indices = dataset_info.get("filtered_indices")
    if filtered_indices is not None:
        if logger is not None:
            logger.info(
                f"Worker: Applying index filter ({len(filtered_indices)} samples)"
            )
        dataset = dataset.select(filtered_indices)

    if target_sample_rate:
        dataset = dataset.cast_column(audio_field, Audio(sampling_rate=target_sample_rate))

    return dataset
