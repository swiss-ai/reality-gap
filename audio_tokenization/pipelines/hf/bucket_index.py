#!/usr/bin/env python3
"""Bucket metadata utility for length-indexed batch tokenization."""

from pathlib import Path
from typing import Dict, List, Optional


class BucketIndex:
    """Loads and queries length bucket metadata from TSV files.

    The bucket metadata file format is:
        YOUTUBE_ID<tab>GLOBAL_ID<tab>BUCKET_LENGTH<tab>ACTUAL_LENGTH

    Where GLOBAL_ID maps directly to HuggingFace dataset indices,
    enabling efficient O(1) filtering via dataset.select(indices).

    Example:
        >>> bi = BucketIndex('/path/to/metadata', 'bal_train').load()
        >>> indices = bi.get_indices(240000)  # Get all 10-second clips
        >>> print(f"Found {len(indices)} samples in bucket 240000")
    """

    def __init__(self, metadata_dir: str, split: str = "bal_train"):
        """Initialize BucketIndex.

        Args:
            metadata_dir: Directory containing bucket TSV files
            split: Dataset split name (used to construct filename)
        """
        self.metadata_dir = Path(metadata_dir)
        self.split = split
        self._bucket_to_indices: Dict[int, List[int]] = {}
        self._loaded = False

    def load(self) -> "BucketIndex":
        """Load bucket metadata from TSV file.

        TSV format: YOUTUBE_ID, GLOBAL_ID, BUCKET_LENGTH, ACTUAL_LENGTH

        Returns:
            self for method chaining

        Raises:
            FileNotFoundError: If metadata file doesn't exist
        """
        path = self.metadata_dir / f"audioset_{self.split}_buckets.tsv"

        if not path.exists():
            raise FileNotFoundError(f"Bucket metadata file not found: {path}")

        self._bucket_to_indices.clear()

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')
                if len(parts) < 3:
                    continue

                # Format: YOUTUBE_ID, GLOBAL_ID, BUCKET_LENGTH, ACTUAL_LENGTH
                global_id = int(parts[1])
                bucket = int(parts[2])

                if bucket not in self._bucket_to_indices:
                    self._bucket_to_indices[bucket] = []
                self._bucket_to_indices[bucket].append(global_id)

        self._loaded = True
        return self

    def get_indices(self, bucket: int) -> List[int]:
        """Get HuggingFace dataset indices for a single bucket.

        Args:
            bucket: Single bucket length (e.g., 240000 for 10-sec at 24kHz)

        Returns:
            Sorted list of dataset indices matching the bucket

        Raises:
            RuntimeError: If metadata hasn't been loaded yet
            ValueError: If bucket not found
        """
        if not self._loaded:
            raise RuntimeError("BucketIndex not loaded. Call load() first.")

        if bucket not in self._bucket_to_indices:
            available = self.get_available_buckets()[:10]
            raise ValueError(
                f"Bucket {bucket} not found. Available buckets (first 10): {available}"
            )

        return sorted(self._bucket_to_indices[bucket])

    def get_available_buckets(self) -> List[int]:
        """List all available bucket lengths.

        Returns:
            Sorted list of bucket lengths

        Raises:
            RuntimeError: If metadata hasn't been loaded yet
        """
        if not self._loaded:
            raise RuntimeError("BucketIndex not loaded. Call load() first.")

        return sorted(self._bucket_to_indices.keys())

    def get_bucket_counts(self) -> Dict[int, int]:
        """Get sample counts for each bucket.

        Returns:
            Dictionary mapping bucket length to sample count

        Raises:
            RuntimeError: If metadata hasn't been loaded yet
        """
        if not self._loaded:
            raise RuntimeError("BucketIndex not loaded. Call load() first.")

        return {bucket: len(indices) for bucket, indices in self._bucket_to_indices.items()}

    def __len__(self) -> int:
        """Return total number of indexed samples."""
        return sum(len(indices) for indices in self._bucket_to_indices.values())

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"BucketIndex(split='{self.split}', status={status}, buckets={len(self._bucket_to_indices)})"
