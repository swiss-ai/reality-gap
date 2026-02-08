"""Megatron indexed-dataset helpers and merge utilities."""

from .indexed_dataset_megatron import DType, IndexedDatasetBuilder
from .merge_indexed_dataset import discover_indexed_prefixes, merge_indexed_dataset

__all__ = [
    "DType",
    "IndexedDatasetBuilder",
    "discover_indexed_prefixes",
    "merge_indexed_dataset",
]
