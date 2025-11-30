#!/usr/bin/env python3
"""Script to inspect parquet files - shows columns and first two rows."""

import pyarrow.parquet as pq
import sys

def inspect_parquet(file_path: str) -> None:
    """Read parquet metadata and first two rows efficiently."""
    print(f"Loading: {file_path}\n")
    
    # Open file without loading all data
    parquet_file = pq.ParquetFile(file_path)
    schema = parquet_file.schema_arrow
    metadata = parquet_file.metadata
    
    print("Columns:")
    print("-" * 40)
    for field in schema:
        print(f"  - {field.name} ({field.type})")
    
    print(f"\nShape: {metadata.num_rows} rows × {metadata.num_columns} columns")
    
    # Read only the first row group, then take 2 rows
    first_rows = parquet_file.read_row_group(0).slice(0, 2).to_pandas()
    
    print("\nFirst 2 rows:")
    print("-" * 40)
    print(first_rows.to_string())

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default path if no argument provided (relative to /scripts)
        file_path = "../data/standardized/voxpopuli/voxpopuli_cs.parquet"
    else:
        file_path = sys.argv[1]
    
    inspect_parquet(file_path)