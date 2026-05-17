#!/usr/bin/env python3
"""Quick schema + row-count peek at a Common Voice processed parquet."""
import argparse
import pyarrow.parquet as pq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("parquet")
    p.add_argument("--show-first-row", action="store_true")
    args = p.parse_args()

    pf = pq.ParquetFile(args.parquet)
    print(f"Rows: {pf.metadata.num_rows:,}")
    print(f"Columns: {pf.schema_arrow.names}")
    if args.show_first_row:
        non_audio = [c for c in pf.schema_arrow.names if "audio" not in c.lower()]
        tbl = pf.read_row_group(0, columns=non_audio)
        print(tbl.slice(0, 1).to_pydict())


if __name__ == "__main__":
    main()
