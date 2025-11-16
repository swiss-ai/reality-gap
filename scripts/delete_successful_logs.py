#!/usr/bin/env python3
"""
Deletes all .out and .err log files from the logs directory that have run successfully.

A job is considered successful if the .out file contains "Exit code: 0".

Usage:
    python scripts/delete_successful_logs.py  # Deletes all successful logs
    python scripts/delete_successful_logs.py --dry-run  # Shows only what would be deleted
"""

import argparse
from pathlib import Path
from typing import List, Tuple

# Project root directory
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
logs_dir = project_root / "logs"


def is_successful(log_file: Path) -> bool:
    """
    Checks if a .out file contains a successful exit code (0).
    
    Args:
        log_file: Path to the .out file
        
    Returns:
        True if "Exit code: 0" is found, otherwise False
    """
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            return "Exit code: 0" in content
    except (IOError, OSError) as e:
        print(f"Warning: Could not read {log_file}: {e}")
        return False


def find_successful_logs() -> List[Tuple[Path, Path]]:
    """
    Finds all successful log file pairs (.out and .err).
    
    Returns:
        List of tuples (out_file, err_file) for successful jobs
    """
    successful_pairs = []
    
    # Find all .out files
    out_files = sorted(logs_dir.glob("*.out"))
    
    for out_file in out_files:
        if is_successful(out_file):
            # Find the corresponding .err file
            err_file = out_file.with_suffix('.err')
            if err_file.exists():
                successful_pairs.append((out_file, err_file))
            else:
                # If no .err file exists, we still delete the .out file
                successful_pairs.append((out_file, None))
    
    return successful_pairs


def main():
    parser = argparse.ArgumentParser(
        description="Deletes successful log files from the logs directory"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Shows only what would be deleted without actually deleting'
    )
    args = parser.parse_args()
    
    if not logs_dir.exists():
        print(f"Error: Logs directory does not exist: {logs_dir}")
        return 1
    
    print(f"Searching for successful log files in {logs_dir}...")
    successful_pairs = find_successful_logs()
    
    if not successful_pairs:
        print("No successful log files found.")
        return 0
    
    print(f"\nFound: {len(successful_pairs)} successful job(s)")
    print("\nFiles that would be deleted:")
    for out_file, err_file in successful_pairs:
        print(f"  - {out_file.name}")
        if err_file:
            print(f"  - {err_file.name}")
    
    if args.dry_run:
        print("\n[DRY-RUN] No files were deleted.")
        return 0
    
    # Confirmation
    print(f"\nDo you want to delete these {len(successful_pairs)} successful log file(s)? (yes/no): ", end='')
    response = input().strip().lower()
    
    if response not in ['yes', 'y']:
        print("Cancelled.")
        return 0
    
    # Delete the files
    deleted_count = 0
    for out_file, err_file in successful_pairs:
        try:
            out_file.unlink()
            deleted_count += 1
            print(f"Deleted: {out_file.name}")
        except OSError as e:
            print(f"Error deleting {out_file.name}: {e}")
        
        if err_file:
            try:
                err_file.unlink()
                deleted_count += 1
                print(f"Deleted: {err_file.name}")
            except OSError as e:
                print(f"Error deleting {err_file.name}: {e}")
    
    print(f"\nDone! {deleted_count} file(s) deleted.")
    return 0


if __name__ == "__main__":
    exit(main())

