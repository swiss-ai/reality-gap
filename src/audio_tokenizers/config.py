"""
Configuration module for audio tokenizers package.

This module defines package-level configurations and paths.
"""

from pathlib import Path

# Get the package root directory (where this config.py file is)
PACKAGE_ROOT = Path(__file__).parent

# Define the repos directory path
# This goes up to src/ then to repos/
REPOS_DIR = PACKAGE_ROOT.parent / "repos"

# Alternative: Could also use an environment variable if needed
import os
if os.environ.get("AUDIO_TOKENIZERS_REPOS_DIR"):
    REPOS_DIR = Path(os.environ["AUDIO_TOKENIZERS_REPOS_DIR"])

# Ensure repos directory exists
if not REPOS_DIR.exists():
    # Try to create it
    try:
        REPOS_DIR.mkdir(parents=True, exist_ok=True)
    except:
        # If we can't create it, at least warn
        import warnings
        warnings.warn(f"Repos directory not found and couldn't be created: {REPOS_DIR}")

def get_repo_path(repo_name: str) -> Path:
    """Get the full path to a specific repo.

    Args:
        repo_name: Name of the repo subdirectory

    Returns:
        Full path to the repo directory
    """
    return REPOS_DIR / repo_name