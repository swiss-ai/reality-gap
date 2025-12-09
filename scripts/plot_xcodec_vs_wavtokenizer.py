#!/usr/bin/env python3
"""
XCodec2 vs WavTokenizer Top/Bottom Languages Comparison
Generates side-by-side plots comparing top and bottom 5 languages for key metrics
"""

import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

# Configuration
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.resolve()
METRICS_TO_PLOT = ['snr_db', 'pesq', 'stoi']
N_LANGUAGES = 5
TOKENIZERS = ['xcodec2', 'wavtokenizer']

# Colors
COLORS = {
    'xcodec2': '#9b59b6',
    'wavtokenizer': '#f39c12'
}


class ComparisonPlotter:
    def __init__(self, data_dir: str = None):
        """Initialize with data directory"""
        if data_dir is None:
            self.data_dir = PROJECT_ROOT / "metrics"
        else:
            self.data_dir = Path(data_dir)
        self.data = defaultdict(list)

    def load_data(self):
        """Load data for XCodec2 and WavTokenizer only"""
        print("Loading data for XCodec2 and WavTokenizer...")

        for tokenizer in TOKENIZERS:
            tokenizer_files = glob.glob(str(self.data_dir / f"{tokenizer}_*_results.json"))
            tokenizer_files = sorted([f for f in tokenizer_files if 'summary' not in f and 'all_results' not in f])

            for file_path in tokenizer_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                    # Extract language from filename
                    basename = Path(file_path).stem
                    if basename.startswith(f"{tokenizer}_"):
                        basename = basename[len(tokenizer) + 1:]
                    if basename.endswith("_results"):
                        basename = basename[:-8]

                    # Remove dataset prefixes
                    known_datasets = ['eurospeech', 'gtzan', 'naturelm']
                    for dataset in known_datasets:
                        if basename.startswith(f"{dataset}_"):
                            basename = basename[len(dataset) + 1:]

                    language = basename

                    # Extract metrics
                    metrics = data.get('metrics', {})
                    entry = {
                        'tokenizer': tokenizer,
                        'language': language,
                        'dataset': data.get('dataset', 'unknown')
                    }

                    # Add metric means
                    for metric in METRICS_TO_PLOT:
                        if metric in metrics and metrics[metric] is not None:
                            if isinstance(metrics[metric], dict):
                                entry[metric] = metrics[metric].get('mean')
                            else:
                                entry[metric] = metrics[metric]

                    self.data[tokenizer].append(entry)

                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue

        print(f"Loaded {len(self.data['xcodec2'])} files for XCodec2")
        print(f"Loaded {len(self.data['wavtokenizer'])} files for WavTokenizer")

    def create_dataframe(self) -> pd.DataFrame:
        """Create DataFrame from loaded data"""
        rows = []
        for tokenizer in TOKENIZERS:
            rows.extend(self.data[tokenizer])
        return pd.DataFrame(rows)

    def plot_top_bottom_comparison(self, df: pd.DataFrame, metric: str):
        """
        Create side-by-side comparison of top and bottom N languages
        for XCodec2 and WavTokenizer
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'Top and Bottom {N_LANGUAGES} Languages by {metric.upper()}: XCodec2 vs WavTokenizer',
                     fontsize=16, fontweight='bold')

        for idx, tokenizer in enumerate(TOKENIZERS):
            ax = axes[idx]
            df_tok = df[df['tokenizer'] == tokenizer].copy()

            if len(df_tok) == 0:
                print(f"Warning: No data for {tokenizer}")
                continue

            # Drop NaN values for this metric
            df_tok = df_tok.dropna(subset=[metric])

            if len(df_tok) == 0:
                print(f"Warning: No valid {metric} data for {tokenizer}")
                continue

            # Sort by metric
            df_sorted = df_tok.sort_values(metric)

            # Get top and bottom N
            bottom_n = df_sorted.head(N_LANGUAGES)
            top_n = df_sorted.tail(N_LANGUAGES)

            # Combine
            combined = pd.concat([bottom_n, top_n])

            # Determine colors based on metric type
            if metric in ['mse']:  # Lower is better
                colors = ['#2ecc71'] * N_LANGUAGES + ['#e74c3c'] * N_LANGUAGES
            else:  # Higher is better
                colors = ['#e74c3c'] * N_LANGUAGES + ['#2ecc71'] * N_LANGUAGES

            # Create horizontal bar plot
            y_positions = range(len(combined))
            bars = ax.barh(y_positions, combined[metric], color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

            # Set labels and title
            ax.set_yticks(y_positions)
            ax.set_yticklabels(combined['language'], fontsize=10)
            ax.set_xlabel(metric.upper(), fontsize=12, fontweight='bold')
            ax.set_title(f'{tokenizer.upper()}', fontsize=14, fontweight='bold',
                         color=COLORS[tokenizer])

            # Add separator line between bottom and top
            ax.axhline(y=N_LANGUAGES - 0.5, color='black', linestyle='--', linewidth=2, alpha=0.5)

            # Add grid
            ax.grid(True, alpha=0.3, axis='x')
            ax.set_axisbelow(True)

            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, combined[metric])):
                # Position text inside bar if bar is long enough, otherwise outside
                bar_width = bar.get_width()
                x_max = ax.get_xlim()[1]

                if abs(bar_width) > 0.15 * x_max:  # If bar is >15% of axis width
                    # Inside bar
                    x_pos = bar_width / 2
                    ha = 'center'
                    color = 'white'
                else:
                    # Outside bar
                    x_pos = bar_width + (0.02 * x_max if bar_width >= 0 else -0.02 * x_max)
                    ha = 'left' if bar_width >= 0 else 'right'
                    color = 'black'

                ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                        f'{val:.3f}',
                        ha=ha, va='center', fontweight='bold', fontsize=9, color=color)

            # Add labels for bottom/top sections
            ax.text(0.02, N_LANGUAGES * 0.5, 'WORST',
                    transform=ax.get_yaxis_transform(),
                    fontsize=10, fontweight='bold', color='#e74c3c',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.text(0.02, N_LANGUAGES * 1.5, 'BEST',
                    transform=ax.get_yaxis_transform(),
                    fontsize=10, fontweight='bold', color='#2ecc71',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        # Save figure
        output_file = f'xcodec2_vs_wavtokenizer_top_bottom_{metric}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

    def find_common_extremes(self, df: pd.DataFrame, metric: str):
        """
        Find languages that appear in both tokenizers' top or bottom lists
        """
        print(f"\n{metric.upper()} - Common Extremes Analysis:")

        for tokenizer in TOKENIZERS:
            df_tok = df[df['tokenizer'] == tokenizer].dropna(subset=[metric])
            if len(df_tok) == 0:
                continue

            df_sorted = df_tok.sort_values(metric)
            bottom_langs = set(df_sorted.head(N_LANGUAGES)['language'])
            top_langs = set(df_sorted.tail(N_LANGUAGES)['language'])

            if tokenizer == 'xcodec2':
                xcodec2_bottom = bottom_langs
                xcodec2_top = top_langs
            else:
                wavtokenizer_bottom = bottom_langs
                wavtokenizer_top = top_langs

        # Find common languages
        common_bottom = xcodec2_bottom & wavtokenizer_bottom
        common_top = xcodec2_top & wavtokenizer_top

        if common_bottom:
            print(f"  Languages in BOTH bottom-{N_LANGUAGES}: {', '.join(sorted(common_bottom))}")
        else:
            print(f"  No common languages in bottom-{N_LANGUAGES}")

        if common_top:
            print(f"  Languages in BOTH top-{N_LANGUAGES}: {', '.join(sorted(common_top))}")
        else:
            print(f"  No common languages in top-{N_LANGUAGES}")

    def run_analysis(self):
        """Run the complete analysis"""
        print("=" * 60)
        print("XCODEC2 vs WAVTOKENIZER COMPARISON")
        print("=" * 60)

        # Load data
        self.load_data()

        if not self.data['xcodec2'] or not self.data['wavtokenizer']:
            print("Error: Missing data for one or both tokenizers")
            return

        # Create dataframe
        df = self.create_dataframe()
        print(f"\nTotal entries: {len(df)}")
        print(f"Metrics to plot: {', '.join(METRICS_TO_PLOT)}")

        # Create output directory
        output_dir = PROJECT_ROOT / "results"
        output_dir.mkdir(exist_ok=True)
        print(f"\nSaving results to: {output_dir.absolute()}")

        # Change to output directory
        import os
        original_dir = os.getcwd()
        os.chdir(str(output_dir))

        try:
            print("\nGenerating comparison plots...")

            for metric in METRICS_TO_PLOT:
                if metric not in df.columns:
                    print(f"Warning: Metric {metric} not found in data")
                    continue

                # Check if we have data for this metric
                df_metric = df.dropna(subset=[metric])
                if len(df_metric) == 0:
                    print(f"Warning: No valid data for {metric}")
                    continue

                # Generate plot
                self.plot_top_bottom_comparison(df, metric)

                # Find common extremes
                self.find_common_extremes(df, metric)

            print("\n" + "=" * 60)
            print("ANALYSIS COMPLETE!")
            print("=" * 60)
            print(f"\nAll plots saved to: {output_dir.absolute()}")
            print("\nGenerated files:")
            for metric in METRICS_TO_PLOT:
                print(f"  - xcodec2_vs_wavtokenizer_top_bottom_{metric}.png")

        finally:
            # Change back to original directory
            os.chdir(original_dir)


if __name__ == "__main__":
    import sys

    # Get data directory from command line or use default
    data_dir = sys.argv[1] if len(sys.argv) > 1 else None

    # Run analysis
    plotter = ComparisonPlotter(data_dir)
    print(f"Looking for data in: {plotter.data_dir}")
    plotter.run_analysis()