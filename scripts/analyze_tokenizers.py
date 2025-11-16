#!/usr/bin/env python3
"""
Tokenizer Benchmark Analysis Script
Analyzes and visualizes all tokenizer performance across languages and datasets
Automatically detects all tokenizers from result files in metrics directory
"""

import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Color palette for tokenizers (will be extended if needed)
DEFAULT_COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
MARKERS = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']

# Get script directory for path resolution
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.resolve()

class TokenizerAnalyzer:
    def __init__(self, data_dir: str = None):
        """Initialize analyzer with data directory"""
        if data_dir is None:
            self.data_dir = PROJECT_ROOT / "metrics"
        else:
            self.data_dir = Path(data_dir)
        self.tokenizer_data = defaultdict(list)  # {tokenizer_name: [data, ...]}
        self.tokenizer_languages = defaultdict(set)  # {tokenizer_name: {languages}}
        self.metrics = ['mse', 'snr_db', 'sdr_db', 'pesq', 'stoi', 'estoi']
        self.all_languages = set()  # Track all unique languages
        self.tokenizers = []  # List of detected tokenizer names (sorted)
        self.language_metrics_count = defaultdict(dict)  # {(tokenizer, language): num_metrics}
    
    def detect_tokenizers(self) -> List[str]:
        """Detect all tokenizers from result files in the data directory"""
        all_files = glob.glob(str(self.data_dir / "*_*_results.json"))
        tokenizers = set()
        
        for file in all_files:
            basename = Path(file).name
            # Extract tokenizer name (first part before first underscore)
            if '_' in basename:
                tokenizer = basename.split('_')[0]
                # Skip summary and aggregate files
                if 'summary' not in basename and 'all_results' not in basename:
                    tokenizers.add(tokenizer)
        
        return sorted(list(tokenizers))
    
    def extract_language_from_filename(self, filename: str, tokenizer: str) -> str:
        """
        Extract language from filename, handling different naming conventions.
        
        Examples:
        - neucodec_estonia_results.json -> estonia
        - neucodec_fleurs_en_us_results.json -> en_us 
        - xcodec2_eurospeech_estonia_results.json -> estonia
        - cosyvoice2_gtzan_blues_results.json -> blues
        
        Note: FLEURS prefix is kept as it's part of the language identifier.
        """
        basename = Path(filename).stem  # Remove .json
        
        # Remove the tokenizer prefix
        if basename.startswith(f"{tokenizer}_"):
            basename = basename[len(tokenizer)+1:]
        
        # Remove _results suffix
        if basename.endswith("_results"):
            basename = basename[:-8]
        
        # Remove dataset prefixes (eurospeech, fleurs, gtzan, naturelm)
        # but keep fleurs prefix as it's part of language identifier
        known_datasets = ['eurospeech', 'gtzan', 'naturelm']
        for dataset in known_datasets:
            if basename.startswith(f"{dataset}_"):
                basename = basename[len(dataset)+1:]
        
        # Keep fleurs prefix as it's part of language identifier
        # (fleurs languages are like fleurs_en_us)
        
        return basename
    
    def validate_metrics_file(self, file_path: Path) -> Tuple[bool, int]:
        """
        Validate that a metrics file is complete and has all required fields with values.
        Based on validation logic from submit_missing_jobs.py.
        
        Returns:
            (is_valid, num_valid_metrics) tuple where:
            - is_valid: True if the file is valid
            - num_valid_metrics: Number of valid metrics (0-6)
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            return (False, 0)
        
        # Check required top-level fields
        required_fields = ['language', 'dataset', 'num_samples', 'metrics', 
                           'tokens_per_second', 'compression_ratio']
        for field in required_fields:
            if field not in data:
                return (False, 0)
        
        # Check num_samples is valid
        if not isinstance(data['num_samples'], int) or data['num_samples'] <= 0:
            return (False, 0)
        
        # Check metrics structure
        metrics = data.get('metrics', {})
        if not isinstance(metrics, dict):
            return (False, 0)
        
        # Expected metric names
        expected_metrics = ['mse', 'snr_db', 'sdr_db', 'pesq', 'stoi', 'estoi']
        
        # Count valid metrics (metrics that exist, are not None, and have all required fields)
        num_valid_metrics = 0
        for metric_name in expected_metrics:
            if metric_name in metrics and metrics[metric_name] is not None:
                metric_data = metrics[metric_name]
                if not isinstance(metric_data, dict):
                    continue
                # Check required fields for non-null metrics
                required_metric_fields = ['mean', 'std', 'min', 'max', 'median']
                is_valid_metric = True
                for field in required_metric_fields:
                    if field not in metric_data:
                        is_valid_metric = False
                        break
                    # Check that values are numbers (not None)
                    if metric_data[field] is None:
                        is_valid_metric = False
                        break
                if is_valid_metric:
                    num_valid_metrics += 1
        
        # Check tokens_per_second structure
        tps = data.get('tokens_per_second', {})
        if not isinstance(tps, dict) or 'mean' not in tps or 'std' not in tps:
            return (False, num_valid_metrics)
        if tps['mean'] is None or tps['std'] is None:
            return (False, num_valid_metrics)
        
        # Check compression_ratio structure
        cr = data.get('compression_ratio', {})
        if not isinstance(cr, dict) or 'mean' not in cr or 'std' not in cr:
            return (False, num_valid_metrics)
        if cr['mean'] is None or cr['std'] is None:
            return (False, num_valid_metrics)
        
        # File is valid if it has at least some structure, even if not all metrics are valid
        # We return True if basic structure is OK, and num_valid_metrics shows completeness
        return (True, num_valid_metrics)
        
    def load_data(self):
        """Load all JSON result files for all detected tokenizers"""
        print("Detecting tokenizers...")
        self.tokenizers = self.detect_tokenizers()
        
        if not self.tokenizers:
            print("Warning: No tokenizers detected in data directory!")
            return
        
        print(f"Found tokenizers: {', '.join(self.tokenizers)}")
        print("\nLoading data files...")
        
        # Load data for each tokenizer
        invalid_files = []
        for tokenizer in self.tokenizers:
            tokenizer_files = glob.glob(str(self.data_dir / f"{tokenizer}_*_results.json"))
            tokenizer_files = [f for f in tokenizer_files if 'summary' not in f and 'all_results' not in f]
            
            for file in tokenizer_files:
                file_path = Path(file)
                try:
                    # Validate metrics file
                    is_valid, num_valid_metrics = self.validate_metrics_file(file_path)
                    
                    if not is_valid:
                        invalid_files.append((file_path.name, "Invalid structure"))
                        continue
                    
                    with open(file, 'r') as f:
                        data = json.load(f)
                        data['tokenizer'] = tokenizer
                        data['file'] = file_path.name
                        data['num_valid_metrics'] = num_valid_metrics
                        
                        # Extract language from filename to ensure consistency
                        filename_language = self.extract_language_from_filename(file, tokenizer)
                        
                        # Use language from JSON, but verify/normalize with filename
                        json_language = data.get('language', filename_language)
                        
                        # For consistency, use the filename-extracted language as canonical
                        data['language_canonical'] = filename_language
                        data['language_from_json'] = json_language
                        data['language'] = filename_language  # Use filename as source of truth
                        
                        # Extract dataset type
                        if 'fleurs' in file:
                            data['dataset'] = 'fleurs'
                        elif 'eurospeech' in file:
                            data['dataset'] = 'eurospeech'
                        elif 'gtzan' in file:
                            data['dataset'] = 'gtzan'
                        elif 'naturelm' in file:
                            data['dataset'] = 'naturelm'
                        else:
                            # Default: assume eurospeech for country names
                            data['dataset'] = 'eurospeech'
                        
                        self.tokenizer_data[tokenizer].append(data)
                        self.tokenizer_languages[tokenizer].add(data['language'])
                        self.all_languages.add(data['language'])
                        
                        # Store metrics count for this tokenizer-language combination
                        key = (tokenizer, data['language'])
                        self.language_metrics_count[key] = num_valid_metrics
                except Exception as e:
                    invalid_files.append((file_path.name, str(e)))
                    continue
        
        # Print summary
        total_files = sum(len(data) for data in self.tokenizer_data.values())
        print(f"\nLoaded {total_files} valid result files:")
        for tokenizer in self.tokenizers:
            count = len(self.tokenizer_data[tokenizer])
            langs = len(self.tokenizer_languages[tokenizer])
            print(f"  {tokenizer}: {count} files, {langs} languages")
        
        # Report invalid files
        if invalid_files:
            print(f"\nWarning: Found {len(invalid_files)} invalid files (skipped):")
            for filename, reason in invalid_files[:10]:  # Show first 10
                print(f"  {filename}: {reason}")
            if len(invalid_files) > 10:
                print(f"  ... and {len(invalid_files) - 10} more invalid files")
        
        # Report on language coverage
        print(f"\nLanguage Coverage:")
        print(f"  Total unique languages: {len(self.all_languages)}")
        
        # Find common languages (languages present in all tokenizers)
        if len(self.tokenizers) > 1:
            common_languages = set.intersection(*[self.tokenizer_languages[t] for t in self.tokenizers])
            print(f"  Common to all tokenizers: {len(common_languages)}")
            if common_languages and len(common_languages) <= 20:
                print(f"    Languages: {', '.join(sorted(common_languages))}")
            elif common_languages:
                common_list = sorted(list(common_languages))
                print(f"    Languages: {', '.join(common_list[:10])}... (+{len(common_languages)-10} more)")
        
        # Report tokenizer-specific languages
        for tokenizer in self.tokenizers:
            other_tokenizers = [t for t in self.tokenizers if t != tokenizer]
            if other_tokenizers:
                other_languages = set.union(*[self.tokenizer_languages[t] for t in other_tokenizers])
                tokenizer_only = self.tokenizer_languages[tokenizer] - other_languages
                if tokenizer_only:
                    print(f"  {tokenizer} only: {len(tokenizer_only)} languages")
                    if len(tokenizer_only) <= 10:
                        print(f"    {', '.join(sorted(tokenizer_only))}")
        
        # Report metrics completeness
        print(f"\nMetrics Completeness:")
        for tokenizer in self.tokenizers:
            metrics_counts = {}
            for data in self.tokenizer_data[tokenizer]:
                num_metrics = data.get('num_valid_metrics', 0)
                metrics_counts[num_metrics] = metrics_counts.get(num_metrics, 0) + 1
            
            print(f"  {tokenizer}:")
            for num_metrics in sorted(metrics_counts.keys(), reverse=True):
                count = metrics_counts[num_metrics]
                print(f"    {num_metrics}/6 metrics: {count} languages")
        
        # Warn about any mismatches between filename and JSON language field
        print(f"\nVerifying language consistency...")
        mismatches = []
        for tokenizer in self.tokenizers:
            for data in self.tokenizer_data[tokenizer]:
                if data['language_canonical'] != data['language_from_json']:
                    mismatches.append(f"  {data['file']}: filename='{data['language_canonical']}' vs json='{data['language_from_json']}'")
        
        if mismatches:
            print(f"Warning: Found {len(mismatches)} filename/JSON language mismatches:")
            for mismatch in mismatches[:5]:  # Show first 5
                print(mismatch)
            if len(mismatches) > 5:
                print(f"  ... and {len(mismatches) - 5} more")
            print("Using filename-based language names for consistency.")
        else:
            print("All filenames match their JSON language fields ✓")
        
    def create_dataframe(self) -> pd.DataFrame:
        """Create a consolidated DataFrame from all results"""
        rows = []
        
        for tokenizer in self.tokenizers:
            for data in self.tokenizer_data[tokenizer]:
                row = {
                    'tokenizer': data['tokenizer'],
                    'language': data['language'],
                    'dataset': data['dataset'],
                    'num_samples': data['num_samples']
                }
                
                # Add metric means
                metrics = data.get('metrics', {})
                for metric in self.metrics:
                    metric_data = metrics.get(metric)
                    if metric_data is not None:
                        row[f'{metric}_mean'] = metric_data.get('mean')
                        if 'std' in metric_data:
                            row[f'{metric}_std'] = metric_data.get('std')
                
                # Add tokens per second and compression ratio if available
                tps = data.get('tokens_per_second')
                if tps is not None and isinstance(tps, dict):
                    row['tokens_per_second'] = tps.get('mean')
                cr = data.get('compression_ratio')
                if cr is not None and isinstance(cr, dict):
                    row['compression_ratio'] = cr.get('mean')
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_tokenizer_color(self, tokenizer: str) -> str:
        """Get color for a tokenizer"""
        idx = self.tokenizers.index(tokenizer) if tokenizer in self.tokenizers else 0
        return DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
    
    def get_tokenizer_marker(self, tokenizer: str) -> str:
        """Get marker for a tokenizer"""
        idx = self.tokenizers.index(tokenizer) if tokenizer in self.tokenizers else 0
        return MARKERS[idx % len(MARKERS)]
    
    def plot_language_coverage(self, df: pd.DataFrame):
        """
        Visualize language coverage for each tokenizer.
        Colors indicate number of valid metrics (0-6) for each language.
        """
        fig, ax = plt.subplots(1, 1, figsize=(14, max(8, len(self.all_languages) * 0.3)))
        
        # Create a matrix showing number of valid metrics for each tokenizer-language combination
        languages = sorted(self.all_languages)
        coverage_data = []
        
        for lang in languages:
            row = []
            for tokenizer in self.tokenizers:
                key = (tokenizer, lang)
                num_metrics = self.language_metrics_count.get(key, 0)
                # Treat missing languages as 0 (both will be red)
                if lang not in self.tokenizer_languages[tokenizer]:
                    num_metrics = 0
                row.append(num_metrics)
            coverage_data.append(row)
        
        coverage_array = np.array(coverage_data)
        
        # Create heatmap with custom colormap
        # 0-6 = number of valid metrics (red to green gradient)
        colors_list = ['#e74c3c',   # Red for 0 metrics (or missing)
                       '#e67e22',   # Dark orange for 1 metric
                       '#f39c12',   # Orange for 2 metrics
                       '#f1c40f',   # Yellow for 3 metrics
                       '#f1c40f',   # Yellow for 4 metrics
                       '#2ecc71',   # Light green for 5 metrics
                       '#27ae60']   # Normal green for 6 metrics (all complete)
        
        # Create colormap: 0, 1, 2, 3, 4, 5, 6
        cmap = ListedColormap(colors_list)
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        norm = BoundaryNorm(bounds, cmap.N)
        
        im = ax.imshow(coverage_array.T, cmap=cmap, norm=norm, aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(languages)))
        ax.set_xticklabels(languages, rotation=90, ha='right', fontsize=8)
        ax.set_yticks(range(len(self.tokenizers)))
        ax.set_yticklabels([t.upper() for t in self.tokenizers])
        
        # Add text annotations showing number of metrics
        for i in range(len(languages)):
            for j in range(len(self.tokenizers)):
                num_metrics = coverage_array[i, j]
                text = str(int(num_metrics))
                color = 'black' if num_metrics >= 3 else 'white'
                
                ax.text(i, j, text,
                       ha="center", va="center", color=color, 
                       fontsize=8, fontweight='bold')
        
        ax.set_title('Language Coverage by Tokenizer\n(Color = Number of Valid Metrics, 0-6)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Custom colorbar with labels
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4, 5, 6])
        cbar.set_ticklabels(['0', '1', '2', '3', '4', '5', '6'])
        cbar.set_label('Number of Valid Metrics', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig('language_coverage.png', dpi=300, bbox_inches='tight')
        print("Saved: language_coverage.png")
        plt.close()
    
    def plot_common_languages_comparison(self, df: pd.DataFrame):
        """Compare tokenizers only on languages all have"""
        if len(self.tokenizers) < 2:
            print("Warning: Need at least 2 tokenizers for comparison")
            return
        
        # Find languages present in all tokenizers
        common_languages = set.intersection(*[self.tokenizer_languages[t] for t in self.tokenizers])
        
        if len(common_languages) == 0:
            print("Warning: No common languages found between all tokenizers")
            return
        
        df_common = df[df['language'].isin(common_languages)].copy()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Performance Comparison (Common Languages Only, n={len(common_languages)})', 
                     fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(self.metrics):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            metric_col = f'{metric}_mean'
            if metric_col not in df_common.columns:
                continue
            
            # Create violin plot for all tokenizers
            data_to_plot = []
            for tokenizer in self.tokenizers:
                data = df_common[df_common['tokenizer'] == tokenizer][metric_col].dropna()
                if len(data) > 0:
                    data_to_plot.append(data)
            
            if len(data_to_plot) > 0:
                positions = list(range(len(data_to_plot)))
                parts = ax.violinplot(data_to_plot, positions=positions, showmeans=True, showmedians=True)
                ax.set_xticks(positions)
                ax.set_xticklabels([t.upper() for t in self.tokenizers])
                ax.set_ylabel(metric.upper())
                ax.set_title(f'{metric.upper()} Distribution')
                ax.grid(True, alpha=0.3)
                
                # Add mean values as text
                for i, data in enumerate(data_to_plot):
                    if len(data) > 0:
                        mean_val = data.mean()
                        ax.text(i, mean_val, f'{mean_val:.3f}', 
                               ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('common_languages_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved: common_languages_comparison.png")
        plt.close()
    
    def plot_overall_comparison(self, df: pd.DataFrame):
        """Create overall comparison between tokenizers"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Overall Tokenizer Performance Comparison', fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(self.metrics):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            metric_col = f'{metric}_mean'
            if metric_col not in df.columns:
                continue
            
            # Create violin plot for all tokenizers
            data_to_plot = []
            for tokenizer in self.tokenizers:
                data = df[df['tokenizer'] == tokenizer][metric_col].dropna()
                if len(data) > 0:
                    data_to_plot.append(data)
            
            if len(data_to_plot) > 0:
                positions = list(range(len(data_to_plot)))
                parts = ax.violinplot(data_to_plot, positions=positions, showmeans=True, showmedians=True)
                ax.set_xticks(positions)
                ax.set_xticklabels([t.upper() for t in self.tokenizers])
                ax.set_ylabel(metric.upper())
                ax.set_title(f'{metric.upper()} Distribution')
                ax.grid(True, alpha=0.3)
                
                # Add mean values as text
                for i, data in enumerate(data_to_plot):
                    if len(data) > 0:
                        mean_val = data.mean()
                        ax.text(i, mean_val, f'{mean_val:.3f}', 
                               ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('overall_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved: overall_comparison.png")
        plt.close()
    
    def plot_metric_comparison_bars(self, df: pd.DataFrame):
        """Create bar charts comparing metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Mean Performance by Metric', fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(self.metrics):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            metric_col = f'{metric}_mean'
            if metric_col not in df.columns:
                continue
            
            # Calculate means and stds for all tokenizers
            # AGGREGATION METHOD: 
            # - Each language file has a pre-computed mean (from ~100 samples)
            # - We compute the mean of these per-language means (grand mean)
            # - This gives equal weight to each language, regardless of sample count
            # - Std shows variation ACROSS languages (not across samples within language)
            means = []
            stds = []
            colors = []
            labels = []
            
            for tokenizer in self.tokenizers:
                data = df[df['tokenizer'] == tokenizer][metric_col].dropna()
                if len(data) > 0:
                    means.append(data.mean())
                    stds.append(data.std())
                    colors.append(self.get_tokenizer_color(tokenizer))
                    labels.append(tokenizer.upper())
            
            if len(means) > 0:
                # Create bar plot
                bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
                
                ax.set_ylabel(metric.upper())
                ax.set_title(f'{metric.upper()}')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for bar, val in zip(bars, means):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.4f}',
                           ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('metric_comparison_bars.png', dpi=300, bbox_inches='tight')
        print("Saved: metric_comparison_bars.png")
        plt.close()
    
    def plot_top_bottom_languages(self, df: pd.DataFrame, metric: str = 'pesq', n: int = 5):
        """Plot top and bottom N languages for a given metric"""
        metric_col = f'{metric}_mean'
        
        if metric_col not in df.columns:
            print(f"Metric {metric} not found in data")
            return
        
        # Create subplots for each tokenizer
        n_tokenizers = len(self.tokenizers)
        if n_tokenizers == 0:
            return
        
        n_cols = min(3, n_tokenizers)
        n_rows = (n_tokenizers + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
        fig.suptitle(f'Top and Bottom {n} Languages by {metric.upper()}', 
                     fontsize=16, fontweight='bold')
        
        # Ensure axes is always a list/array of Axes objects
        if n_tokenizers == 1:
            axes = [axes]
        elif n_rows == 1:
            # axes is already a 1D array, convert to list
            axes = list(axes) if hasattr(axes, '__iter__') else [axes]
        else:
            # axes is a 2D array, flatten to 1D and convert to list
            axes = list(axes.flatten())
        
        for idx, tokenizer in enumerate(self.tokenizers):
            if idx >= len(axes):
                break
            ax = axes[idx]
            df_tok = df[df['tokenizer'] == tokenizer].copy()
            
            if len(df_tok) == 0:
                continue
            
            # Sort by metric
            df_sorted = df_tok.sort_values(metric_col)
            
            # Get top and bottom N
            bottom_n = df_sorted.head(n)
            top_n = df_sorted.tail(n)
            
            # Combine
            combined = pd.concat([bottom_n, top_n])
            
            # Create bar plot
            if metric == 'mse':
                colors = ['#2ecc71'] * n + ['#e74c3c'] * n  # green for low MSE
            else:
                colors = ['#e74c3c'] * n + ['#2ecc71'] * n
            bars = ax.barh(range(len(combined)), combined[metric_col], color=colors, alpha=0.7)
            ax.set_yticks(range(len(combined)))
            ax.set_yticklabels(combined['language'])
            ax.set_xlabel(metric.upper())
            ax.set_title(f'{tokenizer.upper()}')
            ax.axhline(y=n-0.5, color='black', linestyle='--', linewidth=2, alpha=0.3)
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, combined[metric_col])):
                ax.text(val, bar.get_y() + bar.get_height()/2, 
                       f'{val:.3f}',
                       ha='left', va='center', fontweight='bold', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'top_bottom_languages_{metric}.png', dpi=300, bbox_inches='tight')
        print(f"Saved: top_bottom_languages_{metric}.png")
        plt.close()
    
    def plot_dataset_comparison(self, df: pd.DataFrame):
        """Compare performance across datasets"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Performance by Dataset', fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(self.metrics):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            metric_col = f'{metric}_mean'
            if metric_col not in df.columns:
                continue
            
            # Group by tokenizer and dataset
            grouped = df.groupby(['tokenizer', 'dataset'])[metric_col].mean().reset_index()
            
            # Pivot for easier plotting
            pivot = grouped.pivot(index='dataset', columns='tokenizer', values=metric_col)
            
            if not pivot.empty:
                # Get colors for each tokenizer
                colors = [self.get_tokenizer_color(t) for t in pivot.columns if t in self.tokenizers]
                pivot.plot(kind='bar', ax=ax, color=colors, alpha=0.7)
                ax.set_ylabel(metric.upper())
                ax.set_title(f'{metric.upper()} by Dataset')
                ax.set_xlabel('Dataset')
                ax.legend(title='Tokenizer')
                ax.grid(True, alpha=0.3, axis='y')
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('dataset_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved: dataset_comparison.png")
        plt.close()
    
    def plot_compression_efficiency(self, df: pd.DataFrame):
        """Plot compression ratio and tokens per second"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Compression Efficiency', fontsize=16, fontweight='bold')
        
        # Compression ratio
        if 'compression_ratio' in df.columns:
            ax = axes[0]
            data_to_plot = []
            labels = []
            colors = []
            
            for tokenizer in self.tokenizers:
                data = df[df['tokenizer'] == tokenizer]['compression_ratio'].dropna()
                if len(data) > 0:
                    data_to_plot.append(data)
                    labels.append(tokenizer.upper())
                    colors.append(self.get_tokenizer_color(tokenizer))
            
            if len(data_to_plot) > 0:
                bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax.set_ylabel('Compression Ratio')
                ax.set_title('Compression Ratio Distribution')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add mean values
                for i, data in enumerate(data_to_plot):
                    mean_val = data.mean()
                    ax.text(i+1, mean_val, f'{mean_val:.1f}', 
                           ha='center', va='bottom', fontweight='bold', color='red')
        
        # Tokens per second
        if 'tokens_per_second' in df.columns:
            ax = axes[1]
            data_to_plot = []
            labels = []
            colors = []
            
            for tokenizer in self.tokenizers:
                data = df[df['tokenizer'] == tokenizer]['tokens_per_second'].dropna()
                if len(data) > 0:
                    data_to_plot.append(data)
                    labels.append(tokenizer.upper())
                    colors.append(self.get_tokenizer_color(tokenizer))
            
            if len(data_to_plot) > 0:
                bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax.set_ylabel('Tokens per Second')
                ax.set_title('Processing Speed')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add mean values
                for i, data in enumerate(data_to_plot):
                    mean_val = data.mean()
                    ax.text(i+1, mean_val, f'{mean_val:.1f}', 
                           ha='center', va='bottom', fontweight='bold', color='red')
        
        plt.tight_layout()
        plt.savefig('compression_efficiency.png', dpi=300, bbox_inches='tight')
        print("Saved: compression_efficiency.png")
        plt.close()
    
    def plot_correlation_heatmap(self, df: pd.DataFrame):
        """Plot correlation heatmap between metrics"""
        n_tokenizers = len(self.tokenizers)
        if n_tokenizers == 0:
            return
        
        n_cols = min(3, n_tokenizers)
        n_rows = (n_tokenizers + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
        fig.suptitle('Metric Correlations', fontsize=16, fontweight='bold')
        
        metric_cols = [f'{m}_mean' for m in self.metrics if f'{m}_mean' in df.columns]
        
        # Ensure axes is always a list/array of Axes objects
        if n_tokenizers == 1:
            axes = [axes]
        elif n_rows == 1:
            # axes is already a 1D array, convert to list
            axes = list(axes) if hasattr(axes, '__iter__') else [axes]
        else:
            # axes is a 2D array, flatten to 1D and convert to list
            axes = list(axes.flatten())
        
        for idx, tokenizer in enumerate(self.tokenizers):
            if idx >= len(axes):
                break
            ax = axes[idx]
            df_tok = df[df['tokenizer'] == tokenizer][metric_cols]
            
            if len(df_tok) > 0:
                corr = df_tok.corr()
                
                # Create heatmap
                sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                           center=0, square=True, ax=ax,
                           xticklabels=[m.upper() for m in self.metrics if f'{m}_mean' in df.columns],
                           yticklabels=[m.upper() for m in self.metrics if f'{m}_mean' in df.columns])
                ax.set_title(f'{tokenizer.upper()}')
        
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("Saved: correlation_heatmap.png")
        plt.close()
    
    def plot_language_scatter(self, df: pd.DataFrame, metric_x: str = 'pesq', metric_y: str = 'stoi'):
        """Scatter plot comparing two metrics across languages"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        metric_x_col = f'{metric_x}_mean'
        metric_y_col = f'{metric_y}_mean'
        
        if metric_x_col not in df.columns or metric_y_col not in df.columns:
            print(f"Metrics {metric_x} or {metric_y} not found")
            return
        
        # Plot for each tokenizer
        for tokenizer in self.tokenizers:
            df_tok = df[df['tokenizer'] == tokenizer]
            if len(df_tok) > 0:
                color = self.get_tokenizer_color(tokenizer)
                marker = self.get_tokenizer_marker(tokenizer)
                ax.scatter(df_tok[metric_x_col], df_tok[metric_y_col], 
                          c=color, label=tokenizer.upper(), alpha=0.6, 
                          s=100, marker=marker, edgecolors='black', linewidths=0.5)
        
        ax.set_xlabel(metric_x.upper(), fontsize=12)
        ax.set_ylabel(metric_y.upper(), fontsize=12)
        ax.set_title(f'{metric_x.upper()} vs {metric_y.upper()} Across Languages', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'scatter_{metric_x}_vs_{metric_y}.png', dpi=300, bbox_inches='tight')
        print(f"Saved: scatter_{metric_x}_vs_{metric_y}.png")
        plt.close()
    
    def generate_summary_stats(self, df: pd.DataFrame):
        """Generate and save summary statistics"""
        summary = []
        
        summary.append(f"\n{'='*60}")
        summary.append("AGGREGATION METHODOLOGY")
        summary.append(f"{'='*60}")
        summary.append("\nHow metrics are aggregated across languages:")
        summary.append("")
        summary.append("1. Each language file contains MEAN metrics already computed")
        summary.append("   from multiple audio samples (typically 100 samples per language)")
        summary.append("")
        summary.append("2. For OVERALL tokenizer statistics:")
        summary.append("   - We compute the MEAN of the per-language means")
        summary.append("   - This is effectively a 'grand mean' across all languages")
        summary.append("   - Each language contributes equally, regardless of sample count")
        summary.append("")
        summary.append("3. Example calculation:")
        summary.append("   - Language A: PESQ mean = 1.75 (from 100 samples)")
        summary.append("   - Language B: PESQ mean = 1.80 (from 100 samples)")
        summary.append("   - Language C: PESQ mean = 1.70 (from 100 samples)")
        summary.append("   - Overall PESQ = (1.75 + 1.80 + 1.70) / 3 = 1.75")
        summary.append("")
        summary.append("4. Standard deviation calculation:")
        summary.append("   - Computed from the per-language means")
        summary.append("   - Shows variation ACROSS languages (not across samples)")
        summary.append("   - High std = performance varies significantly by language")
        summary.append("")
        summary.append("5. For FAIR comparison (common languages only):")
        summary.append("   - Same methodology, but only using languages both tokenizers have")
        summary.append("   - Ensures apples-to-apples comparison")
        summary.append("")
        summary.append("NOTE: This is language-weighted averaging. If you need")
        summary.append("      sample-weighted averaging (each audio sample counts equally),")
        summary.append("      you would need to implement a weighted mean using num_samples.")
        
        # Add language coverage section
        summary.append(f"\n{'='*60}")
        summary.append("LANGUAGE COVERAGE ANALYSIS")
        summary.append(f"{'='*60}")
        
        summary.append(f"\nTotal unique languages: {len(self.all_languages)}")
        
        # Find common languages (present in all tokenizers)
        if len(self.tokenizers) > 1:
            common_languages = set.intersection(*[self.tokenizer_languages[t] for t in self.tokenizers])
            summary.append(f"Languages in all tokenizers: {len(common_languages)}")
            if common_languages and len(common_languages) <= 20:
                summary.append(f"  {', '.join(sorted(common_languages))}")
            elif common_languages:
                common_list = sorted(list(common_languages))
                summary.append(f"  {', '.join(common_list[:10])}... (+{len(common_languages)-10} more)")
        
        # Report tokenizer-specific languages
        for tokenizer in self.tokenizers:
            other_tokenizers = [t for t in self.tokenizers if t != tokenizer]
            if other_tokenizers:
                other_languages = set.union(*[self.tokenizer_languages[t] for t in other_tokenizers])
                tokenizer_only = self.tokenizer_languages[tokenizer] - other_languages
                if tokenizer_only:
                    summary.append(f"\nLanguages only in {tokenizer.upper()}: {len(tokenizer_only)}")
                    if len(tokenizer_only) <= 10:
                        summary.append(f"  {', '.join(sorted(tokenizer_only))}")
        
        summary.append(f"\nNOTE: Overall comparisons include ALL languages for each tokenizer.")
        summary.append(f"      This means the comparison may not be perfectly fair if")
        summary.append(f"      the tokenizers were tested on different language sets.")
        if len(self.tokenizers) > 1:
            common_languages = set.intersection(*[self.tokenizer_languages[t] for t in self.tokenizers])
            summary.append(f"      See 'common_languages_comparison.png' for a fair comparison")
            summary.append(f"      using only the {len(common_languages)} languages all tokenizers have.")
        
        for tokenizer in self.tokenizers:
            df_tok = df[df['tokenizer'] == tokenizer]
            
            summary.append(f"\n{'='*60}")
            summary.append(f"{tokenizer.upper()} SUMMARY STATISTICS")
            summary.append(f"{'='*60}")
            summary.append(f"Total languages tested: {len(df_tok)}")
            summary.append(f"Total samples: {df_tok['num_samples'].sum()}")
            
            summary.append(f"\n{'-'*60}")
            summary.append("METRIC STATISTICS")
            summary.append(f"{'-'*60}")
            
            for metric in self.metrics:
                metric_col = f'{metric}_mean'
                if metric_col in df.columns:
                    values = df_tok[metric_col].dropna()
                    if len(values) > 0:
                        summary.append(f"\n{metric.upper()}:")
                        summary.append(f"  Mean: {values.mean():.4f}")
                        summary.append(f"  Std:  {values.std():.4f}")
                        summary.append(f"  Min:  {values.min():.4f}")
                        summary.append(f"  Max:  {values.max():.4f}")
                        summary.append(f"  Median: {values.median():.4f}")
            
            if 'compression_ratio' in df.columns:
                cr_values = df_tok['compression_ratio'].dropna()
                if len(cr_values) > 0:
                    summary.append(f"\nCOMPRESSION RATIO:")
                    summary.append(f"  Mean: {cr_values.mean():.2f}")
                    summary.append(f"  Std:  {cr_values.std():.2f}")
            
            if 'tokens_per_second' in df.columns:
                tps_values = df_tok['tokens_per_second'].dropna()
                if len(tps_values) > 0:
                    summary.append(f"\nTOKENS PER SECOND:")
                    summary.append(f"  Mean: {tps_values.mean():.2f}")
                    summary.append(f"  Std:  {tps_values.std():.2f}")
        
        # Overall comparison section (pairwise comparisons)
        if len(self.tokenizers) > 1:
            summary.append(f"\n{'='*60}")
            summary.append("OVERALL COMPARISON (All Tokenizers)")
            summary.append(f"{'='*60}")
            summary.append("NOTE: Includes all languages each tokenizer was tested on")
            
            for metric in self.metrics:
                metric_col = f'{metric}_mean'
                if metric_col not in df.columns:
                    continue
                
                summary.append(f"\n{metric.upper()}:")
                means = {}
                for tokenizer in self.tokenizers:
                    data = df[df['tokenizer'] == tokenizer][metric_col].dropna()
                    if len(data) > 0:
                        means[tokenizer] = data.mean()
                        summary.append(f"  {tokenizer.upper()}: {means[tokenizer]:.4f}")
                
                # Find best tokenizer
                if len(means) > 0:
                    if metric == 'mse':
                        best_tokenizer = min(means, key=means.get)
                    else:
                        best_tokenizer = max(means, key=means.get)
                    summary.append(f"  Best: {best_tokenizer.upper()}")
        
        # Common languages comparison
        if len(self.tokenizers) > 1:
            common_languages = set.intersection(*[self.tokenizer_languages[t] for t in self.tokenizers])
            if len(common_languages) > 0:
                df_common = df[df['language'].isin(common_languages)].copy()
                
                summary.append(f"\n{'='*60}")
                summary.append(f"FAIR COMPARISON (Common Languages Only, n={len(common_languages)})")
                summary.append(f"{'='*60}")
                if len(common_languages) <= 20:
                    summary.append(f"Common languages: {', '.join(sorted(common_languages))}")
                else:
                    common_list = sorted(list(common_languages))
                    summary.append(f"Common languages: {', '.join(common_list[:10])}... (+{len(common_languages)-10} more)")
                
                for metric in self.metrics:
                    metric_col = f'{metric}_mean'
                    if metric_col not in df_common.columns:
                        continue
                    
                    summary.append(f"\n{metric.upper()}:")
                    means = {}
                    for tokenizer in self.tokenizers:
                        data = df_common[df_common['tokenizer'] == tokenizer][metric_col].dropna()
                        if len(data) > 0 and not np.isnan(data.mean()):
                            means[tokenizer] = data.mean()
                            summary.append(f"  {tokenizer.upper()}: {means[tokenizer]:.4f}")
                    
                    # Find best tokenizer
                    if len(means) > 0:
                        if metric == 'mse':
                            best_tokenizer = min(means, key=means.get)
                        else:
                            best_tokenizer = max(means, key=means.get)
                        summary.append(f"  Best: {best_tokenizer.upper()}")
        
        # Save to file
        summary_text = '\n'.join(summary)
        with open('analysis_summary.txt', 'w') as f:
            f.write(summary_text)
        
        print("\n" + summary_text)
        print("\nSaved: analysis_summary.txt")
    
    def run_full_analysis(self):
        """Run complete analysis and generate all visualizations"""
        print("="*60)
        print("TOKENIZER BENCHMARK ANALYSIS")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Create dataframe
        df = self.create_dataframe()
        print(f"\nTotal entries in dataframe: {len(df)}")
        print(f"Tokenizers: {df['tokenizer'].unique()}")
        print(f"Datasets: {df['dataset'].unique()}")
        
        # Create results directory
        results_dir = PROJECT_ROOT / "results"
        results_dir.mkdir(exist_ok=True)
        print(f"\nSaving results to: {results_dir.absolute()}")
        
        # Change to results directory for saving outputs
        import os
        original_dir = os.getcwd()
        os.chdir(str(results_dir))
        
        try:
            # Generate all visualizations
            print("\nGenerating visualizations...")
            
            # Language coverage analysis
            self.plot_language_coverage(df)
            
            # Overall comparison (all languages)
            self.plot_overall_comparison(df)
            
            # Fair comparison (common languages only)
            self.plot_common_languages_comparison(df)
            
            # Other analyses
            self.plot_metric_comparison_bars(df)
            self.plot_dataset_comparison(df)
            self.plot_compression_efficiency(df)
            self.plot_correlation_heatmap(df)
            
            # Top/bottom languages for key metrics
            for metric in ['pesq', 'stoi', 'mse']:
                if f'{metric}_mean' in df.columns:
                    self.plot_top_bottom_languages(df, metric=metric, n=5)
            
            # Scatter plots for metric relationships
            self.plot_language_scatter(df, 'pesq', 'stoi')
            self.plot_language_scatter(df, 'snr_db', 'sdr_db')
            
            # Generate summary statistics
            self.generate_summary_stats(df)
            
            print("\n" + "="*60)
            print("ANALYSIS COMPLETE!")
            print("="*60)
            print(f"\nAll results saved to: {results_dir.absolute()}")
            print("\nGenerated files:")
            print("  - language_coverage.png (shows which languages each tokenizer has)")
            print("  - overall_comparison.png (all languages, may not be fair)")
            print("  - common_languages_comparison.png (only common languages, fair comparison)")
            print("  - metric_comparison_bars.png")
            print("  - dataset_comparison.png")
            print("  - compression_efficiency.png")
            print("  - correlation_heatmap.png")
            print("  - top_bottom_languages_*.png (for PESQ, STOI, MSE)")
            print("  - scatter_*_vs_*.png")
            print("  - analysis_summary.txt (includes language coverage details)")
            print("\nIMPORTANT: Check language_coverage.png and the summary to see which")
            print("           languages are missing from each tokenizer. The 'common languages'")
            print("           comparison provides the fairest head-to-head comparison.")
        finally:
            # Change back to original directory
            os.chdir(original_dir)


if __name__ == "__main__":
    import sys
    
    # Get data directory from command line argument or use default path
    # Default assumes script is in benchmark-audio-tokenizers/scripts/
    # and data is in benchmark-audio-tokenizers/metrics/
    data_dir = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Run analysis
    analyzer = TokenizerAnalyzer(data_dir)
    print(f"Looking for data in: {analyzer.data_dir}")
    analyzer.run_full_analysis()