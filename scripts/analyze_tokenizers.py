#!/usr/bin/env python3
"""
Tokenizer Benchmark Analysis Script
Analyzes and visualizes neucodec and xcodec2 performance across languages and datasets
"""

import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class TokenizerAnalyzer:
    def __init__(self, data_dir: str = "../metrics"):
        """Initialize analyzer with data directory"""
        self.data_dir = Path(data_dir)
        self.neucodec_data = []
        self.xcodec2_data = []
        self.metrics = ['mse', 'snr_db', 'sdr_db', 'pesq', 'stoi', 'estoi']
        self.all_languages = set()  # Track all unique languages
        self.neucodec_languages = set()
        self.xcodec2_languages = set()
    
    def extract_language_from_filename(self, filename: str, tokenizer: str) -> str:
        """
        Extract language from filename, handling different naming conventions.
        
        Examples:
        - neucodec_estonia_results.json -> estonia
        - neucodec_fleurs_en_us_results.json -> en_us 
        - xcodec2_eurospeech_estonia_results.json -> estonia
        
        Note: FLEURS prefix is kept as it's part of the language identifier.
        """
        basename = Path(filename).stem  # Remove .json
        
        # Remove the tokenizer prefix
        if basename.startswith(f"{tokenizer}_"):
            basename = basename[len(tokenizer)+1:]
        
        # Remove _results suffix
        if basename.endswith("_results"):
            basename = basename[:-8]
        
        # For xcodec2, remove eurospeech prefix if present (but NOT fleurs prefix)
        if tokenizer == "xcodec2" and basename.startswith("eurospeech_"):
            basename = basename[11:]  # len("eurospeech_") = 11
        
        if basename.startswith("fleurs_"):
            basename = basename[len("fleurs_"):]
        
        return basename
        
    def load_data(self):
        """Load all JSON result files"""
        print("Loading data files...")
        
        # Load neucodec results (excluding summaries and all_results)
        neucodec_files = glob.glob(str(self.data_dir / "neucodec_*_results.json"))
        neucodec_files = [f for f in neucodec_files if 'summary' not in f and 'all_results' not in f]
        
        for file in neucodec_files:
            with open(file, 'r') as f:
                data = json.load(f)
                data['tokenizer'] = 'neucodec'
                data['file'] = Path(file).name
                
                # Extract language from filename to ensure consistency
                filename_language = self.extract_language_from_filename(file, 'neucodec')
                
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
                else:
                    data['dataset'] = 'eurospeech'  # country names are eurospeech
                
                self.neucodec_data.append(data)
                self.neucodec_languages.add(data['language'])
                self.all_languages.add(data['language'])
        
        # Load xcodec2 results
        xcodec2_files = glob.glob(str(self.data_dir / "xcodec2_*_results.json"))
        xcodec2_files = [f for f in xcodec2_files if 'summary' not in f and 'all_results' not in f]
        
        for file in xcodec2_files:
            with open(file, 'r') as f:
                data = json.load(f)
                data['tokenizer'] = 'xcodec2'
                data['file'] = Path(file).name
                
                # Extract language from filename to ensure consistency
                filename_language = self.extract_language_from_filename(file, 'xcodec2')
                
                # Use language from JSON, but verify/normalize with filename
                json_language = data.get('language', filename_language)
                
                # For consistency, use the filename-extracted language as canonical
                data['language_canonical'] = filename_language
                data['language_from_json'] = json_language
                data['language'] = filename_language  # Use filename as source of truth
                
                # Extract dataset type
                if 'eurospeech' in file:
                    data['dataset'] = 'eurospeech'
                else:
                    data['dataset'] = 'unknown'
                
                self.xcodec2_data.append(data)
                self.xcodec2_languages.add(data['language'])
                self.all_languages.add(data['language'])
        
        print(f"Loaded {len(self.neucodec_data)} neucodec files and {len(self.xcodec2_data)} xcodec2 files")
        
        # Report on language coverage
        neucodec_only = self.neucodec_languages - self.xcodec2_languages
        xcodec2_only = self.xcodec2_languages - self.neucodec_languages
        common = self.neucodec_languages & self.xcodec2_languages
        
        print(f"\nLanguage Coverage:")
        print(f"  Total unique languages: {len(self.all_languages)}")
        print(f"  Common to both: {len(common)}")
        if common:
            common_list = sorted(list(common))
            print(f"    Languages: {', '.join(common_list[:10])}{'...' if len(common) > 10 else ''}")
        print(f"  NeuCodec only: {len(neucodec_only)}")
        if neucodec_only:
            neucodec_only_list = sorted(list(neucodec_only))
            print(f"    Languages: {', '.join(neucodec_only_list[:10])}{'...' if len(neucodec_only) > 10 else ''}")
        print(f"  XCodec2 only: {len(xcodec2_only)}")
        if xcodec2_only:
            xcodec2_only_list = sorted(list(xcodec2_only))
            print(f"    Languages: {', '.join(xcodec2_only_list[:10])}{'...' if len(xcodec2_only) > 10 else ''}")
        
        # Warn about any mismatches between filename and JSON language field
        print(f"\nVerifying language consistency...")
        mismatches = []
        for data in self.neucodec_data + self.xcodec2_data:
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
        
        for data in self.neucodec_data + self.xcodec2_data:
            row = {
                'tokenizer': data['tokenizer'],
                'language': data['language'],
                'dataset': data['dataset'],
                'num_samples': data['num_samples']
            }
            
            # Add metric means
            for metric in self.metrics:
                if metric in data['metrics']:
                    row[f'{metric}_mean'] = data['metrics'][metric]['mean']
                    row[f'{metric}_std'] = data['metrics'][metric]['std']
            
            # Add tokens per second and compression ratio if available
            if 'tokens_per_second' in data:
                row['tokens_per_second'] = data['tokens_per_second']['mean']
            if 'compression_ratio' in data:
                row['compression_ratio'] = data['compression_ratio']['mean']
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def plot_language_coverage(self, df: pd.DataFrame):
        """Visualize language coverage for each tokenizer"""
        fig, ax = plt.subplots(1, 1, figsize=(14, max(8, len(self.all_languages) * 0.3)))
        
        # Create a matrix showing which languages are available for which tokenizer
        languages = sorted(self.all_languages)
        coverage_data = []
        
        for lang in languages:
            has_neucodec = 1 if lang in self.neucodec_languages else 0
            has_xcodec2 = 1 if lang in self.xcodec2_languages else 0
            coverage_data.append([has_neucodec, has_xcodec2])
        
        coverage_array = np.array(coverage_data)
        
        # Create heatmap
        im = ax.imshow(coverage_array.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(len(languages)))
        ax.set_xticklabels(languages, rotation=90, ha='right')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['NeuCodec', 'XCodec2'])
        
        # Add text annotations
        for i in range(len(languages)):
            for j in range(2):
                text = ax.text(i, j, '✓' if coverage_array[i, j] == 1 else '✗',
                             ha="center", va="center", color="black", fontsize=10, fontweight='bold')
        
        ax.set_title('Language Coverage by Tokenizer', fontsize=14, fontweight='bold', pad=20)
        plt.colorbar(im, ax=ax, label='Available')
        
        plt.tight_layout()
        plt.savefig('language_coverage.png', dpi=300, bbox_inches='tight')
        print("Saved: language_coverage.png")
        plt.close()
    
    def plot_common_languages_comparison(self, df: pd.DataFrame):
        """Compare tokenizers only on languages both have"""
        common_languages = self.neucodec_languages & self.xcodec2_languages
        
        if len(common_languages) == 0:
            print("Warning: No common languages found between tokenizers")
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
            
            # Create violin plot
            data_to_plot = [
                df_common[df_common['tokenizer'] == 'neucodec'][metric_col].dropna(),
                df_common[df_common['tokenizer'] == 'xcodec2'][metric_col].dropna()
            ]
            
            if all(len(d) > 0 for d in data_to_plot):
                parts = ax.violinplot(data_to_plot, positions=[0, 1], showmeans=True, showmedians=True)
                ax.set_xticks([0, 1])
                ax.set_xticklabels(['NeuCodec', 'XCodec2'])
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
            
            # Create violin plot
            data_to_plot = [
                df[df['tokenizer'] == 'neucodec'][metric_col].dropna(),
                df[df['tokenizer'] == 'xcodec2'][metric_col].dropna()
            ]
            
            parts = ax.violinplot(data_to_plot, positions=[0, 1], showmeans=True, showmedians=True)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['NeuCodec', 'XCodec2'])
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
            
            # Calculate means and stds
            # AGGREGATION METHOD: 
            # - Each language file has a pre-computed mean (from ~100 samples)
            # - We compute the mean of these per-language means (grand mean)
            # - This gives equal weight to each language, regardless of sample count
            # - Std shows variation ACROSS languages (not across samples within language)
            neucodec_mean = df[df['tokenizer'] == 'neucodec'][metric_col].mean()
            xcodec2_mean = df[df['tokenizer'] == 'xcodec2'][metric_col].mean()
            neucodec_std = df[df['tokenizer'] == 'neucodec'][metric_col].std()
            xcodec2_std = df[df['tokenizer'] == 'xcodec2'][metric_col].std()
            
            # Create bar plot
            bars = ax.bar(['NeuCodec', 'XCodec2'], 
                         [neucodec_mean, xcodec2_mean],
                         yerr=[neucodec_std, xcodec2_std],
                         capsize=5,
                         color=['#3498db', '#e74c3c'],
                         alpha=0.7)
            
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{metric.upper()}')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, val in zip(bars, [neucodec_mean, xcodec2_mean]):
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
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Top and Bottom {n} Languages by {metric.upper()}', 
                     fontsize=16, fontweight='bold')
        
        for idx, tokenizer in enumerate(['neucodec', 'xcodec2']):
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
                pivot.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c'], alpha=0.7)
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
            data_to_plot = [
                df[df['tokenizer'] == 'neucodec']['compression_ratio'].dropna(),
                df[df['tokenizer'] == 'xcodec2']['compression_ratio'].dropna()
            ]
            
            if all(len(d) > 0 for d in data_to_plot):
                bp = ax.boxplot(data_to_plot, labels=['NeuCodec', 'XCodec2'], patch_artist=True)
                for patch, color in zip(bp['boxes'], ['#3498db', '#e74c3c']):
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
            data_to_plot = [
                df[df['tokenizer'] == 'neucodec']['tokens_per_second'].dropna(),
                df[df['tokenizer'] == 'xcodec2']['tokens_per_second'].dropna()
            ]
            
            if all(len(d) > 0 for d in data_to_plot):
                bp = ax.boxplot(data_to_plot, labels=['NeuCodec', 'XCodec2'], patch_artist=True)
                for patch, color in zip(bp['boxes'], ['#3498db', '#e74c3c']):
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
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Metric Correlations', fontsize=16, fontweight='bold')
        
        metric_cols = [f'{m}_mean' for m in self.metrics if f'{m}_mean' in df.columns]
        
        for idx, tokenizer in enumerate(['neucodec', 'xcodec2']):
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
        for tokenizer, color, marker in [('neucodec', '#3498db', 'o'), 
                                         ('xcodec2', '#e74c3c', 's')]:
            df_tok = df[df['tokenizer'] == tokenizer]
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
        
        neucodec_only = self.neucodec_languages - self.xcodec2_languages
        xcodec2_only = self.xcodec2_languages - self.neucodec_languages
        common = self.neucodec_languages & self.xcodec2_languages
        
        summary.append(f"\nTotal unique languages: {len(self.all_languages)}")
        summary.append(f"Languages in both tokenizers: {len(common)}")
        summary.append(f"Languages only in NeuCodec: {len(neucodec_only)}")
        if neucodec_only:
            summary.append(f"  {', '.join(sorted(neucodec_only))}")
        summary.append(f"Languages only in XCodec2: {len(xcodec2_only)}")
        if xcodec2_only:
            summary.append(f"  {', '.join(sorted(xcodec2_only))}")
        
        summary.append(f"\nNOTE: Overall comparisons include ALL languages for each tokenizer.")
        summary.append(f"      This means the comparison may not be perfectly fair if")
        summary.append(f"      the tokenizers were tested on different language sets.")
        summary.append(f"      See 'common_languages_comparison.png' for a fair comparison")
        summary.append(f"      using only the {len(common)} languages both tokenizers have.")
        
        for tokenizer in ['neucodec', 'xcodec2']:
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
        
        # Overall comparison section
        summary.append(f"\n{'='*60}")
        summary.append("OVERALL COMPARISON (NeuCodec vs XCodec2)")
        summary.append(f"{'='*60}")
        summary.append("NOTE: Includes all languages each tokenizer was tested on")
        
        for metric in self.metrics:
            metric_col = f'{metric}_mean'
            if metric_col in df.columns:
                neucodec_mean = df[df['tokenizer'] == 'neucodec'][metric_col].mean()
                xcodec2_mean = df[df['tokenizer'] == 'xcodec2'][metric_col].mean()
                diff = neucodec_mean - xcodec2_mean
                pct_diff = (diff / abs(xcodec2_mean)) * 100 if xcodec2_mean != 0 else 0
                
                better = "NeuCodec" if diff > 0 else "XCodec2"
                # For MSE, lower is better
                if metric == 'mse':
                    better = "NeuCodec" if diff < 0 else "XCodec2"
                
                summary.append(f"\n{metric.upper()}:")
                summary.append(f"  NeuCodec: {neucodec_mean:.4f}")
                summary.append(f"  XCodec2:  {xcodec2_mean:.4f}")
                summary.append(f"  Difference: {diff:.4f} ({pct_diff:.2f}%)")
                summary.append(f"  Better: {better}")
        
        # Common languages comparison
        common_languages = self.neucodec_languages & self.xcodec2_languages
        if len(common_languages) > 0:
            df_common = df[df['language'].isin(common_languages)].copy()
            
            summary.append(f"\n{'='*60}")
            summary.append(f"FAIR COMPARISON (Common Languages Only, n={len(common_languages)})")
            summary.append(f"{'='*60}")
            summary.append(f"Common languages: {', '.join(sorted(common_languages))}")
            
            for metric in self.metrics:
                metric_col = f'{metric}_mean'
                if metric_col in df_common.columns:
                    neucodec_mean = df_common[df_common['tokenizer'] == 'neucodec'][metric_col].mean()
                    xcodec2_mean = df_common[df_common['tokenizer'] == 'xcodec2'][metric_col].mean()
                    
                    if not np.isnan(neucodec_mean) and not np.isnan(xcodec2_mean):
                        diff = neucodec_mean - xcodec2_mean
                        pct_diff = (diff / abs(xcodec2_mean)) * 100 if xcodec2_mean != 0 else 0
                        
                        better = "NeuCodec" if diff > 0 else "XCodec2"
                        # For MSE, lower is better
                        if metric == 'mse':
                            better = "NeuCodec" if diff < 0 else "XCodec2"
                        
                        summary.append(f"\n{metric.upper()}:")
                        summary.append(f"  NeuCodec: {neucodec_mean:.4f}")
                        summary.append(f"  XCodec2:  {xcodec2_mean:.4f}")
                        summary.append(f"  Difference: {diff:.4f} ({pct_diff:.2f}%)")
                        summary.append(f"  Better: {better}")
        
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
        results_dir = Path("../results")
        results_dir.mkdir(exist_ok=True)
        print(f"\nSaving results to: {results_dir.absolute()}")
        
        # Change to results directory for saving outputs
        import os
        original_dir = os.getcwd()
        os.chdir(results_dir)
        
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
    
    # Get data directory from command line argument or use default relative path
    # Default assumes script is in benchmark-audio-tokenizers/scripts/
    # and data is in benchmark-audio-tokenizers/metrics/
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "../metrics"
    
    print(f"Looking for data in: {data_dir}")
    
    # Run analysis
    analyzer = TokenizerAnalyzer(data_dir)
    analyzer.run_full_analysis()