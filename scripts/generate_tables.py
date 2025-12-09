#!/usr/bin/env python3
"""
LaTeX Table Generator with Bootstrap Statistics
Calculates Mean +/- Std Dev (via 100 bootstrap samples) for audio tokenizers.
"""

import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# --- Configuration ---
N_BOOTSTRAP = 100
METRICS = ['pesq', 'stoi', 'estoi', 'snr_db', 'sdr_db', 'mse']
# Map internal metric names to nice LaTeX column headers
METRIC_MAP = {
    'pesq': 'PESQ',
    'stoi': 'STOI',
    'estoi': 'eSTOI',
    'snr_db': 'SNR (dB)',
    'sdr_db': 'SDR (dB)',
    'mse': 'MSE'
}


def get_directories():
    """Resolve paths relative to this script location"""
    script_dir = Path(__file__).parent.resolve()
    # Assuming structure: /root/scripts/this_script.py and /root/metrics/*.json
    metrics_dir = script_dir.parent / "metrics"
    return metrics_dir


def load_data(metrics_dir):
    """Load JSON files and aggregate by tokenizer"""
    data_map = defaultdict(lambda: defaultdict(list))

    files = glob.glob(str(metrics_dir / "*_*_results.json"))
    valid_files = [f for f in files if 'summary' not in f and 'all_results' not in f]

    print(f"Found {len(valid_files)} result files.")

    for file_path in valid_files:
        try:
            with open(file_path, 'r') as f:
                content = json.load(f)

            # Extract tokenizer name (first part of filename)
            fname = Path(file_path).name
            tokenizer = fname.split('_')[0]

            # Extract metrics
            if 'metrics' in content:
                for metric in METRICS:
                    # Check if metric exists and is not None
                    if metric in content['metrics'] and content['metrics'][metric] is not None:
                        metric_data = content['metrics'][metric]

                        # Check if metric_data is a dict (should be)
                        if isinstance(metric_data, dict):
                            # Get the mean value
                            val = metric_data.get('mean')

                            # Only append if val is not None and is a valid number
                            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                                data_map[tokenizer][metric].append(val)
                        else:
                            # Handle case where metric_data might be a direct value
                            if not (isinstance(metric_data, float) and np.isnan(metric_data)):
                                data_map[tokenizer][metric].append(metric_data)

        except Exception as e:
            print(f"Skipping error in {Path(file_path).name}: {e}")

    # Debug: Print data counts for each tokenizer and metric
    print("\nData counts by tokenizer and metric:")
    for tokenizer in sorted(data_map.keys()):
        print(f"\n{tokenizer.upper()}:")
        for metric in METRICS:
            count = len(data_map[tokenizer].get(metric, []))
            print(f"  {metric}: {count} values")

    return data_map


def bootstrap_confidence(data, n_boot=100):
    """
    Performs bootstrap resampling to estimate mean and std dev.
    Returns: (mean, std_dev)
    """
    if not data:
        return np.nan, np.nan

    data_array = np.array(data)

    # Check for NaN values
    data_array = data_array[~np.isnan(data_array)]

    if len(data_array) == 0:
        return np.nan, np.nan

    means = []

    # Random seed for reproducibility
    np.random.seed(42)

    for _ in range(n_boot):
        # Resample with replacement
        sample = np.random.choice(data_array, size=len(data_array), replace=True)
        means.append(np.mean(sample))

    # Calculate stats of the bootstrap distribution
    boot_mean = np.mean(means)
    boot_std = np.std(means)

    return boot_mean, boot_std


def generate_latex(data_map):
    """Generates the LaTeX table string"""
    tokenizers = sorted(data_map.keys())

    # Prepare table header
    latex = []
    latex.append(r"\begin{table*}[ht]")
    latex.append(r"\centering")
    latex.append(
        r"\caption{Tokenizer performance comparison across all datasets. Values represent mean $\pm$ standard deviation calculated via " + str(
            N_BOOTSTRAP) + r" bootstrap resamples over language means. Higher values indicate better performance for PESQ, STOI, eSTOI, SNR, and SDR; lower values are better for MSE. Best performance in each metric is shown in \textbf{bold}.}")
    latex.append(r"\label{tab:tokenizer_benchmark}")

    # Dynamic column definition
    col_def = "l" + "c" * len(METRICS)
    latex.append(r"\begin{tabular}{" + col_def + r"}")
    latex.append(r"\toprule")

    # Header Row with arrows
    headers = []
    for m in METRICS:
        if m == 'mse':
            headers.append(f"{METRIC_MAP[m]} $\\downarrow$")
        else:
            headers.append(f"{METRIC_MAP[m]} $\\uparrow$")
    header_row = "Model & " + " & ".join(headers) + r" \\"
    latex.append(header_row)
    latex.append(r"\midrule")

    # First pass: calculate all means to find best values
    best_values = {}
    for metric in METRICS:
        values_by_tokenizer = {}
        for tokenizer in tokenizers:
            values = data_map[tokenizer].get(metric, [])
            if len(values) > 0:
                mean, std = bootstrap_confidence(values, N_BOOTSTRAP)
                if not np.isnan(mean):
                    values_by_tokenizer[tokenizer] = mean

        if values_by_tokenizer:
            # For MSE, lower is better; for others, higher is better
            if metric == 'mse':
                best_tokenizer = min(values_by_tokenizer, key=values_by_tokenizer.get)
            else:
                best_tokenizer = max(values_by_tokenizer, key=values_by_tokenizer.get)
            best_values[metric] = best_tokenizer

    # Data Rows
    for tokenizer in tokenizers:
        row_str = f"\\textbf{{{tokenizer.upper()}}}"

        for metric in METRICS:
            values = data_map[tokenizer].get(metric, [])
            if len(values) > 0:
                mean, std = bootstrap_confidence(values, N_BOOTSTRAP)

                # Check if this is the best value for this metric
                is_best = (metric in best_values and
                           best_values[metric] == tokenizer and
                           not np.isnan(mean))

                if not np.isnan(mean):
                    if is_best:
                        cell = f"$\\mathbf{{{mean:.3f} \\pm {std:.3f}}}$"
                    else:
                        cell = f"${mean:.3f} \\pm {std:.3f}$"
                else:
                    cell = "$-$"
            else:
                cell = "$-$"

            row_str += f" & {cell}"

        row_str += r" \\"
        latex.append(row_str)

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table*}")

    return "\n".join(latex)


def main():
    metrics_dir = get_directories()
    if not metrics_dir.exists():
        print(f"Error: Metrics directory not found at {metrics_dir}")
        return

    print("Loading data and computing bootstrap statistics...")
    data = load_data(metrics_dir)

    if not data:
        print("No valid data found.")
        return

    print(f"\nGenerating LaTeX table for {len(data)} tokenizers...")
    latex_output = generate_latex(data)

    # Output to console
    print("\n" + "=" * 40)
    print("LaTeX TABLE OUTPUT")
    print("=" * 40 + "\n")
    print(latex_output)

    # Output to file
    out_file = Path(__file__).parent / "tokenizer_metrics_table.tex"
    with open(out_file, "w") as f:
        f.write(latex_output)
    print(f"\nSaved LaTeX table to: {out_file}")


if __name__ == "__main__":
    main()