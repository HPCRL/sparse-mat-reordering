"""
draw_performance_profiles.py

Generates a single performance profile plot per pivoted CSV in the specified directory,
without binning. Each column is assumed to be numeric GFLOPs for some reordering.

Usage:
    python draw_performance_profiles.py --indir output_pivoted
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import argparse

def performance_profile(method_times, best_times):
    """
    bigger-is-better => ratio = best_times / method_times
    """
    ratio = best_times / method_times
    ratio_sorted = np.sort(ratio.dropna())
    y = np.arange(1, len(ratio_sorted) + 1) / len(ratio_sorted)
    return ratio_sorted, y

def plot_performance_profile_for_csv(csv_file):
    df = pd.read_csv(csv_file)
    # Drop descriptor columns if present
    df = df.drop(columns=['matrix_name','m','k','nnz','n'], errors='ignore')
    
    # Drop fully empty rows
    df = df.dropna(how='all')
    
    # Exclude 'IOU' if present
    if 'IOU' in df.columns:
        df = df.drop(columns=['IOU'])
    
    # If no columns left
    reorderings = df.columns
    if len(reorderings) == 0:
        print(f"No valid reorderings in {csv_file}; skipping.")
        return
    
    # Convert all to numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    # Drop rows where all reorderings are NaN
    df = df.dropna(subset=reorderings, how='all')
    
    # Compute best
    best_times = df.max(axis=1)
    
    plt.figure(figsize=(8,6))
    
    for method_name in reorderings:
        series = df[method_name]
        if series.isna().all():
            continue
        x, y = performance_profile(series, best_times)
        plt.step(x, y, where='post', label=method_name)
    
    # Not using log scale, limit x to [1,2]
    plt.xlim(1, 2)
    plt.ylim(0, 1.05)
    plt.xlabel("Performance Relative to Best")
    plt.ylabel("Fraction of Problems")
    plt.title(f"Performance Profile: {os.path.basename(csv_file)}")
    plt.grid(True)
    plt.legend(loc='lower right')
    
    out_png = csv_file.replace(".csv", ".png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created {out_png}")

def main(indir):
    csv_files = glob.glob(os.path.join(indir, "*.csv"))
    for f in csv_files:
        plot_performance_profile_for_csv(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", default="output_pivoted", help="Directory of pivoted CSVs")
    args = parser.parse_args()
    main(args.indir)
