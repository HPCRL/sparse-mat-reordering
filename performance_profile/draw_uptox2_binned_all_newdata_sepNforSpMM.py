"""
draw_binned_performance_profiles.py

Generates performance-profile plots for each pivoted CSV,
with binning by X relative to L2 cache.

Usage:
    python draw_binned_performance_profiles.py --indir output_pivoted
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
import argparse

# Machine -> L2 size in bytes
# (1 MB = 1024 * 1024, 512 KB = 512 * 1024, etc.)
MACHINE_L2_CACHE = {
    "athena":       1024 * 1024,  # 1MB
    "AMD-Desktop":  512 * 1024,
    "intel-Desktop":512 * 1024,
    "zeus":         512 * 1024
}

def parse_machine_and_nth(filename):
    """
    Attempt to parse <method>_<submethod>_<machine>_nth<NN>_n<M>.csv
    or e.g. spmv_basic_athena_nth16.csv
    If fails, return (None, None).
    """
    base = os.path.basename(filename)
    # We'll try to find "machine" as the 3rd chunk, nth as the 4th chunk
    parts = base.split("_")
    # e.g. spmm_basic_zeus_nth16_n64.csv => 
    # parts: [spmm, basic, zeus, nth16, n64.csv]
    if len(parts) < 4:
        return (None, None)
    machine = parts[2]
    
    nth_part = parts[3]  # e.g. "nth16"
    match = re.search(r'nth(\d+)', nth_part)
    if not match:
        return (machine, None)
    nth = int(match.group(1))
    
    return (machine, nth)

def performance_profile(method_times, best_times):
    """ bigger-is-better => ratio = best_times / method_times """
    ratio = best_times / method_times
    ratio_sorted = np.sort(ratio.dropna())
    y = np.arange(1, len(ratio_sorted) + 1) / len(ratio_sorted)
    return ratio_sorted, y

def plot_binned_performance_profiles(df, reorderings, l2_cache_bytes, P, out_png):
    """
    Bins: [small: X < L2], [medium: L2 <= X < 2*L2], [large: X >= 2*L2].
    X = (nnz/P)*4 + (n*4)
    Then draw subplots [ALL, SMALL, MEDIUM, LARGE].
    """
    # Convert reorderings to numeric
    df[reorderings] = df[reorderings].apply(pd.to_numeric, errors='coerce')
    
    # Convert nnz, n to numeric
    if 'nnz' not in df.columns or 'n' not in df.columns:
        print(f"Missing nnz/n in pivoted CSV => cannot bin. Skipping {out_png}")
        return
    df['nnz'] = pd.to_numeric(df['nnz'], errors='coerce')
    df['n']   = pd.to_numeric(df['n'], errors='coerce')
    
    # Drop rows missing nnz or n
    df = df.dropna(subset=['nnz','n'], how='any')
    if df.empty:
        print(f"No data left after dropping NaNs in {out_png}")
        return
    
    # Compute X
    if P == 0:
        print(f"P=0 => skipping binning in {out_png}")
        return
    df['X'] = (df['nnz'] / P)*4 + df['n']*4
    
    # Bins
    small_df  = df[df['X'] < l2_cache_bytes]
    medium_df = df[(df['X'] >= l2_cache_bytes) & (df['X'] < 2*l2_cache_bytes)]
    large_df  = df[df['X'] >= 2*l2_cache_bytes]
    
    subsets = {
        'ALL':    df,
        'SMALL':  small_df,
        'MEDIUM': medium_df,
        'LARGE':  large_df
    }
    
    # 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(12,10))
    fig.suptitle(f"Binned Perf Profiles (P={P}, L2={l2_cache_bytes})")
    
    order = [('ALL',0,0), ('SMALL',0,1), ('MEDIUM',1,0), ('LARGE',1,1)]
    for (subset_name,r,c) in order:
        ax = axes[r,c]
        subdf = subsets[subset_name]
        if subdf.empty:
            ax.text(0.5,0.5,f"No data for {subset_name}",ha='center',va='center')
            ax.set_xlim(1,2)
            ax.set_ylim(0,1)
            continue
        
        best = subdf[reorderings].max(axis=1)
        for mth in reorderings:
            series = subdf[mth]
            if series.isna().all():
                continue
            x, y = performance_profile(series, best)
            ax.step(x, y, where='post', label=mth)
        
        ax.set_xlim(1,2)
        ax.set_ylim(0,1.05)
        ax.set_title(f"{subset_name} (N={len(subdf)})")
        ax.grid(True)
    
    # Put one legend in bottom-right
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created {out_png}")

def main(indir):
    csv_files = glob.glob(os.path.join(indir,"*.csv"))
    
    for csv_file in csv_files:
        # Parse machine, nth from filename
        machine, nth = parse_machine_and_nth(csv_file)
        if machine is None or nth is None:
            print(f"Cannot parse machine/nth from {csv_file}, skipping.")
            continue
        
        if machine not in MACHINE_L2_CACHE:
            print(f"Machine {machine} not in L2 dict, skipping.")
            continue
        l2_cache = MACHINE_L2_CACHE[machine]
        
        # Decide on P = nth or nth+1
        P = nth  # or nth+1, depending on how you define it
        
        df = pd.read_csv(csv_file)
        
        # We keep nnz, n for binning. Exclude 'IOU' if present
        if 'IOU' in df.columns:
            df = df.drop(columns='IOU')
        
        # The reorderings = all numeric columns except descriptors
        # But we need 'nnz','n' for binning, so exclude them from reorderings
        descriptor_cols = {'matrix_name','m','k','nnz','n','X','method','machine','nth'}
        reorderings = [c for c in df.columns if c not in descriptor_cols]
        
        out_png = csv_file.replace(".csv","_binned.png")
        plot_binned_performance_profiles(df, reorderings, l2_cache, P, out_png)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", default="output_pivoted", help="Directory of pivoted CSVs")
    args = parser.parse_args()
    main(args.indir)
