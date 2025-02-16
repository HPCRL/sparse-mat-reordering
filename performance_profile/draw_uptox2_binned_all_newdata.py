import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re

# -------------------------------------------------------------------------
# Configure your machine -> L2 cache size (bytes)
# (Here 1024 KB = 1,048,576 bytes, 512 KB = 524,288 bytes)
# -------------------------------------------------------------------------
MACHINE_L2_CACHE = {
    "athena":       1024 * 1024,  # 1 MB
    "AMD-Desktop":  512 * 1024,
    "intel-Desktop":512 * 1024,
    "zeus":         512 * 1024
}

# -------------------------------------------------------------------------
# Helper function: parse machine and nth from filename
# Assumes your files are named like: "spmv_basic_<machine>_nth<NN>.csv"
# e.g., "spmv_basic_athena_nth16.csv"
# Feel free to adapt if your naming scheme is different.
# -------------------------------------------------------------------------
def parse_machine_and_nth(filename):
    """
    Attempt to parse <machine> and nth<NN> from the base filename.
    Returns (machine, nth_int).
    If something fails, returns (None, None).
    """
    base = os.path.basename(filename)  # e.g. spmv_basic_athena_nth16.csv
    parts = base.split('_')
    # Typical structure: [spmv|spmm, submethod, machineName, nthXX.csv]
    if len(parts) < 4:
        return None, None
    
    machine = parts[2]  # e.g. "athena", "zeus", "AMD-Desktop", or "intel-Desktop"
    
    nth_part = parts[3]  # e.g. "nth16.csv"
    # Extract the digits from "nth16.csv"
    match = re.search(r'nth(\d+)', nth_part)
    if not match:
        return machine, None
    nth = int(match.group(1))  # e.g. 16
    
    return machine, nth

# -------------------------------------------------------------------------
# Performance profile function
# "Bigger is better" (GFLOPs). So ratio = best_times / method_times
# -------------------------------------------------------------------------
def performance_profile(method_times, best_times):
    """
    Given arrays of GFLOPs for one method (method_times)
    and the per-problem best GFLOPs (best_times),
    compute the sorted ratio distribution and fraction of problems (CDF).
    ratio = best_times / method_times
    """
    ratio = best_times / method_times
    ratio_sorted = np.sort(ratio.dropna())  # discard NaNs
    y = np.arange(1, len(ratio_sorted) + 1) / len(ratio_sorted)
    return ratio_sorted, y

# -------------------------------------------------------------------------
# Plot routine: we create subplots for [small, medium, large, all]
# -------------------------------------------------------------------------
def plot_binned_performance_profiles(df, reorderings, l2_cache_bytes, P, out_png):
    """
    df: pivoted DataFrame with numeric reorderings + columns 'nnz', 'n'
    reorderings: columns (GFLOPs) to consider (exclude descriptor columns)
    l2_cache_bytes: L2 size in bytes
    P: number of threads (nth)
    out_png: path to save the figure
    
    This function:
     1) computes X = (nnz/P)*4 + n*4 for each row
     2) bins them into [small, medium, large]
     3) plots 4 subplots: all, small, medium, large
    """
    # Ensure reorderings are numeric (coerce to NaN if not)
    df[reorderings] = df[reorderings].apply(pd.to_numeric, errors='coerce')
    # We'll also coerce nnz, n to numeric
    df['nnz'] = pd.to_numeric(df['nnz'], errors='coerce')
    df['n']   = pd.to_numeric(df['n'], errors='coerce')
    
    # Drop rows where nnz or n is missing
    df = df.dropna(subset=['nnz', 'n'], how='any')
    
    # Compute X for each row
    # X = (nnz / P)*4 + n*4
    # 4 bytes per nonzero (float) and 4 bytes per float in the vector
    # If P=0 for some reason, you might want to skip or handle that
    if P == 0:
        print(f"Warning: P = 0. Skipping binning in {out_png}")
        return
    
    df['X'] = (df['nnz'] / P) * 4 + df['n'] * 4
    
    # Define bins
    small_df  = df[df['X'] < l2_cache_bytes]
    medium_df = df[(df['X'] >= l2_cache_bytes) & (df['X'] < 2*l2_cache_bytes)]
    large_df  = df[df['X'] >= 2*l2_cache_bytes]
    
    subsets = {
        'ALL':    df,
        'SMALL':  small_df,
        'MEDIUM': medium_df,
        'LARGE':  large_df
    }
    
    # Create a 2 x 2 figure for subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Performance Profiles (bins) - P={P}, L2={l2_cache_bytes} bytes')
    
    # We'll iterate in a fixed order: ALL, SMALL, MEDIUM, LARGE
    # so the layout is consistent
    order = [('ALL', 0,0), ('SMALL', 0,1), ('MEDIUM', 1,0), ('LARGE', 1,1)]
    
    for (subset_name, r, c) in order:
        ax = axes[r, c]
        subdf = subsets[subset_name]
        
        # If no data in this subset, just annotate
        if subdf.empty:
            ax.text(0.5, 0.5, f"No data for {subset_name}", ha='center', va='center')
            ax.set_xlim(1, 2)
            ax.set_ylim(0, 1)
            continue
        
        # For each reordering, compute performance profile
        # row-wise best
        best_times = subdf[reorderings].max(axis=1)
        
        for method_name in reorderings:
            series = subdf[method_name]
            if series.isna().all():
                # skip empty
                continue
            
            x, y = performance_profile(series, best_times)
            ax.step(x, y, where='post', label=method_name)
        
        ax.set_xlim(1, 2)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{subset_name} (N={len(subdf)})")
        ax.grid(True)
    
    # Put legend in bottom-right of the figure
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right')
    
    plt.tight_layout()
    # Adjust for suptitle
    plt.subplots_adjust(top=0.92)
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Created {out_png}")

# -------------------------------------------------------------------------
# Main routine
# -------------------------------------------------------------------------
def main():
    # Glob all your pivoted CSVs
    csv_files = glob.glob("pivoted_data/*.csv")
    
    for csv_file in csv_files:
        # Parse machine and nth from the filename
        machine, nth = parse_machine_and_nth(csv_file)
        
        if not machine or nth is None:
            print(f"Could not parse machine/nth from {csv_file}; skipping.")
            continue
        
        # Look up the L2 cache size
        if machine not in MACHINE_L2_CACHE:
            print(f"Machine {machine} not in MACHINE_L2_CACHE dict. Skipping {csv_file}.")
            continue
        l2_cache = MACHINE_L2_CACHE[machine]
        
        # Set P = nth
        P = nth
        
        # Read the CSV
        df = pd.read_csv(csv_file)
        
        # We expect 'nnz' and 'n' to be present; if not, we can't do binning
        if 'nnz' not in df.columns or 'n' not in df.columns:
            print(f"CSV {csv_file} missing 'nnz' or 'n' columns - cannot do binning. Skipping.")
            continue
        
        # Possibly drop 'IOU' if you do not want to include it
        if 'IOU' in df.columns:
            df = df.drop(columns=['IOU'])
        
        # Identify reorderings = all columns except descriptors
        descriptor_cols = {'matrix_name','m','k','nnz','n','X','method','machine','nth'}
        reorderings = [c for c in df.columns if c not in descriptor_cols]
        
        # Now we do the multi-subplot performance profile for bins
        out_png = csv_file.replace('.csv', '_binned.png')
        plot_binned_performance_profiles(df, reorderings, l2_cache, P, out_png)

if __name__ == "__main__":
    main()
