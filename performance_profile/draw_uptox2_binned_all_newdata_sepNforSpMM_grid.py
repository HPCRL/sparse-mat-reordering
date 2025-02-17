import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Dictionary of machine -> L2 cache size in bytes
MACHINE_L2_CACHE = {
    "athena":        1024 * 1024,  # 1 MB
    "AMD-Desktop":   512 * 1024,
    "intel-Desktop": 512 * 1024,
    "zeus":          512 * 1024
}

def parse_pivoted_filename(filename):
    """
    Attempt to parse a pivoted CSV name like:
      spmv_basic_zeus_nth8.csv
      spmv_basic_zeus_nth16.csv
      spmm_gcn_athena_nth8_n256.csv
    Returns (method_submethod, machine, nth, n).

    - method_submethod: e.g. "spmv_basic", "spmm_gcn"
    - machine: e.g. "zeus", "athena"
    - nth: integer (or None if not found)
    - n: integer (or None if not found)
    """
    base = os.path.basename(filename)
    # Example: spmv_basic_zeus_nth8.csv  => parts = ["spmv", "basic", "zeus", "nth8.csv"]
    # Example: spmm_gcn_athena_nth8_n256.csv => parts = ["spmm", "gcn", "athena", "nth8", "n256.csv"]
    parts = base.split("_")
    if len(parts) < 3:
        # not a valid naming
        return None, None, None, None
    
    # method_submethod = spmv_basic or spmm_gcn
    method_submethod = parts[0] + "_" + parts[1]  # "spmv_basic" or "spmm_gcn"
    
    machine = parts[2]
    
    # Next we look for "nthXX" in the subsequent chunk(s)
    nth = None
    n_val = None
    
    # The rest: parts[3], parts[4], ...
    # For spmv: might be ["nth8.csv"]
    # For spmm: might be ["nth8", "n256.csv"]
    
    # If there's a part that starts with "nth", parse the digits
    for p in parts[3:]:
        m_nth = re.match(r"nth(\d+)", p)
        if m_nth:
            nth = int(m_nth.group(1))
            continue
        
        # Also check if there's a part that starts with "n" for dimension
        m_n = re.match(r"n(\d+)", p)
        if m_n:
            n_val = int(m_n.group(1))
            continue
    
    return method_submethod, machine, nth, n_val

def performance_profile(method_times, best_times):
    """
    bigger-is-better => ratio = best_times / method_times
    """
    ratio = best_times / method_times
    ratio_sorted = np.sort(ratio.dropna())
    y = np.arange(1, len(ratio_sorted) + 1) / len(ratio_sorted)
    return ratio_sorted, y

def bin_df(df, P, l2_bytes):
    """
    Compute X = (nnz/P)*4 + n*4, and split the DataFrame into bins.
    Returns dict with keys: ALL, SMALL, MEDIUM, LARGE -> sub-DataFrames
    If P=0 or missing nnz/n, returns empty subsets accordingly.
    """
    # Must have nnz, n columns
    if "nnz" not in df.columns or "n" not in df.columns:
        return {
            "ALL": df, 
            "SMALL": df.iloc[0:0],
            "MEDIUM": df.iloc[0:0],
            "LARGE": df.iloc[0:0]
        }
    # Convert to numeric
    df["nnz"] = pd.to_numeric(df["nnz"], errors="coerce")
    df["n"]   = pd.to_numeric(df["n"], errors="coerce")
    # Drop rows missing nnz or n
    df = df.dropna(subset=["nnz","n"], how="any")
    if df.empty or P == 0:
        return {
            "ALL": df, 
            "SMALL": df.iloc[0:0],
            "MEDIUM": df.iloc[0:0],
            "LARGE": df.iloc[0:0]
        }
    # Compute X
    df["X"] = (df["nnz"] / P)*4 + df["n"]*4
    
    # Bins
    small_df  = df[df["X"] < l2_bytes]
    medium_df = df[(df["X"] >= l2_bytes) & (df["X"] < 2*l2_bytes)]
    large_df  = df[df["X"] >= 2*l2_bytes]
    
    return {
        "ALL": df,
        "SMALL": small_df,
        "MEDIUM": medium_df,
        "LARGE": large_df
    }

def plot_binned_grid(csv_group, out_png):
    """
    csv_group: list of filepaths that share the same (machine, method_submethod, n),
               but differ in nth (exactly 3 distinct nth).
    We produce a single figure with 3 rows (nth sorted ascending),
    and 4 columns (ALL, SMALL, MEDIUM, LARGE).
    """
    # Sort the CSVs by nth
    # We'll parse nth from each file name again
    def get_nth(f):
        _, _, nth, _ = parse_pivoted_filename(f)
        return nth if nth is not None else 0
    csv_group_sorted = sorted(csv_group, key=get_nth)
    
    # Create figure: 3 rows × 4 columns
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 10))
    fig.suptitle(f"{os.path.basename(out_png)}", fontsize=14)
    
    # Column order
    bin_order = ["ALL", "SMALL", "MEDIUM", "LARGE"]
    
    # We'll collect handles/labels for the legend
    handles_labels = None
    
    for row_idx, csv_file in enumerate(csv_group_sorted):
        # Parse nth from filename
        _, _, nth, _ = parse_pivoted_filename(csv_file)
        # Read the data
        df = pd.read_csv(csv_file)
        
        # Exclude IOU if present
        if "IOU" in df.columns:
            df = df.drop(columns=["IOU"])
        
        # Identify reorderings => numeric columns except descriptors
        descriptor_cols = {"matrix_name","m","k","nnz","n","X","method","machine","nth"}
        reorderings = [c for c in df.columns if c not in descriptor_cols]
        
        # Convert reorderings to numeric
        df[reorderings] = df[reorderings].apply(pd.to_numeric, errors="coerce")
        
        # Bin
        machine_subm, machine, nth_, n_val = parse_pivoted_filename(csv_file)
        l2_cache = MACHINE_L2_CACHE.get(machine, 512*1024)  # fallback
        P = nth if nth else 1
        
        subsets = bin_df(df, P, l2_cache)
        
        for col_idx, bin_name in enumerate(bin_order):
            ax = axes[row_idx, col_idx]
            subdf = subsets[bin_name]
            # The number of matrices in this bin
            n_matrices = len(subdf)

            if subdf.empty:
                # If we have no data here, just label it
                ax.text(0.5, 0.5, f"No data for {bin_name}", ha="center", va="center")
                ax.set_xlim(1,2)
                ax.set_ylim(0,1)
                ax.set_title(f"nth={nth}, {bin_name}")
                continue
            
            # Compute row-wise best
            best_times = subdf[reorderings].max(axis=1)
            # Plot
            for rname in reorderings:
                series = subdf[rname]
                if series.isna().all():
                    continue
                x, y = performance_profile(series, best_times)
                line_obj, = ax.step(x, y, where="post", label=rname)
            
            ax.set_xlim(1,2)
            ax.set_ylim(0,1.05)
            ax.grid(True)
            ax.set_title(f"nth={nth}, {bin_name}:{n_matrices}")
            
            # Collect legend handles/labels from this subplot
            hl = ax.get_legend_handles_labels()
            if len(hl[0]) > 0:  # if there's something
                handles_labels = hl
    
    # We only put a single legend for the entire figure
    if handles_labels:
        handles, labels = handles_labels
        fig.legend(handles, labels, loc="lower right")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Created", out_png)

def main(input_dir):
    """
    1) Collect all pivoted CSVs in input_dir.
    2) Parse out (method_submethod, machine, nth, n).
    3) Group by (machine, method_submethod, n) so each group has 3 distinct nth.
    4) Generate one big 3x4 figure per group.
    """
    all_csvs = glob.glob(os.path.join(input_dir, "*.csv"))
    
    # We build a dict: groups[(machine, method_submethod, n)] = list of csv_file paths
    groups = {}
    for f in all_csvs:
        ms_sub, mach, nth, n_val = parse_pivoted_filename(f)
        # If something is missing, skip
        if not ms_sub or not mach or nth is None:
            continue
        group_key = (mach, ms_sub, n_val)  # n_val can be None for spmv
        
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(f)
    
    # Now for each group, we expect 3 distinct nth
    for (mach, ms_sub, n_val), csv_list in groups.items():
        # Check if we indeed have 3 CSVs. If not, you can skip or handle gracefully
        nth_values = set()
        for cf in csv_list:
            _, _, nth, _ = parse_pivoted_filename(cf)
            nth_values.add(nth)
        
        # We only proceed if we have exactly 3 distinct nth
        if len(nth_values) != 3:
            print(f"Group {mach}, {ms_sub}, n={n_val} has nth={nth_values} != 3 distinct. Skipping.")
            continue
        
        # Build output name
        # e.g. "spmv_basic_zeus_nNone_binned.png" or "spmm_gcn_athena_n256_binned.png"
        n_part = f"n{n_val}" if n_val is not None else "nNone"
        out_png = os.path.join(input_dir, f"{ms_sub}_{mach}_{n_part}_binnedGrid.png")
        
        plot_binned_grid(csv_list, out_png)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", default="output_pivoted", help="Directory of pivoted CSVs")
    args = parser.parse_args()
    main(args.indir)
