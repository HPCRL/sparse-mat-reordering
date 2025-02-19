"""
draw_binned_grid.py

Generates performance-profile plots in a 3-row x N-column grid
(for 3 distinct nth values) with either:
- cache-based binning (4 columns: ALL, SMALL, MEDIUM, LARGE), or
- density-based binning (11 columns: ALL + 10 density bins).

We also have a global ALLOWED_REORDERINGS variable to control
which reorderings to include in the performance profiles.

Usage Examples:
  # 1) Cache-based binning, all reorderings
  python draw_binned_grid.py --indir output_pivoted --binning cacheX

  # 2) Density-based binning, read average-block file,
  #    restricting reorderings to Gorder, RCM, baseline
  #    (by editing ALLOWED_REORDERINGS at the top)
  python draw_binned_grid.py --indir output_pivoted --binning density --avgblock matrix_avgblock.csv
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

##############################################################################
# 1) GLOBAL REORDERINGS LIST
##############################################################################
# If None: use ALL reorderings in the CSV.
# Otherwise: keep only these if they exist.
# ALLOWED_REORDERINGS = None
# Example:
ALLOWED_REORDERINGS = ["RCM", "Louvain", "METIS", "baseline"]


##############################################################################
# 2) BINS FOR DENSITY, ETC.
##############################################################################

# L2 cache sizes for each machine (in bytes)
MACHINE_L2_CACHE = {
    "athena":        1024 * 1024,  # 1 MB
    "AMD-Desktop":   512 * 1024,
    "intel-Desktop": 512 * 1024,
    "zeus":          512 * 1024
}

# Density bins and labels (excluding [0,0.01), with an "ALL" column appended at runtime)
DENSITY_BINS = [
    0.01, 0.02, 0.03, 0.04, 0.05,
    0.06, 0.07, 0.08, 0.09, 0.1, 
    float("inf")
]
DENSITY_LABELS = [
    "0.01-0.02",
    "0.02-0.03",
    "0.03-0.04",
    "0.04-0.05",
    "0.05-0.06",
    "0.06-0.07",
    "0.07-0.08",
    "0.08-0.09",
    "0.09-0.1",
    ">0.1"
]


##############################################################################
# 3) HELPER FUNCTIONS
##############################################################################

def parse_pivoted_filename(filename):
    """
    Attempt to parse a pivoted CSV name like:
      spmv_basic_zeus_nth8.csv
      spmm_gcn_athena_nth16_n256.csv

    Returns (method_submethod, machine, nth, nVal).

    - method_submethod: e.g. "spmv_basic" or "spmm_gcn"
    - machine: e.g. "zeus", "athena", ...
    - nth: integer (or None if not found)
    - nVal: integer (or None if not found)
    """
    base = os.path.basename(filename)
    parts = base.split("_")
    if len(parts) < 3:
        return (None, None, None, None)
    
    method_submethod = parts[0] + "_" + parts[1]  # "spmv_basic" or "spmm_gcn"
    machine = parts[2]
    
    nth = None
    n_val = None
    
    for p in parts[3:]:
        m_nth = re.match(r"nth(\d+)", p)
        if m_nth:
            nth = int(m_nth.group(1))
            continue
        
        m_n = re.match(r"n(\d+)", p)
        if m_n:
            n_val = int(m_n.group(1))
            continue
    
    return (method_submethod, machine, nth, n_val)

def performance_profile(method_times, best_times):
    """
    bigger-is-better => ratio = best_times / method_times
    """
    ratio = best_times / method_times
    ratio_sorted = np.sort(ratio.dropna())
    y = np.arange(1, len(ratio_sorted) + 1) / len(ratio_sorted)
    return ratio_sorted, y


##############################################################################
# 4) BINNING
##############################################################################

def bin_df_cacheX(df, P, l2_bytes):
    """
    Cache-based binning:
      X = (nnz / P) * 4 + (n * 4)
      => SMALL: X < l2_bytes
         MEDIUM: l2_bytes <= X < 2*l2_bytes
         LARGE: X >= 2*l2_bytes
      + an ALL bin.
    """
    if "nnz" not in df.columns or "n" not in df.columns:
        # Return minimal bins
        return {
            "ALL": df,
            "SMALL": df.iloc[0:0],
            "MEDIUM": df.iloc[0:0],
            "LARGE": df.iloc[0:0],
        }
    df["nnz"] = pd.to_numeric(df["nnz"], errors="coerce")
    df["n"]   = pd.to_numeric(df["n"], errors="coerce")
    df = df.dropna(subset=["nnz","n"], how="any")
    if df.empty or P == 0:
        return {
            "ALL": df,
            "SMALL": df.iloc[0:0],
            "MEDIUM": df.iloc[0:0],
            "LARGE": df.iloc[0:0],
        }
    
    df["X"] = (df["nnz"] / P)*4 + df["n"]*4
    
    small_df  = df[df["X"] < l2_bytes]
    medium_df = df[(df["X"] >= l2_bytes) & (df["X"] < 2*l2_bytes)]
    large_df  = df[df["X"] >= 2*l2_bytes]
    
    return {
        "ALL": df,
        "SMALL": small_df,
        "MEDIUM": medium_df,
        "LARGE": large_df
    }

def bin_df_avg_density(df, avg_col="avg"):
    """
    Density-based binning:
      - "ALL" => entire DataFrame
      - Then 10 bins from [0.01..0.02) up to [0.1..inf).

    We assume the DataFrame has been merged with the average-block CSV
    so that each row has 'avg' for that matrix.
    """
    subsets = {}
    subsets["ALL"] = df
    
    df[avg_col] = pd.to_numeric(df[avg_col], errors="coerce")
    df = df.dropna(subset=[avg_col])
    if df.empty:
        for label in DENSITY_LABELS:
            subsets[label] = df
        return subsets
    
    df["density_bin"] = pd.cut(
        df[avg_col],
        bins=DENSITY_BINS,
        right=False,
        labels=DENSITY_LABELS,
        include_lowest=True
    )
    
    for label in DENSITY_LABELS:
        subsets[label] = df[df["density_bin"] == label]
    
    return subsets


##############################################################################
# 5) PLOT A 3-ROW GRID
##############################################################################
def plot_binned_grid(
    csv_group,
    out_png,
    binning_mode="cacheX",
    avgblock_df=None
):
    """
    csv_group: list of pivoted CSV files for the same (machine, method_submethod, nVal),
               each with a distinct nth. We want exactly 3 nth => 3 rows.

    binning_mode:
      - "cacheX": 4 columns => [ALL, SMALL, MEDIUM, LARGE]
      - "density": 11 columns => [ALL] + DENSITY_LABELS
        (the first column is "ALL", next 10 are density bins)

    avgblock_df: used if binning_mode="density", merged by "matrix_name".
    """
    def get_nth(f):
        _, _, nth, _ = parse_pivoted_filename(f)
        return nth if nth is not None else 0
    csv_group_sorted = sorted(csv_group, key=get_nth)
    
    if binning_mode == "cacheX":
        bin_order = ["ALL", "SMALL", "MEDIUM", "LARGE"]
        ncols = 4
    else:
        # density => "ALL" + 10 bins
        bin_order = ["ALL"] + DENSITY_LABELS
        ncols = len(bin_order)
    
    # We'll have 3 rows (for 3 nth values)
    fig, axes = plt.subplots(nrows=3, ncols=ncols, figsize=(4*ncols, 10))
    fig.suptitle(f"{os.path.basename(out_png)}", fontsize=14)
    
    handles_labels = None
    
    for row_idx, csv_file in enumerate(csv_group_sorted):
        # Parse nth
        _, machine, nth, _ = parse_pivoted_filename(csv_file)
        
        # Read pivoted CSV
        df = pd.read_csv(csv_file)
        
        # Exclude IOU if present
        if "IOU" in df.columns:
            df = df.drop(columns=["IOU"])
        
        # If density binning, merge with avgblock
        if binning_mode == "density" and avgblock_df is not None:
            df = pd.merge(df, avgblock_df, on="matrix_name", how="left")
        
        # Identify reorderings => numeric columns except descriptors
        descriptor_cols = {
            "matrix_name","m","k","nnz","n","X",
            "method","machine","nth","density_bin","avg"
        }
        reorderings = [c for c in df.columns if c not in descriptor_cols]
        
        # Filter reorderings if ALLOWED_REORDERINGS is set
        if ALLOWED_REORDERINGS is not None:
            reorderings = [r for r in reorderings if r in ALLOWED_REORDERINGS]
        
        # Convert reorderings to numeric
        df[reorderings] = df[reorderings].apply(pd.to_numeric, errors="coerce")
        
        # Bin
        if binning_mode == "cacheX":
            l2_cache = MACHINE_L2_CACHE.get(machine, 512*1024)
            P = nth if nth else 1
            subsets = bin_df_cacheX(df, P, l2_cache)
        else:
            # "density"
            subsets = bin_df_avg_density(df, avg_col="avg")
        
        for col_idx, bin_name in enumerate(bin_order):
            ax = axes[row_idx, col_idx] if ncols > 1 else axes[row_idx]
            
            if bin_name not in subsets:
                ax.text(0.5,0.5,f"No bin {bin_name}",ha="center",va="center")
                ax.set_title(f"nth={nth}, {bin_name} (N=0)")
                continue
            
            subdf = subsets[bin_name]
            n_matrices = len(subdf)
            
            if subdf.empty or len(reorderings) == 0:
                ax.text(0.5, 0.5, f"No data", ha="center", va="center")
                ax.set_xlim(1,2)
                ax.set_ylim(0,1)
                ax.set_title(f"nth={nth}, {bin_name} (N={n_matrices})")
                continue
            
            # Compute best among these reorderings
            best_times = subdf[reorderings].max(axis=1)
            
            for rname in reorderings:
                series = subdf[rname]
                if series.isna().all():
                    continue
                x, y = performance_profile(series, best_times)
                ax.step(x, y, where="post", label=rname)
            
            ax.set_xlim(1,2)
            ax.set_ylim(0,1.05)
            ax.grid(True)
            ax.set_title(f"nth={nth}, {bin_name} (N={n_matrices})")
            
            hl = ax.get_legend_handles_labels()
            if len(hl[0]) > 0:
                handles_labels = hl
    
    # Single legend for entire figure
    if handles_labels:
        handles, labels = handles_labels
        fig.legend(handles, labels, loc="lower right")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Created", out_png)


##############################################################################
# 6) MAIN
##############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", default="output_pivoted",
                        help="Directory with pivoted CSVs")
    parser.add_argument("--binning", choices=["cacheX","density"],
                        default="cacheX",
                        help="Which binning strategy to use")
    parser.add_argument("--avgblock", default=None,
                        help="CSV file with matrix_name,avg (for density binning)")
    args = parser.parse_args()

    # If density binning, load the average block density file
    avgblock_df = None
    if args.binning == "density":
        if not args.avgblock:
            print("Error: --avgblock is required for density binning.")
            return
        avgblock_df = pd.read_csv(args.avgblock)
        if "matrix_name" not in avgblock_df.columns or "avg" not in avgblock_df.columns:
            print("Error: avgblock CSV must have columns [matrix_name, avg].")
            return
    
    # Gather pivoted CSVs
    all_csvs = glob.glob(os.path.join(args.indir, "*.csv"))
    
    # Group by (machine, method_submethod, nVal)
    groups = {}
    for f in all_csvs:
        ms_sub, mach, nth, n_val = parse_pivoted_filename(f)
        if not ms_sub or not mach or nth is None:
            continue
        group_key = (mach, ms_sub, n_val)
        groups.setdefault(group_key, []).append(f)
    
    # We want exactly 3 nth per group => 3 rows
    for (mach, ms_sub, n_val), csv_list in groups.items():
        nth_values = set()
        for cf in csv_list:
            _, _, nth, _ = parse_pivoted_filename(cf)
            nth_values.add(nth)
        
        if len(nth_values) != 3:
            print(f"Group {(mach, ms_sub, n_val)} has {nth_values} != 3 nth. Skipping.")
            continue
        
        n_part = f"n{n_val}" if n_val is not None else "nNone"
        out_name = f"{ms_sub}_{mach}_{n_part}_{args.binning}Grid.png"
        out_png = os.path.join(args.indir, out_name)
        
        plot_binned_grid(csv_list, out_png, args.binning, avgblock_df)


if __name__ == "__main__":
    main()
