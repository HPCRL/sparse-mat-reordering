#!/usr/bin/env python3
"""
draw_binned_grid.py

Generates performance-profile plots in a grid of R rows x C columns,
where R is either 2 or 3 (controlled by `--rowmode`), and C is either 4
(cache-based binning) or 11 (density-based binning).

We can draw either:
- 3 rows => nth in [1, cores//2, cores-1]
- 2 rows => nth in [1, cores-1]

We skip the group if we don't have pivoted CSVs for those exact nth values.
Results are saved as PNGs in the specified --outdir.

Usage Examples:
  # 1) Cache-based binning, 3 rows (1, half_cores, cores-1):
  python draw_binned_grid.py --indir output_pivoted --outdir figures_out \
                             --binning cacheX --rowmode 3row

  # 2) Density-based binning, 2 rows (1, cores-1) reading average-block file:
  python draw_binned_grid.py --indir output_pivoted --outdir figures_out \
                             --binning density --avgblock matrix_avgblock.csv --rowmode 2row
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict

##############################################################################
# 1) GLOBAL REORDERINGS LIST
##############################################################################
# If None: use ALL reorderings in the CSV.
# Otherwise: keep only these if they exist.
ALLOWED_REORDERINGS = None
# Example:
# ALLOWED_REORDERINGS = ["Louvain", "RCM", "baseline", "METIS", "PaToH", "Gorder", "Grappolo"]
# ALLOWED_REORDERINGS = ["Grappolo", "baseline"]

##############################################################################
# 2) MACHINE INFO & BIN DEFINITIONS
##############################################################################

MACHINE_NUM_CORES = {
    "athena":        18,
    "AMD-Desktop":   8,
    "intel-Desktop": 8,
    "zeus":          64
}

MACHINE_L2_CACHE = {
                "athena":        1024 * 1024,
                "AMD-Desktop":   512 * 1024,
                "intel-Desktop": 512 * 1024,
                "zeus":          512 * 1024
}

# For cache-based binning => 4 columns: [ALL, SMALL, MEDIUM, LARGE]
# For density-based binning => 1 + 10 columns: ["ALL"] + [0.01..0.1, inf)
DENSITY_BINS = [
    0.01, 0.04, 0.08, 0.12, 0.16, float("inf")
]
DENSITY_LABELS = [
    "0.01-0.04",
    "0.04-0.08",
    "0.08-0.12",
    "0.12-0.16",
    ">0.16"
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
    
    method_submethod = parts[0] + "_" + parts[1]  # e.g. "spmv_basic" or "spmm_gcn"
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
      X = (nnz / P)*4 + (n * 4)
      => SMALL: X < l2_bytes
         MEDIUM: l2_bytes <= X < 2*l2_bytes
         LARGE: X >= 2*l2_bytes
      + an ALL bin.
    """
    if "nnz" not in df.columns or "n" not in df.columns:
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
# 5) GRID PLOTTING
##############################################################################

def plot_binned_grid(
    nth_csv_map,
    out_png,
    binning_mode="cacheX",
    avgblock_df=None
):
    """
    nth_csv_map: dict of {nth_value: path_to_csv}, typically 2 or 3 distinct nth.
    binning_mode: 'cacheX' or 'density'
    avgblock_df: merged with pivoted df if binning_mode='density'

    We produce R=2 or 3 rows, depending on len(nth_csv_map).
    C=4 if cacheX, or C=11 if density.
    """
    nth_list = sorted(nth_csv_map.keys())
    row_count = len(nth_list)
    
    if binning_mode == "cacheX":
        bin_order = ["ALL", "SMALL", "MEDIUM", "LARGE"]
        col_count = 4
    else:
        bin_order = ["ALL"] + DENSITY_LABELS
        col_count = len(bin_order)
    
    fig, axes = plt.subplots(nrows=row_count, ncols=col_count,
                             figsize=(4*col_count, 3*row_count+2))
    fig.suptitle(f"{os.path.basename(out_png)}", fontsize=14)
    
    handles_labels = None
    
    for r_index, nth_val in enumerate(nth_list):
        csv_file = nth_csv_map[nth_val]
        df = pd.read_csv(csv_file)
        
        _, machine, nth_parsed, _ = parse_pivoted_filename(csv_file)
        
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
        
        # Bin the data
        if binning_mode == "cacheX":
            # 512k if missing
            l2_cache = MACHINE_L2_CACHE.get(machine, 512*1024)
            
            P = nth_val if nth_val else 1
            subsets = bin_df_cacheX(df, P, l2_cache)
        else:
            subsets = bin_df_avg_density(df, avg_col="avg")
        
        for c_index, bin_name in enumerate(bin_order):
            # Handle potential 1D axes if row_count=1 or col_count=1
            if row_count == 1 and col_count == 1:
                ax = axes
            elif row_count == 1:
                ax = axes[c_index]
            elif col_count == 1:
                ax = axes[r_index]
            else:
                ax = axes[r_index, c_index]
            
            if bin_name not in subsets:
                ax.text(0.5,0.5,f"No bin {bin_name}",ha="center",va="center")
                ax.set_title(f"nth={nth_val}, {bin_name} (N=0)")
                continue
            
            subdf = subsets[bin_name]
            n_matrices = len(subdf)
            
            if subdf.empty or len(reorderings) == 0:
                ax.text(0.5, 0.5, f"No data", ha="center", va="center")
                ax.set_xlim(1,2)
                ax.set_ylim(0,1)
                ax.set_title(f"nth={nth_val}, {bin_name} (N={n_matrices})")
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
            ax.set_title(f"nth={nth_val}, {bin_name} (N={n_matrices})")
            
            hl = ax.get_legend_handles_labels()
            if len(hl[0]) > 0:
                handles_labels = hl
    
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
    parser.add_argument("--indir",  default="output_pivoted",
                        help="Directory with pivoted CSVs")
    parser.add_argument("--outdir", default="figures_out",
                        help="Directory to store the PNG files")
    parser.add_argument("--binning", choices=["cacheX","density"],
                        default="cacheX",
                        help="Which binning strategy to use")
    parser.add_argument("--avgblock", default=None,
                        help="CSV file with matrix_name,avg (for density binning)")
    parser.add_argument("--rowmode", choices=["2row","3row"],
                        default="3row",
                        help="How many nth rows to plot: '3row' => [1, half, cores-1], '2row' => [1, cores-1].")
    args = parser.parse_args()

    # Ensure outdir exists
    os.makedirs(args.outdir, exist_ok=True)

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
    
    # We'll group them by (machine, method_submethod, nVal)
    groups = defaultdict(dict)
    # i.e. groups[(mach, ms_sub, n_val)][nth] = csvfile

    for f in all_csvs:
        ms_sub, mach, nth, n_val = parse_pivoted_filename(f)
        if not ms_sub or not mach or nth is None:
            continue
        groups[(mach, ms_sub, n_val)][nth] = f
    
    # Decide which nth values we want per machine
    def desired_nth_list(machine, mode):
        if machine not in MACHINE_NUM_CORES:
            return []
        cores = MACHINE_NUM_CORES[machine]
        if mode == "3row":
            return [1, cores//2, cores-1]
        else:
            # "2row"
            return [1, cores-1]

    for (mach, ms_sub, n_val), nth_map in groups.items():
        dnths = desired_nth_list(mach, args.rowmode)
        missing = [d for d in dnths if d not in nth_map]
        if missing:
            print(f"Group {(mach, ms_sub, n_val)} missing nth {missing}, skipping.")
            continue
        
        # Build output name
        n_part = f"n{n_val}" if n_val is not None else "nNone"
        out_name = f"{ms_sub}_{mach}_{n_part}_{args.binning}_{args.rowmode}Grid.png"
        out_png = os.path.join(args.outdir, out_name)
        
        # Sub-map for the chosen nth
        chosen_map = {nth: nth_map[nth] for nth in dnths}
        
        plot_binned_grid(chosen_map, out_png, args.binning, avgblock_df)

if __name__ == "__main__":
    main()
