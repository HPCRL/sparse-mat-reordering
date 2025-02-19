import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

##############################################################################
# Configuration
##############################################################################

# Dictionary of machine -> L2 cache size in bytes
MACHINE_L2_CACHE = {
    "athena":        1024 * 1024,   # 1 MB
    "AMD-Desktop":   512 * 1024,
    "intel-Desktop": 512 * 1024,
    "zeus":          512 * 1024
}

# Fixed bin edges/labels for average block density
DENSITY_BINS = [0.01, 0.02, 0.03, 0.04, 0.05,
                0.06, 0.07, 0.08, 0.09, 0.1, float("inf")]
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
# Helper Functions
##############################################################################

def parse_pivoted_filename(filename):
    """
    Attempt to parse a pivoted CSV name like:
      spmv_basic_zeus_nth8.csv
      spmm_gcn_athena_nth16_n256.csv
    Returns (method_submethod, machine, nth, nVal).

    - method_submethod: e.g. "spmv_basic", "spmm_gcn"
    - machine: e.g. "zeus", "athena", ...
    - nth: integer (or None if not found)
    - nVal: integer (or None if not found)
    """
    base = os.path.basename(filename)
    # Examples:
    # spmv_basic_zeus_nth8.csv => ["spmv","basic","zeus","nth8.csv"]
    # spmm_gcn_athena_nth16_n256.csv => ["spmm","gcn","athena","nth16","n256.csv"]
    parts = base.split("_")
    if len(parts) < 3:
        return (None, None, None, None)
    
    method_submethod = parts[0] + "_" + parts[1]  # e.g. "spmv_basic" or "spmm_gcn"
    machine = parts[2]
    
    nth = None
    n_val = None
    
    # Check leftover parts for "nthXX" and "nYY"
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
# Binning: Cache-based (original approach)
##############################################################################
def bin_df_cacheX(df, P, l2_bytes):
    """
    Bin by X = (nnz / P)*4 + (n * 4):
      - ALL: entire df
      - SMALL: X < l2_bytes
      - MEDIUM: l2_bytes <= X < 2*l2_bytes
      - LARGE: X >= 2*l2_bytes
    Returns a dict: {"ALL": dfAll, "SMALL": dfSmall, ...}
    """
    # Must have nnz, n
    if "nnz" not in df.columns or "n" not in df.columns:
        # Return an empty bin partition except "ALL"
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
    
    df["X"] = (df["nnz"] / P) * 4 + df["n"] * 4
    
    small_df  = df[df["X"] < l2_bytes]
    medium_df = df[(df["X"] >= l2_bytes) & (df["X"] < 2*l2_bytes)]
    large_df  = df[df["X"] >= 2*l2_bytes]
    
    return {
        "ALL": df,
        "SMALL": small_df,
        "MEDIUM": medium_df,
        "LARGE": large_df
    }

##############################################################################
# Binning: Average Block Density
##############################################################################
def bin_df_avg_density(df, avg_col="avg"):
    """
    Bin by the 'avg' column (average block density).
    We omit the [0,0.01) bin and start from 0.01, plus an "ALL" column first.
    
    Returns a dict with:
      - "ALL": all rows
      - and 10 bins from [0.01, 0.02) up to [0.1, inf).
    """
    if avg_col not in df.columns:
        # If we don't have the 'avg' column, just return "ALL" as everything
        # and empty dataframes for the 10 bins
        out = {"ALL": df}
        for label in DENSITY_LABELS:
            out[label] = df.iloc[0:0]  # empty
        return out

    # Convert avg to numeric and drop missing
    df[avg_col] = pd.to_numeric(df[avg_col], errors="coerce")
    df = df.dropna(subset=[avg_col])
    
    # Always put all rows in an "ALL" bin
    subsets = {}
    subsets["ALL"] = df
    
    if df.empty:
        # If everything is gone, the other bins are empty
        for label in DENSITY_LABELS:
            subsets[label] = df
        return subsets
    
    # Now do the standard binning for [0.01..0.02), etc.
    df["density_bin"] = pd.cut(
        df[avg_col],
        bins=DENSITY_BINS,
        right=False,       # [0.01,0.02) excludes 0.02
        labels=DENSITY_LABELS,
        include_lowest=True
    )
    
    # Gather sub-DataFrames
    for label in DENSITY_LABELS:
        subsets[label] = df[df["density_bin"] == label]
    
    return subsets


##############################################################################
# Plot a 3-row grid
##############################################################################
def plot_binned_grid(
    csv_group,
    out_png,
    binning_mode="cacheX",
    avgblock_df=None
):
    """
    csv_group: list of pivoted CSV files for the same (machine, method_submethod, n),
               each with a different nth. We want 3 distinct nth => 3 rows.

    binning_mode:
      - "cacheX": do the old X-based binning -> 4 columns
      - "density": do the average-block-density binning -> 11 columns

    avgblock_df: DataFrame with columns ["matrix_name","avg"] if binning_mode="density".
                 We'll merge pivoted DF with this data (on "matrix_name").
    """
    # Sort the CSVs by nth
    def get_nth(f):
        _, _, nth, _ = parse_pivoted_filename(f)
        return nth if nth is not None else 0
    csv_group_sorted = sorted(csv_group, key=get_nth)
    
    # Decide how many columns, bin order, etc.
    if binning_mode == "cacheX":
        bin_order = ["ALL", "SMALL", "MEDIUM", "LARGE"]
        ncols = 4
    else:
        # "density"
        # bin_order = DENSITY_LABELS  # 11 labels
        bin_order = ["ALL"] + DENSITY_LABELS  # insert "ALL" at the beginning
        ncols = len(bin_order)  # 1 + 10 = 11
    
    # Create figure: 3 rows (for nth) x N columns (for bins)
    fig, axes = plt.subplots(nrows=3, ncols=ncols, figsize=(4*ncols, 10))
    fig.suptitle(f"{os.path.basename(out_png)}", fontsize=14)
    
    # We'll collect handles/labels for the legend
    handles_labels = None
    
    for row_idx, csv_file in enumerate(csv_group_sorted):
        # Parse nth from filename
        _, machine, nth, _ = parse_pivoted_filename(csv_file)
        
        # Read pivoted data
        df = pd.read_csv(csv_file)
        
        # Exclude IOU if present
        if "IOU" in df.columns:
            df = df.drop(columns=["IOU"])
        
        # Merge with avgblock_df if in density mode
        if binning_mode == "density" and avgblock_df is not None:
            # We assume both have 'matrix_name' col
            df = pd.merge(df, avgblock_df, on="matrix_name", how="left")
            # Now df has a column "avg" for density
        
        # Identify reorderings => numeric columns except descriptors
        descriptor_cols = {
            "matrix_name","m","k","nnz","n","X",
            "method","machine","nth","density_bin","avg"
        }
        reorderings = [c for c in df.columns if c not in descriptor_cols]
        
        # Convert reorderings to numeric
        df[reorderings] = df[reorderings].apply(pd.to_numeric, errors="coerce")
        
        # Bin
        if binning_mode == "cacheX":
            l2_cache = MACHINE_L2_CACHE.get(machine, 512*1024)  # fallback
            P = nth if nth else 1  # or nth+1 if you prefer
            subsets = bin_df_cacheX(df, P, l2_cache)
        else:
            # "density"
            subsets = bin_df_avg_density(df, avg_col="avg")
        
        # For each bin in bin_order, plot
        for col_idx, bin_name in enumerate(bin_order):
            if ncols == 1:
                # if somehow only 1 col, `axes[row_idx]` is 1D
                ax = axes[row_idx]
            else:
                ax = axes[row_idx, col_idx]
            
            if bin_name not in subsets:
                # unknown bin -> skip
                ax.text(0.5,0.5,f"No bin {bin_name}",ha="center",va="center")
                ax.set_title(f"nth={nth}, {bin_name} (N=0)")
                continue
            
            subdf = subsets[bin_name]
            n_matrices = len(subdf)
            
            if subdf.empty:
                ax.text(0.5, 0.5, f"No data for {bin_name}", ha="center", va="center")
                ax.set_xlim(1,2)
                ax.set_ylim(0,1)
                ax.set_title(f"nth={nth}, {bin_name} (N={n_matrices})")
                continue
            
            # Compute row-wise best
            best_times = subdf[reorderings].max(axis=1)
            
            # Plot
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
            
            # Collect legend handles/labels
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

    # 1) If doing density binning, load the average block density file
    avgblock_df = None
    if args.binning == "density":
        if not args.avgblock:
            print("Error: --avgblock is required for density binning.")
            return
        # load it
        avgblock_df = pd.read_csv(args.avgblock)
        # Ensure columns: matrix_name, avg
        if "matrix_name" not in avgblock_df.columns or "avg" not in avgblock_df.columns:
            print("Error: avgblock CSV must have columns [matrix_name, avg].")
            return
    
    # 2) Collect all pivoted CSVs in indir
    all_csvs = glob.glob(os.path.join(args.indir, "*.csv"))
    
    # 3) Group them by (machine, method_submethod, n)
    groups = {}
    for f in all_csvs:
        ms_sub, mach, nth, n_val = parse_pivoted_filename(f)
        if not ms_sub or not mach or nth is None:
            continue
        group_key = (mach, ms_sub, n_val)  # e.g. ("zeus", "spmv_basic", None)
        groups.setdefault(group_key, []).append(f)
    
    # 4) For each group, we want to see if it has exactly 3 distinct nth
    for (mach, ms_sub, n_val), csv_list in groups.items():
        nth_values = set()
        for cf in csv_list:
            _, _, nth, _ = parse_pivoted_filename(cf)
            nth_values.add(nth)
        
        if len(nth_values) != 3:
            print(f"Group {(mach, ms_sub, n_val)} has {nth_values} != 3 distinct nth. Skipping.")
            continue
        
        # Build output name
        n_part = f"n{n_val}" if n_val is not None else "nNone"
        # e.g. "spmv_basic_zeus_nNone_densityGrid.png"
        out_name = f"{ms_sub}_{mach}_{n_part}_{args.binning}Grid.png"
        out_png = os.path.join(args.indir, out_name)
        
        plot_binned_grid(
            csv_list, 
            out_png,
            binning_mode=args.binning,
            avgblock_df=avgblock_df
        )

if __name__ == "__main__":
    main()
