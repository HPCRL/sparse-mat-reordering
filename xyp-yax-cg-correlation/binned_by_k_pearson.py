import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def bin_edges_for_powers_of_two(max_value, start=0):
    """
    Generate bin edges for intervals [0,1), [1,2), [2,4), [4,8), [8,16), ...
    until the edges exceed max_value.

    You can modify or replace this function if you prefer a different scheme.
    """
    edges = []
    # If you want an explicit 0->1 bin
    if start == 0:
        edges = [0, 1]
        current = 1
    else:
        current = 2 ** start
        edges.append(current)

    while edges[-1] < max_value:
        edges.append(edges[-1] * 2)

    return edges

def main():
    # 1. Read CSV
    input_csv = "athena_spmv_cgSpmv_cgAll_baseline_median.csv"  # replace with your actual CSV filename
    df = pd.read_csv(input_csv)
    
    # 2. Keep only the columns of interest
    df = df[
        [
            "matrix_name",
            "reordering",
            "m",
            "k",
            "nnz",
            "nth",
            "median_GFLOPS_yax_basic",
            "median_GFLOPS_yax_xyp_basic",
            "median_GFLOPS_cg_basic_spmv",
        ]
    ]
    
    # 3. Bin the data by k in powers-of-two intervals: [0,1), [1,2), [2,4), [4,8), etc.
    max_k = df['k'].max()
    edges = bin_edges_for_powers_of_two(max_k, start=0)

    # Use pd.cut to bin 'k'. right=False means intervals are [left, right)
    df["k_bin"] = pd.cut(df["k"], bins=edges, right=False)
    
    # 4. Group by (k_bin, nth) and compute the Pearson correlations + count
    results = []
    grouped = df.groupby(["k_bin", "nth"], dropna=True)
    
    for (bin_label, nth_value), group_df in grouped:
        # Count of all rows in this subset
        subset_count = len(group_df)
        
        # For correlation, we specifically need to ensure no NaNs in the
        # column pairs used for correlation. So we handle each pair separately.
        if subset_count < 2:
            # Fewer than 2 data points --> correlation is NaN
            corr_yax = np.nan
            corr_xyp = np.nan
            sub_yax_count = 0
            sub_xyp_count = 0
        else:
            # Pearson correlation: (median_GFLOPS_yax_basic, median_GFLOPS_cg_basic_spmv)
            sub_yax = group_df[["median_GFLOPS_yax_basic", "median_GFLOPS_cg_basic_spmv"]].dropna()
            sub_yax_count = len(sub_yax)
            if sub_yax_count < 2:
                corr_yax = np.nan
            else:
                corr_yax, _ = pearsonr(
                    sub_yax["median_GFLOPS_yax_basic"],
                    sub_yax["median_GFLOPS_cg_basic_spmv"]
                )

            # Pearson correlation: (median_GFLOPS_yax_xyp_basic, median_GFLOPS_cg_basic_spmv)
            sub_xyp = group_df[["median_GFLOPS_yax_xyp_basic", "median_GFLOPS_cg_basic_spmv"]].dropna()
            sub_xyp_count = len(sub_xyp)
            if sub_xyp_count < 2:
                corr_xyp = np.nan
            else:
                corr_xyp, _ = pearsonr(
                    sub_xyp["median_GFLOPS_yax_xyp_basic"],
                    sub_xyp["median_GFLOPS_cg_basic_spmv"]
                )
        
        results.append({
            "k_bin": str(bin_label),  # convert the bin interval to string
            "nth": nth_value,
            "pearson_yax_cg": corr_yax,
            "pearson_xyp_cg": corr_xyp,
            "subset_count_beforeNAdrop": subset_count,
            "yax_cg_subset_count": sub_yax_count,
            "xyp_cg_subset_count": sub_xyp_count
        })
    
    # 5. Create a DataFrame of results and write to CSV
    results_df = pd.DataFrame(results)
    output_csv = "output_correlations.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"Results written to {output_csv}")

if __name__ == "__main__":
    main()
