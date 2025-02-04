import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def bin_edges_for_powers_of_two(max_value, start=0):
    """
    Generate bin edges for intervals [0,1), [1,2), [2,4), [4,8), ...
    until the edges exceed max_value.
    """
    edges = []
    if start == 0:
        edges = [0, 1]
        current = 1
    else:
        current = 2 ** start
        edges.append(current)

    while edges[-1] < max_value:
        edges.append(edges[-1] * 2)

    return edges

def compute_and_save_correlations(
    df,
    group_cols,
    col_pair_1,  # (x_col_1, y_col_1)
    col_pair_2,  # (x_col_2, y_col_2)
    output_csv
):
    """
    1. Bin data by 'k' into powers-of-two intervals.
    2. Group by group_cols (e.g., ["k_bin","nth"] or ["k_bin","nth","n"]).
    3. Compute Pearson correlations for:
       - col_pair_1[0] vs col_pair_1[1]
       - col_pair_2[0] vs col_pair_2[1]
    4. Save results to output_csv.
    """

    if df.empty:
        print(f"DataFrame is empty; no results written for {output_csv}")
        return

    # 1. Bin by k
    max_k = df["k"].max()
    edges = bin_edges_for_powers_of_two(max_k, start=0)
    df = df.copy()  # so we don't mutate the caller's dataframe
    df["k_bin"] = pd.cut(df["k"], bins=edges, right=False)

    # 2. Group by group_cols
    grouped = df.groupby(group_cols, dropna=True)

    results = []
    for group_val, group_df in grouped:
        subset_count = len(group_df)

        # Prepare correlation outputs (init as NaN, update if we have >=2 data points)
        corr_1 = np.nan
        corr_2 = np.nan
        sub_count_1 = 0
        sub_count_2 = 0

        if subset_count >= 2:
            # For the first pair
            sub_df_1 = group_df[[col_pair_1[0], col_pair_1[1]]].dropna()
            sub_count_1 = len(sub_df_1)
            if sub_count_1 >= 2:
                corr_1, _ = pearsonr(sub_df_1[col_pair_1[0]], sub_df_1[col_pair_1[1]])

            # For the second pair
            sub_df_2 = group_df[[col_pair_2[0], col_pair_2[1]]].dropna()
            sub_count_2 = len(sub_df_2)
            if sub_count_2 >= 2:
                corr_2, _ = pearsonr(sub_df_2[col_pair_2[0]], sub_df_2[col_pair_2[1]])

        # Convert group_val (tuple or scalar) into a dict
        if isinstance(group_val, tuple):
            group_info = dict(zip(group_cols, group_val))
        else:
            group_info = {group_cols[0]: group_val}

        group_info.update({
            f"pearson_{col_pair_1[0]}_vs_{col_pair_1[1]}": corr_1,
            f"pearson_{col_pair_2[0]}_vs_{col_pair_2[1]}": corr_2,
            "subset_count_beforeNAdrop": subset_count,
            f"{col_pair_1[0]}_vs_{col_pair_1[1]}_count": sub_count_1,
            f"{col_pair_2[0]}_vs_{col_pair_2[1]}_count": sub_count_2
        })
        results.append(group_info)

    # Build output DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Results written to {output_csv}")

def main():
    # new data: 
    # input_csv = "/media/datassd/sina/spmv-omid/SparseLibs-Characterization/experiments/CPU/spmm-spmv-proxy-comparisons/data/athena-spmm-spmv.csv"
    input_csv = "/media/datassd/sina/spmv-omid/SparseLibs-Characterization/experiments/CPU/spmm-spmv-proxy-comparisons/data/intel-Desktop-spmm-spmv.csv"
    # 1. Read combined data (change the file name to your actual input)
    # input_csv = "combined_spmm_spmv.csv"
    df = pd.read_csv(input_csv)

    # 2. Split into spmv vs spmm
    df_spmv = df[df["method"] == "SpMV"].copy()
    df_spmm = df[df["method"] == "SpMM"].copy()

    # 3. SPMV correlation
    #    Group by [k_bin, nth], but k_bin is handled inside compute_and_save_correlations,
    #    so we only pass ["nth"] here. The function will add k_bin automatically.
    #    We'll compare:
    #      - (median_GFLOPS_yax_spmv_basic) vs (median_GFLOPS_cg_basic_spmv)
    #      - (median_GFLOPS_yax_xyp_spmv_basic) vs (median_GFLOPS_cg_basic_spmv)
    compute_and_save_correlations(
        df=df_spmv,
        group_cols=["k_bin", "nth"],  # the function adds k_bin internally
        col_pair_1=("median_GFLOPS_yax_spmv_basic", "median_GFLOPS_cg_basic_spmv"),
        col_pair_2=("median_GFLOPS_yax_xyp_spmv_basic", "median_GFLOPS_cg_basic_spmv"),
        output_csv="outputs/intel_output_correlations_spmv.csv"
    )

    # 4. SPMM correlation
    #    Group by [k_bin, nth, n], but we only pass ["nth","n"] — again,
    #    the function adds k_bin internally.
    #    We'll compare:
    #      - (median_GFLOPs_spmm_basic) vs (median_GFLOPs_spmm_GCN)
    #      - (median_GFLOPs_spmm_basic_xyp) vs (median_GFLOPs_spmm_GCN)
    compute_and_save_correlations(
        df=df_spmm,
        group_cols=["k_bin", "nth", "n"],
        col_pair_1=("median_GFLOPs_spmm_basic", "median_GFLOPs_spmm_GCN"),
        col_pair_2=("median_GFLOPs_spmm_basic_xyp", "median_GFLOPs_spmm_GCN"),
        output_csv="outputs/intel_output_correlations_spmm.csv"
    )

if __name__ == "__main__":
    main()
