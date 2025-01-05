import pandas as pd

def reshape_data_by_method(input_csv):
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # remove ookami-skylake machien due to unstabality
    df = df[df["machine"] != "ookami-skylake"]

    # Drop unnecessary columns
    columns_to_drop = [
        'avg_GFLOPS', 'min_GFLOPS', 'max_GFLOPS', 'stddev_GFLOPS', 'Relative_STD', 
        'avg_GFLOPS_baseline', 'max_GFLOPS_baseline', 'min_GFLOPS_baseline', 'stddev_GFLOPS_baseline',
        'median_speedup', 'avg_speedup', 'max_speedup', 'min_speedup'
    ]
    df = df.drop(columns=columns_to_drop)

    # List all unique methods
    methods = df['method'].unique()

    # Columns that define a unique problem (except 'reordering')
    # Adjust this list if needed; include everything that doesn't vary across
    # the 6 rows, plus 'method' if you want that in your output too.
    group_cols = [
        'machine', 'matrix_name', 
        # 'm', 'k', 'nnz', 'n', 'nth', 
        # 'method'
    ]

    for method in methods:
        # Filter rows for this particular method
        method_df = df[df['method'] == method].copy()

        # Pivot: index = group_cols, columns = reordering, values = median_GFLOPS
        pivot_df = method_df.pivot_table(
            index=group_cols,
            columns='reordering',
            values='median_GFLOPS',
            aggfunc='first'  # or 'mean' if duplicates exist
        ).reset_index()

        # # We'll also keep a single median_GFLOPS_baseline column for each group
        # baseline_df = (
        #     method_df
        #     .groupby(group_cols)['median_GFLOPS_baseline']
        #     .first()
        #     .reset_index()
        # )

        # # Merge the pivoted data with the baseline values
        # final_df = pd.merge(pivot_df, baseline_df, on=group_cols, how='left')
        final_df = pivot_df

        # Write out to a CSV named after the method value
        output_csv_name = f"{method}_output.csv"
        final_df.to_csv(output_csv_name, index=False)
        print(f"Created {output_csv_name}")


if __name__ == "__main__":
    # Example usage:
    reshape_data_by_method("../csvdata/perf-cores-1-n128.csv")
