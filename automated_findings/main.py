import pandas as pd

def filter_for_spmv_xyp(df, speedup_col='median_speedup_spmv_xyp', max_removal_fraction=0.05):
    """
    Filters the dataset to only SpMV, n=1, and nonzero `speedup_col` rows,
    then keeps only the maximum nth per machine.

    :param df:                The full DataFrame to filter.
    :param speedup_col:       The column with speedup (e.g. 'median_speedup_spmv_xyp').
    :param max_removal_fraction: If more than this fraction of rows are removed 
                                (e.g. 0.05 for 5%), print a warning.
    :return:                  Filtered DataFrame.
    """
    # 1. Keep only SpMV method
    df_spmv = df[df['method'] == 'SpMV']
    
    # 2. Keep only rows where n=1
    df_spmv = df_spmv[df_spmv['n'] == 1]
    
    # 3. Count how many remain before we remove zero-speedup rows
    num_before = len(df_spmv)
    
    # 4. Filter out rows that have zero in 'speedup_col'
    # TODO: double check
    df_spmv = df_spmv[df_spmv[speedup_col] != 0]
    
    # 5. Check how many were removed
    num_after = len(df_spmv)
    num_removed = num_before - num_after
    fraction_removed = num_removed / num_before if num_before > 0 else 0
    
    # 6. Warn if we removed more than max_removal_fraction (i.e. 5%)
    if fraction_removed > max_removal_fraction:
        print(f"WARNING: {fraction_removed * 100:.2f}% of rows were removed "
              f"because {speedup_col} == 0.")
    
    # 7. Keep only rows where nth is the maximum for each machine
    #    (In case multiple rows tie for “maximum nth”, this will keep them all.)
    # TODO: check if this is correct
    # TODO: Is there a case where we end up comparing athena 17 with amd 7? How should we handle this?
    df_spmv = (df_spmv.groupby('machine', group_keys=False)
               .apply(lambda g: g[g['nth'] == g['nth'].max()]))
    
    return df_spmv


def explore_scenarios(df, speedup_col='median_speedup_spmv_xyp',
                      consistency_threshold=0.7,
                      speedup_threshold=1.05):
    """
    Explore all ways of choosing fixed, reference, and free dims among
    [machine, matrix_name, reordering]. Then perform the two evaluations
    (consistency check and ranking) on each partition.
    
    :param df: The filtered DataFrame.
    :param speedup_col: The column in df containing the spmv-xyp speedup values.
    :param consistency_threshold: CT, fraction threshold for "consistency"
    :param speedup_threshold: ST, the speedup threshold
    :return: A dictionary summarizing results for each (fixed_dim, reference_dim).
    """
    
    dims = ['machine', 'matrix_name', 'reordering']
    
    results = {}
    
    for fixed_dim in dims:
        remaining_dims = [d for d in dims if d != fixed_dim]
        for ref_dim in remaining_dims:
            free_dim = [d for d in remaining_dims if d != ref_dim][0]
            
            scenario_key = f"Fixed={fixed_dim}, Reference={ref_dim}, Free={free_dim}"
            scenario_info = []
            
            grouped_fixed = df.groupby(fixed_dim)
            
            # Partition the data by the fixed_dim
            for fixed_value, fixed_group in grouped_fixed:
                # Sub-partition by the reference_dim
                grouped_ref = fixed_group.groupby(ref_dim)
                
                subpartition_consistency = {}
                subpartition_count = {}
                
                # 1) Consistency check
                for ref_value, ref_group in grouped_ref:
                    speeds = ref_group[speedup_col].values
                    if len(speeds) == 0:
                        fraction_above = 0
                    else:
                        fraction_above = sum(speeds > speedup_threshold) / len(speeds)
                    
                    is_consistent = (fraction_above >= consistency_threshold)
                    subpartition_consistency[ref_value] = True if is_consistent else False
                    subpartition_count[ref_value] = (sum(speeds > speedup_threshold), len(speeds))
                
                # 2) Ranking sub-partitions (distinct values of ref_dim) 
                #    by fraction of data points above speedup_threshold
                ranking = sorted(
                    subpartition_count.items(),
                    key=lambda x: x[1][0] / x[1][1] if x[1][1] > 0 else 0,
                    reverse=True
                )
                
                ranking_list = [
                    {
                        'ref_value': r[0],
                        'fraction_above_ST': float((r[1][0] / r[1][1])) if r[1][1] > 0 else 0
                    }
                    for r in ranking
                ]
                
                scenario_info.append({
                    'fixed_dim': fixed_dim,
                    'fixed_value': fixed_value,
                    'reference_dim': ref_dim,
                    'subpartition_consistency': subpartition_consistency,
                    'ranking': ranking_list
                })
            
            results[scenario_key] = scenario_info
    
    return results


if __name__ == "__main__":
    # TODO: binning of matrices based on n and nnz
    # TODO: sort based on least fixed dims
    # TODO: hierarchy?
    # Example usage (assuming you have a CSV file or some other data source)
    df_all = pd.read_csv("../../csvdata/cleaned-only-speedups.csv")

    # Filter for SpMV + n=1 + nonzero median_speedup_spmv_xyp + max nth
    df_filtered = filter_for_spmv_xyp(df_all)

    # Then explore scenarios
    results = explore_scenarios(df_filtered,
                                speedup_col='median_speedup_spmv_xyp',
                                consistency_threshold=0.7,
                                speedup_threshold=1.05)

    # Inspect or print out the results
    for scenario, info_list in results.items():
        if ", Reference=matrix_name" in scenario:
            continue
        print("\n--------------------------------------------------------------------------------------------------------------")
        print(f"Scenario: {scenario}")
        for info in info_list:
            print("  Fixed value:", info['fixed_value'])
            print("  Consistency per ref_value:", info['subpartition_consistency'])
            print("  Ranking of ref_values:", info['ranking'])
            print()

    # pass
