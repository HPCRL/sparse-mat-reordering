import pandas as pd

def explore_scenarios(df, speedup_col='median_speedup_spmv_xyp',
                      consistency_threshold=0.7,   # example, 70% 
                      speedup_threshold=1.05):     # example, 1.05
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
    
    # The three dimensions of interest
    dims = ['machine', 'matrix_name', 'reordering']
    
    # To store all results
    results = {}
    
    # Enumerate all ways to choose exactly one fixed dim, 
    # one reference dim, and one free dim
    for fixed_dim in dims:
        remaining_dims = [d for d in dims if d != fixed_dim]
        for ref_dim in remaining_dims:
            free_dim = [d for d in remaining_dims if d != ref_dim][0]
            
            # We’ll collect results here
            scenario_key = f"Fixed={fixed_dim}, Reference={ref_dim}, Free={free_dim}"
            scenario_info = []
            
            # Group by the fixed dimension
            grouped_fixed = df.groupby(fixed_dim)
            
            for fixed_value, fixed_group in grouped_fixed:
                # Now within this group, sub-partition by reference_dim
                grouped_ref = fixed_group.groupby(ref_dim)
                
                subpartition_consistency = {}  # Will map ref_value -> True/False for consistency
                subpartition_count = {}       # Will map ref_value -> (count of above ST, total)
                
                # First pass: figure out the "consistency" in each sub-partition
                for ref_value, ref_group in grouped_ref:
                    # ref_group is a sub-partition with the free_dim varying 
                    # (plus any other columns we filtered on earlier).
                    
                    # Evaluate how many data points are above ST in this sub-part
                    speeds = ref_group[speedup_col].values
                    if len(speeds) == 0:
                        fraction_above = 0
                    else:
                        fraction_above = sum(speeds > speedup_threshold) / len(speeds)
                    
                    is_consistent = (fraction_above >= consistency_threshold)
                    
                    subpartition_consistency[ref_value] = is_consistent
                    subpartition_count[ref_value] = (sum(speeds > speedup_threshold), len(speeds))
                
                # Second pass: "ranking" the distinct values of reference_dim. 
                # For demonstration, let’s rank by the fraction of points above ST.
                # You might prefer average speedup or something else.
                ranking = sorted(
                    subpartition_count.items(),
                    key=lambda x: x[1][0] / x[1][1] if x[1][1] > 0 else 0,
                    reverse=True
                )
                # ranking is a list of tuples: [(ref_value, (num_above, total)), ... ] 
                # sorted by fraction_above in descending order.
                
                # Format the ranking into something more direct
                ranking_list = [
                    {
                        'ref_value': r[0],
                        'fraction_above_ST': (r[1][0]/r[1][1]) if r[1][1] > 0 else 0
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
    # ---------------------------
    # Example usage
    # ---------------------------

    # 1. Read your CSV or however you load the dataset
    # df_all = pd.read_csv("your_data.csv")

    # 2. Filter to SpMV, n=1
    #    Also, keep only rows where median_speedup_spmv_xyp != 0 
    #    if that is the correct interpretation of "measurement method = xyp".
    #    Then, keep rows where nth is the maximum for that machine.
    #
    # IMPORTANT QUESTION:
    #   - Do you want to keep only rows with median_speedup_spmv_xyp != 0
    #     or do you have an explicit dimension for "measurement method"? 
    #   - Or do you prefer to keep any row labeled as "xyp" in some other way?
    #
    # For demonstration, let's assume we detect "measurement method = xyp"
    # by spmv_xyp != 0.
    
    # df_spmv = df_all[df_all['method'] == 'SpMV']
    # df_n1 = df_spmv[df_spmv['n'] == 1]
    # df_xyp = df_n1[df_n1['median_speedup_spmv_xyp'] != 0]
    
    # Now find the max nth per machine:
    # max_nth_per_machine = df_xyp.groupby('machine')['nth'].transform('max')
    # df_filtered = df_xyp[df_xyp['nth'] == max_nth_per_machine]
    
    # Or if you prefer a more direct approach:
    # df_filtered = (df_spmv.query("n == 1 and median_speedup_spmv_xyp != 0")
    #                          .groupby('machine', group_keys=False)
    #                          .apply(lambda g: g[g['nth'] == g['nth'].max()]))

    # For demonstration, let’s say df_filtered is your final dataset
    # after applying all the above filters:
    # results = explore_scenarios(df_filtered,
    #                             speedup_col='median_speedup_spmv_xyp',
    #                             consistency_threshold=0.7,
    #                             speedup_threshold=1.05)

    # Now 'results' is a dictionary keyed by the scenario
    # (like "Fixed=machine, Reference=matrix_name, Free=reordering") 
    # with details on each partition’s consistency and ranking.

    # You can then iterate over 'results' to generate a summary or
    # print them out in a more readable format.
    pass
