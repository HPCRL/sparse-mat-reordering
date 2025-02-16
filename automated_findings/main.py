import pandas as pd
import numpy as np
from scipy.stats import spearmanr


def get_avg_spearsman_of_ranked_lists(ordered_lists):
    # Ensure we can iterate more than once
    ordered_lists = list(ordered_lists)
    # Extract all keys (ref_values) in the current set (they should be the same for all lists in the set)
    all_keys = sorted({entry['ref_value'] 
                        for ordered_list in ordered_lists 
                        for entry in ordered_list})

    # Convert each list into a ranking dictionary
    rankings = []
    for ordered_list in ordered_lists:
        rank_dict = {
            entry['ref_value']: rank 
            for rank, entry in enumerate(ordered_list, start=1)
        }
        rankings.append([rank_dict[key] for key in all_keys])  # Align keys

    # Compute Spearman's correlation coefficient for all pairs
    num_lists = len(rankings)
    correlation_values = []

    for i in range(num_lists):
        for j in range(i + 1, num_lists):  # Avoid redundant computations
            corr, _ = stats.spearmanr(rankings[i], rankings[j])
            correlation_values.append(corr)

    # Compute the average Spearman correlation for the set
    avg_correlation = np.mean(correlation_values) if correlation_values else None
    return avg_correlation

'''
def compute_average_spearman(scenario_info):
    """
    Given the scenario_info for one scenario, which is a list of partition dicts
    each with 'ranking_list', compute the *average pairwise Spearman correlation*
    among those sub-partitions' rankings.

    :param scenario_info: List of dicts for each fixed-dim partition
                         with a field 'ranking_list'.
    :return: A float representing the average pairwise Spearman correlation,
             or None if there's fewer than 2 partitions to compare.
    """
    # If there's 0 or 1 partitions, we can't compute pairwise correlation
    if len(scenario_info) < 2:
        return None
    
    # Collect all possible ref_values that appear in ANY partition's ranking_list
    all_ref_values = set()
    for partition in scenario_info:
        for entry in partition['ranking']:
            all_ref_values.add(entry['ref_value'])
    all_ref_values = list(all_ref_values)  # fix an order
    # (We could sort them lexicographically or keep them as-is.)
    
    # Build an array of shape (num_partitions, num_distinct_ref_values)
    # Each row i corresponds to one partition's ranking, 
    # and the columns correspond to the ranks for each ref_value in `all_ref_values`.
    # We'll fill with np.nan if that ref_value was not present (though in our data 
    # it probably always is, but let's be safe).
    num_partitions = len(scenario_info)
    num_refs = len(all_ref_values)
    rank_matrix = np.full((num_partitions, num_refs), np.nan, dtype=float)
    
    for i, partition in enumerate(scenario_info):
        # partition['ranking_list'] is sorted in descending order by fraction_above_ST
        # So the item at index 0 has "rank = 1", index 1 => rank = 2, etc.
        # We'll store that integer rank in the array.
        for rank_idx, entry in enumerate(partition['ranking']):
            ref_val = entry['ref_value']
            # The rank is rank_idx + 1 (1-based rank)
            # Find the column in all_ref_values
            if ref_val in all_ref_values:
                j = all_ref_values.index(ref_val)
                rank_matrix[i, j] = rank_idx + 1
    
    # Now we compute pairwise Spearman correlations among the rows of rank_matrix
    # We'll do a double loop to gather all off-diagonal correlations
    corrs = []
    for i in range(num_partitions):
        for j in range(i + 1, num_partitions):
            # Spearman correlation on row i vs row j
            # If there's missing data (NaN), spearmanr will ignore those pairs 
            # by default with 'nan_policy="omit"'.
            x = rank_matrix[i, :]
            y = rank_matrix[j, :]
            
            # Create a mask of non-NaN indices in both x and y:
            valid = np.isfinite(x) & np.isfinite(y)
            
            # Keep only valid entries:
            x_valid = x[valid]
            y_valid = y[valid]
            
            # Only compute Spearman if there's enough data
            if len(x_valid) >= 2:
                rho, pval = spearmanr(x_valid, y_valid)  # nan_policy no longer needed
                if not np.isnan(rho):
                    corrs.append(rho)
            else:
                # There's not enough data points to compute correlation
                # so skip or handle in some default way
                pass
    
    if len(corrs) == 0:
        return None
    else:
        return float(np.mean(corrs))
'''

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
                      speedup_threshold=1.05,
                      local_consistency_ratio_knob=0.5,
                      ):
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
            
            # keep sub-partition consistency ratio over all free dim vals for each fixed value
            local_cons_ratios = {}

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
                
                local_per_fixed_val_cons_ratio = sum(subpartition_consistency.values())/len(subpartition_consistency) if len(subpartition_consistency) > 0 else 0
                local_cons_ratios[fixed_value] = True if local_per_fixed_val_cons_ratio > local_consistency_ratio_knob else False
            
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
                    'subpartition_consistency_ratio': local_per_fixed_val_cons_ratio,
                    'ranking': ranking_list
                })
            
            global_scenario_consistency_ratio = sum(local_cons_ratios.values())/len(local_cons_ratios) if len(local_cons_ratios) > 0 else 0
            global_rankings_avg_spearman = get_avg_spearsman_of_ranked_lists(scen['ranking'] for scen in scenario_info)
            # global_rankings_avg_spearman = compute_average_spearman(scenario_info)
            results[scenario_key] = {
                    'scenario_info': scenario_info,
                    'scenario_global_consistency_ratio': global_scenario_consistency_ratio,
                    'scenario_rankings_avg_spearman': global_rankings_avg_spearman
                }
    
    return results


if __name__ == "__main__":
    # TODO: binning of matrices based on n and nnz
    # TODO: sort based on least fixed dims
    # TODO: hierarchy?
    # TODO loop over these values and output separate files
    sts = [1.01, 1.02, 1.05]
    cts = [0.5, 0.55, 0.6, 0.65, 0.7]


    # Calculate a correlation metric (spearman) for output rankings (inside a scenario) and filter the prints to keep highly correlated ones 
    # should probably have a knob for correlation to print

    # percentage of true consistency inside each scenario for each fixed dim value
    # print the scenario if any of these percentages are more than a knob (local-knob)
    # print the global percentage of fixed-dim-value cases that have a consistency percentage more than the mentioned knob --> global-knob to print
    
    # in case of matrix_name as fixed dim, only print the global percentage with global-knob 30%




    df_all = pd.read_csv("../../csvdata/cleaned-only-speedups.csv")

    # Filter for SpMV + n=1 + nonzero median_speedup_spmv_xyp + max nth
    df_filtered = filter_for_spmv_xyp(df_all)

    for st in sts:
        for ct in cts:
            # Then explore scenarios
            results = explore_scenarios(df_filtered,
                                        speedup_col='median_speedup_spmv_xyp',
                                        consistency_threshold=ct,
                                        speedup_threshold=st)
            filename = f"results/results_output_st{st}_ct{ct}.txt"
            with open(filename, "w") as file:
                file.write(f"Processing pair (st={st}, ct={ct})\n")
                for scenario, info_dict in results.items():
                    if ", Reference=matrix_name" in scenario:
                        continue
                    file.write("\n--------------------------------------------------------------------------------------------------------------\n")
                    file.write(f"Scenario: {scenario}\n")
                    file.write(f"Global consistency ratio: {info_dict['scenario_global_consistency_ratio']}\n")
                    file.write(f"Global rankings average spearman: {info_dict['scenario_rankings_avg_spearman']}\n")
                    info_list = info_dict['scenario_info']
                    for info in info_list:
                        file.write(f"  Fixed value: {info['fixed_value']}\n")
                        file.write(f"  Local consistency for fixed value: {info['subpartition_consistency_ratio']}\n")
                        file.write(f"  Consistency per ref_value: {info['subpartition_consistency']}\n")
                        file.write(f"  Ranking of ref_values: {info['ranking']}\n")
                        file.write("\n")
    
    # # Inspect or print out the results
    # for scenario, info_dict in results.items():
    #     if ", Reference=matrix_name" in scenario:
    #         continue
    #     print("\n--------------------------------------------------------------------------------------------------------------")
    #     print(f"Scenario: {scenario}")
    #     print(f"Global consistency ratio: {info_dict['scenario_global_consistency_ratio']}")
    #     info_list = info_dict['scenario_info']
    #     for info in info_list:
    #         print("  Fixed value:", info['fixed_value'])
    #         print("  Local consistency for fixed value:", info['subpartition_consistency_ratio'])
    #         print("  Consistency per ref_value:", info['subpartition_consistency'])
    #         print("  Ranking of ref_values:", info['ranking'])
    #         print()

    # pass
