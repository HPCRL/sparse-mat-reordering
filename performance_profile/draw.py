import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def performance_profile(method_times, best_times):
    """
    Given Series/array of times for one method (method_times)
    and the per-problem best times (best_times),
    return sorted performance ratios and the corresponding
    fraction of problems (CDF).
    """
    # Compute performance ratio per problem
    # ratio = method_times / best_times
    ratio = best_times / method_times
    
    # Sort these ratios in ascending order
    ratio_sorted = np.sort(ratio)
    print(ratio_sorted[-10:])
    ratio_sorted = ratio_sorted[:-10]
    
    # Fraction of problems that are at or below each ratio
    y = np.arange(1, len(ratio_sorted) + 1) / len(ratio_sorted)
    return ratio_sorted, y

def main():
    '''
    input_csv = "../csvdata/perf-cores-1-n128.csv"
    # input_csv = "spmm_basic_output.csv"
    # input_csv = "spmm_mkl_output.csv"

    df = pd.read_csv(input_csv)
    # df = df[df['method']=="spmm_basic"]
    df = df[df['method']=="spmm_mkl"]

    # print(len(df))

    # Drop unnecessary columns
    columns_to_drop = [
        'median_GFLOPS', 'median_GFLOPS_baseline',
        'm', 'k', 'nnz', 'n', 'nth',
        'avg_GFLOPS', 'min_GFLOPS', 'max_GFLOPS', 'stddev_GFLOPS', 'Relative_STD', 
        'avg_GFLOPS_baseline', 'max_GFLOPS_baseline', 'min_GFLOPS_baseline', 'stddev_GFLOPS_baseline',
        'median_speedup', 'avg_speedup', 'max_speedup', 'min_speedup'
    ]

    # columns_to_drop = [
    #     'median_GFLOPS_baseline',
    #     'method',
    #     'm', 'k', 'nnz', 'n', 'nth',
    #     "Gorder", "Grappolo", "IOU", "Louvain", "METIS", "PaToH", "RCM", "baseline",
    # ]
    
    df = df.drop(columns=columns_to_drop)
    # print(len(df))

    # duplicated_rows = df[df.duplicated()]
    # print("Duplicated rows:")
    # print(duplicated_rows)

    
    df = df.drop_duplicates()
    # print(len(df))


    reorderings = ["Gorder", "Grappolo", "IOU", "Louvain", "METIS", "PaToH", "RCM", "baseline"]
    for reordering in reorderings:
        df2 = df[df['reordering'] == reordering]
        print(reordering, "\t", len(df2))

    # print(df.iloc[0])
    # mnames = df['matrix_name'].unique()
    # mnames = df['machine'].unique()
    # mnames = df['reordering'].unique()
    # print(len(mnames))

    exit(0)
    '''

    # x = [
    #     [,,,,,], # 1
    #     [,,,,,], # 2
    #     [,,,,,], # 3
    # ]

    # y = [
    #     [1,     0.7,    ,,,], # r1
    #     [0.5,   1,      ,,,], # r2
    #     [0.7,   0.4,    ,,,], # r3
    # ]

    # 1) Read CSV file with timing data. Each column is a method, each row is a test problem.
    # df = pd.read_csv('spmm_basic_output.csv')
    df = pd.read_csv('spmm_mkl_output.csv')
    df = df.dropna()
    columns = ["Gorder", "Grappolo", "IOU", "Louvain", "METIS", "PaToH", "RCM", "baseline"]
    df = df[columns]

    
    
    # 2) Compute the best (minimum) time across methods for each problem
    best_times = df.max(axis=1)
    # print(df.iloc[:10])
    # print(best_times[:10])
    
    # 3) For each method (column) in the DataFrame, compute and plot its performance profile
    plt.figure(figsize=(8, 6))
    
    for method_name in df.columns:
        method_times = df[method_name]
        x, y = performance_profile(method_times, best_times)
        
        # Plot as a step plot (common in performance profiles)
        plt.step(x, y, where='post', label=method_name)
    
    # 4) Beautify the plot
    plt.xlim([1, None])
    plt.ylim([0, 1.05])
    plt.xlabel('Performance Relative to the Best')
    plt.ylabel('Fraction of Problems')
    plt.title('Performance Profile')
    plt.grid(True)
    plt.legend(loc='lower right')
    
    # plt.show()
    plt.savefig('my_plot.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()
