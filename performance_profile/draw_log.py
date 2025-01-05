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
    print(ratio_sorted[-10:])  # Debug
    ratio_sorted = ratio_sorted[:-10]  # Dropping last 10, per your code
    
    # Fraction of problems that are at or below each ratio
    y = np.arange(1, len(ratio_sorted) + 1) / len(ratio_sorted)
    return ratio_sorted, y

def main():
    # 1) Read CSV file with timing data. Each column is a method, each row is a test problem.
    # df = pd.read_csv('spmm_basic_all_pivotted.csv')
    df = pd.read_csv('spmm_mkl_all_pivotted.csv')
    df = df.dropna()
    columns = ["Gorder", "Grappolo", "IOU", "Louvain", "METIS", "PaToH", "RCM", "baseline"]
    df = df[columns]

    # 2) Compute the best (maximum in your code) time across methods for each problem
    best_times = df.max(axis=1)

    # 3) For each method (column) in the DataFrame, compute and plot its performance profile
    plt.figure(figsize=(8, 6))
    
    for method_name in df.columns:
        method_times = df[method_name]
        x, y = performance_profile(method_times, best_times)
        
        # Plot as a step plot (common in performance profiles)
        plt.step(x, y, where='post', label=method_name)
    
    # 4) Beautify the plot
    plt.xscale('log')               # <<< Set the x-axis to log scale here
    plt.xlim([1, None])             # You can adjust as needed
    plt.ylim([0, 1.05])
    plt.xlabel('Performance Relative to the Best (log scale)')
    plt.ylabel('Fraction of Problems')
    plt.title('Performance Profile')
    plt.grid(True)
    plt.legend(loc='lower right')
    
    # Save the plot to a file (no plt.show())
    plt.savefig('my_plot_log.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()
