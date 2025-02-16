import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def performance_profile(method_times, best_times):
    """
    Given Series/array of times for one method (method_times)
    and the per-problem best times (best_times),
    return sorted performance ratios and the corresponding
    fraction of problems (CDF).
    """
    # ratio = method_times / best_times  # If "time" means smaller is better
    # But your code uses best_times / method_times, 
    # meaning bigger (GFLOPs) is better, so let's keep that:
    ratio = best_times / method_times
    
    # Sort these ratios in ascending order
    ratio_sorted = np.sort(ratio.dropna())  # Make sure to drop NaNs
    # **Do not drop the last 10** (unlike your old code).
    
    # Fraction of problems that are at or below each ratio
    y = np.arange(1, len(ratio_sorted) + 1) / len(ratio_sorted)
    return ratio_sorted, y

def plot_performance_profile_for_csv(csv_file, reorderings=None):
    """
    Reads one pivoted CSV file and produces a performance profile plot.
    - `reorderings` can be a list of reorderings to keep in the plot 
      (any that do not appear in the CSV will be ignored).
    - Excludes 'IOU' if present.
    - Saves figure as <csv_file>.png in the same directory.
    """
    # Read the pivoted CSV
    df = pd.read_csv(csv_file)

    # 1) Drop descriptor columns
    df = df.drop(columns=['matrix_name', 'm', 'k', 'nnz', 'n'])

    # Drop any rows completely filled with NaN (optional, depending on your data)
    df = df.dropna(how='all')
    
    # If you already know the reorderings you want, keep them if they exist
    if reorderings is None:
        # If not specified, just take all columns
        # but exclude "IOU" if present
        reorderings = [c for c in df.columns if c != "IOU"]
    else:
        # Filter reorderings by those actually in df, also exclude IOU
        reorderings = [r for r in reorderings if (r in df.columns and r != "IOU")]
    
    # If there are no reorderings left after filtering, skip
    if not reorderings:
        print(f"No valid reorderings to plot in {csv_file}. Skipping.")
        return
    
    # Drop rows where all reorderings are NaN
    df = df.dropna(subset=reorderings, how='all')
    
    # Compute the "best" for each row. Since we interpret "bigger is better"
    # (GFLOPs), the best is the max across reorderings.
    best_times = df[reorderings].max(axis=1)
    
    # Create a new figure
    plt.figure(figsize=(8, 6))
    
    for method_name in reorderings:
        method_times = df[method_name]
        # If everything is NaN for this method, skip it
        if method_times.isna().all():
            continue
        x, y = performance_profile(method_times, best_times)
        # Step plot
        plt.step(x, y, where='post', label=method_name)
    
    # Beautify the plot
    # Not using log scale
    # Limit x-axis up to 2
    plt.xlim([1, 2])
    plt.ylim([0, 1.05])
    plt.xlabel('Performance Relative to Best')
    plt.ylabel('Fraction of Problems')
    plt.title(f'Performance Profile: {os.path.basename(csv_file)}')
    plt.grid(True)
    plt.legend(loc='lower right')
    
    # Save the figure
    out_png = csv_file.replace('.csv', '.png')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure so we don't reuse it
    print(f"Created {out_png}")

def main():
    # Example: process all CSV files in "pivoted_data" folder
    csv_files = glob.glob("pivoted_data/*.csv")
    
    # You may define reorderings explicitly (e.g. if you know the columns):
    # reorder_list = ["Gorder", "Grappolo", "IOU", "Louvain", 
    #                 "METIS", "PaToH", "RCM", "baseline"]
    # or just let the function handle them automatically.
    
    for csv_file in csv_files:
        plot_performance_profile_for_csv(csv_file)

if __name__ == "__main__":
    main()
