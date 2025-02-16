import pandas as pd
import os

def reshape_spmv_spmm_data(input_csv, output_dir="."):
    """
    Reads the CSV and creates pivoted tables for each submethod of SpMV and SpMM.
    Submethods for SpMV:  basic, xyp, cg
    Submethods for SpMM:  basic, xyp, gcn

    Each combination of (machine, nth, submethod) will be saved to its own CSV.
    The pivot is done on 'reordering', with values = median GFLOPs for that submethod.
    """

    # Read the CSV file
    df = pd.read_csv(input_csv)

    # If you want to remove any rows (e.g., a particular machine), do so here.
    # For example:
    # df = df[df["machine"] != "ookami-skylake"]
    
    # -- Define which columns hold the median GFLOPs for each sub-method --
    # SPMV sub-method columns
    spmv_submethods = {
        'basic': 'median_GFLOPs_yax_spmv_basic',
        'xyp':   'median_GFLOPs_yax_xyp_spmv_basic',
        'cg':    'median_GFLOPs_cg_basic_spmv'
    }
    
    # SPMM sub-method columns
    spmm_submethods = {
        'basic': 'median_GFLOPs_spmm_basic',
        'xyp':   'median_GFLOPs_spmm_basic_xyp',
        'gcn':   'median_GFLOPs_spmm_GCN'
    }
    
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Loop over machines
    for machine_name in df['machine'].unique():
        machine_df = df[df['machine'] == machine_name]
        
        # Now loop over nth values actually present for this machine
        for nth_val in machine_df['nth'].unique():
            machine_nth_df = machine_df[machine_df['nth'] == nth_val]
            
            # Loop over method (spmv, spmm)
            for method_val in ['SpMV', 'SpMM']:
                subset_df = machine_nth_df[machine_nth_df['method'] == method_val]
                
                # Pick the correct dictionary of submethods
                if method_val == 'SpMV':
                    submethods = spmv_submethods
                else:
                    submethods = spmm_submethods
                
                # For each submethod, pivot on 'reordering' with the chosen median GFLOPs column
                for submethod_name, gflops_column in submethods.items():
                    if gflops_column not in subset_df.columns:
                        continue
                    
                    # Pivot table on reordering
                    pivot_df = subset_df.pivot_table(
                        index=['matrix_name', 'm', 'k', 'nnz', 'n'],
                        columns='reordering',
                        values=gflops_column,
                        aggfunc='first'  # or 'mean', if necessary
                    ).reset_index()
                    
                    # If pivot_df is empty (no rows), skip writing
                    if pivot_df.empty:
                        continue
                    
                    # Build an output filename like "spmv_basic_fooMachine_nth4.csv"
                    out_filename = (
                        f"{method_val}_{submethod_name}_"
                        f"{machine_name}_nth{nth_val}.csv"
                    )
                    out_path = os.path.join(output_dir, out_filename)
                    
                    pivot_df.to_csv(out_path, index=False)
                    print(f"Created {out_path}")

if __name__ == "__main__":
    # Example usage:
    reshape_spmv_spmm_data(
        input_csv="../../csvdata/cleaned.csv",
        output_dir="pivoted_data"  # put all pivoted files here
    )
