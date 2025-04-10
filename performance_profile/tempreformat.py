import os
import pandas as pd
import argparse

def pivot_spmv_spmm(input_csv, output_dir="."):
    """
    - Reads the CSV file containing both SpMV and SpMM data.
    - For each combination of machine, nth, and method in [spmv, spmm]:
      - If spmv, pivot submethods [yax, xyp, ios, real (cg)].
      - If spmm, pivot submethods [yax, xyp, ios, real (gcn)].
        Additionally, for SpMM, separate for each unique n in the subset.
    - Writes out pivoted CSVs to output_dir.
    """
    
    # Read the CSV file
    df = pd.read_csv(input_csv)
    os.makedirs(output_dir, exist_ok=True)
    
    # Rename columns to match old format
    df.rename(columns={"real": "cg_gcn"}, inplace=True)
    df["nth"] = df["nth_"]  # Replace nth with actual values
    
    # SpMV submethod columns
    spmv_submethods = {
        'yax': 'median_GFLOPs_yax',
        'xyp': 'median_GFLOPs_xyp',
        'cg':  'median_GFLOPs_real',
        'ios': 'median_GFLOPs_ios'
    }
    
    # SpMM submethod columns
    spmm_submethods = {
        'yax': 'median_GFLOPs_yax',
        'xyp': 'median_GFLOPs_xyp',
        'gcn': 'median_GFLOPs_real',
        'ios': 'median_GFLOPs_ios'
    }
    
    # Unique machines
    machines = df['machine'].unique()
    
    for machine_name in machines:
        machine_df = df[df['machine'] == machine_name]
        nth_values = machine_df['nth'].unique()
        
        for nth_val in nth_values:
            machine_nth_df = machine_df[machine_df['nth'] == nth_val]
            
            # We only have two methods: spmv or spmm
            for method_val in ['SpMV', 'SpMM']:
                subset_df = machine_nth_df[machine_nth_df['method'] == method_val]
                if subset_df.empty:
                    continue
                
                if method_val == 'SpMV':
                    submethods = spmv_submethods
                    # We don't separate by n for spmv
                    for subm_name, col_name in submethods.items():
                        if col_name not in subset_df.columns:
                            continue
                        
                        # Pivot on 'reordering'
                        pivot_df = subset_df.pivot_table(
                            index=['matrix_name','m','k','nnz','n'],
                            columns='reordering',
                            values=col_name,
                            aggfunc='first'
                        ).reset_index()
                        
                        if pivot_df.empty:
                            continue
                        
                        # Build output filename
                        out_name = f"{method_val}_{subm_name}_{machine_name}_nth{nth_val}.csv"
                        out_path = os.path.join(output_dir, out_name)
                        pivot_df.to_csv(out_path, index=False)
                        print("Created", out_path)
                
                else:
                    # method_val == 'SpMM'
                    submethods = spmm_submethods
                    unique_n_vals = subset_df['n'].unique()
                    
                    for n_val in unique_n_vals:
                        spmm_n_df = subset_df[subset_df['n'] == n_val]
                        if spmm_n_df.empty:
                            continue
                        
                        for subm_name, col_name in submethods.items():
                            if col_name not in spmm_n_df.columns:
                                continue
                            
                            pivot_df = spmm_n_df.pivot_table(
                                index=['matrix_name','m','k','nnz','n'],
                                columns='reordering',
                                values=col_name,
                                aggfunc='first'
                            ).reset_index()
                            
                            if pivot_df.empty:
                                continue
                            
                            out_name = (
                                f"{method_val}_{subm_name}_{machine_name}"
                                f"_nth{nth_val}_n{n_val}.csv"
                            )
                            out_path = os.path.join(output_dir, out_name)
                            pivot_df.to_csv(out_path, index=False)
                            print("Created", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to input CSV", required=True)
    parser.add_argument("--outdir", help="Output directory", default="output_pivoted")
    args = parser.parse_args()
    
    pivot_spmv_spmm(args.input, args.outdir)
