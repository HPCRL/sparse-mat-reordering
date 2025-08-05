import pandas as pd

def generate_rcm_vs_metis_table(df):
    method_column_map = {
        'IOS': 'median_GFLOPs_ios',
        'CG': 'median_GFLOPs_real',
        'RAX': 'median_GFLOPs_yax'
    }

    # Replace machine names
    df['machine'] = df['machine'].replace({
        'athena': 'Intel-Server',
        'zeus': 'AMD-Server',
        'intel-Desktop': 'Intel-Desktop'
    })

    # Only keep RCM and METIS and static schedule
    df = df[df['schedule'] == 'static']
    df = df[df['reordering'].isin(['RCM', 'METIS'])]

    results = []

    for method, gflops_col in method_column_map.items():
        for machine in df['machine'].unique():
            subdf = df[df['machine'] == machine]

            win = lose = 0

            for mat in subdf['matrix_name'].unique():
                rcm_val = subdf[(subdf['matrix_name'] == mat) & (subdf['reordering'] == 'RCM')][gflops_col]
                metis_val = subdf[(subdf['matrix_name'] == mat) & (subdf['reordering'] == 'METIS')][gflops_col]

                if rcm_val.empty or metis_val.empty:
                    continue

                rcm_val = rcm_val.values[0]
                metis_val = metis_val.values[0]

                if rcm_val > metis_val:
                    win += 1
                elif rcm_val < metis_val:
                    lose += 1
                # ties are ignored in this analysis

            results.append({
                'Machine': machine,
                f'{method}_w': win,
                f'{method}_l': lose
            })

    # Merge results row-wise by Machine
    result_df = pd.DataFrame(results)
    final_df = result_df.groupby('Machine').sum().reset_index()

    # Save as CSV
    final_df.to_csv("rcm_vs_metis_comparison.csv", index=False)
    print("✅ Cleaned table saved to: rcm_vs_metis_comparison.csv")

    return final_df


# Main execution
if __name__ == "__main__":
    file_path = './Perf-m>10k-all-schedules.csv'
    df = pd.read_csv(file_path)
    df = df[df['m'] > 10000]
    cleaned_table = generate_rcm_vs_metis_table(df)
    print(cleaned_table)
