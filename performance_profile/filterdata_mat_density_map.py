import pandas as pd

# Load the dataset
# file_path = "../../csvdata/data_wdensities.csv"  # Change this to your actual file path
file_path = "../../csvdata/cleaned_all_speedups-with-structure.csv"
df = pd.read_csv(file_path)

# Filter data where 'reordering' column is 'baseline'
filtered_df = df[df['reordering'] == 'baseline']

# to get it from all data use the following to get all 1167 matrices' avg block density:
filtered_df = filtered_df[filtered_df['nth'] == 1]
filtered_df = filtered_df[filtered_df['machine'] == 'zeus']
filtered_df = filtered_df[filtered_df['method'] == 'SpMM']
filtered_df = filtered_df[filtered_df['n'] == 8]



# Keep only 'matrix_name' and 'avg' columns
# filtered_df = filtered_df[['matrix_name', 'avg']]
filtered_df = filtered_df[['matrix_name', 'baseline_structure_metric']]
# Rename the second column to 'avg'
filtered_df.columns = ['matrix_name', 'avg']

# Save the filtered data to a new CSV file
output_file = "mat2density_k10_map.csv"
filtered_df.to_csv(output_file, index=False)

print(f"Filtered dataset saved to {output_file}")
