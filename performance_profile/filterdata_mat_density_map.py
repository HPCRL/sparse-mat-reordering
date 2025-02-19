import pandas as pd

# Load the dataset
file_path = "../../csvdata/data_wdensities.csv"  # Change this to your actual file path
df = pd.read_csv(file_path)

# Filter data where 'reordering' column is 'baseline'
filtered_df = df[df['reordering'] == 'baseline']

# Keep only 'matrix_name' and 'avg' columns
filtered_df = filtered_df[['matrix_name', 'avg']]

# Save the filtered data to a new CSV file
output_file = "mat2density_map.csv"
filtered_df.to_csv(output_file, index=False)

print(f"Filtered dataset saved to {output_file}")
