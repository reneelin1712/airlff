import pandas as pd

# Read the "path.csv" file into a DataFrame
path_df = pd.read_csv("../../data/cross_validation/train_CV0_size1000.csv")

# Read the "shortest_paths.csv" file into a DataFrame
shortest_path_df = pd.read_csv("../../data/shortest/shortest_paths.csv")

# Concatenate the two DataFrames side by side
merged_df = pd.concat([path_df, shortest_path_df["shortest_path"]], axis=1)

# Compare the "path" and "shortest_path" columns
merged_df["Same"] = merged_df.apply(lambda row: "Yes" if row["path"] == row["shortest_path"] else "No", axis=1)

# Select the desired columns for the output
output_df = merged_df[["ori", "des", "path", "shortest_path", "Same"]]

# Save the comparison result to a new CSV file
output_df.to_csv("../../data/shortest/comparison_result.csv", index=False)