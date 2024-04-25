import pandas as pd

data_path = 'datasets/keypoint_for_train_val.csv'
data = pd.read_csv(data_path)

# Attempt to convert all columns (except the first one if it's labels) to numeric, coercing errors to NaN
numeric_data = data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

# Find any rows with NaNs, which indicate non-numeric values
non_numeric_rows = numeric_data[numeric_data.isna().any(axis=1)]

if not non_numeric_rows.empty:
    print("Non-numeric data found in dataset:")
    print(non_numeric_rows)
else:
    print("No non-numeric data found in the feature columns.")
