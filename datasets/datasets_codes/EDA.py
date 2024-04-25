import pandas as pd

# Load the data
data_path = 'datasets/keypoint_for_train_val.csv'
data = pd.read_csv(data_path)

# Display basic info about the dataset
print(data.info())
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Summary statistics for the features
print(data.describe())

# Distribution of labels
print(data['label'].value_counts())
