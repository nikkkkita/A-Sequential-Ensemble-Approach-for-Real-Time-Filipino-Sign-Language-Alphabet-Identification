import pandas as pd
from sklearn.model_selection import train_test_split

# Path to your dataset
data_path = 'model/keypoint_classifier/keypoint.csv'

# Load the dataset
data = pd.read_csv(data_path)

# Assuming the label column is the first one, adjust if otherwise
X = data.drop(data.columns[0], axis=1)  # Features: all columns except the first
y = data[data.columns[0]]  # Labels: the first column

# Split the data into training/validation and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, 
    test_size=0.2,    # 20% of the data will be used for testing
    random_state=42,  # Ensures reproducibility of your split
    stratify=y        # Ensures the distribution of classes is balanced in both sets
)

# Combine the test features and labels back into a single DataFrame
test_data = pd.concat([y_test, X_test], axis=1)

# Combine the train/validation features and labels back into a single DataFrame
train_val_data = pd.concat([y_train_val, X_train_val], axis=1)

# Paths to save the CSV files
test_data_path = 'datasets/keypoint_for_testing.csv'
train_val_data_path = 'datasets/keypoint_for_train_val.csv'

# Save the test set
test_data.to_csv(test_data_path, index=False)
print("Test data saved to:", test_data_path)

# Save the train/validation set
train_val_data.to_csv(train_val_data_path, index=False)
print("Train/Validation data saved to:", train_val_data_path)
