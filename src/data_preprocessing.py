import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
def load_data(file_path):
    """Load the dataset from a CSV file."""
    df = pd.read_csv(r"A:\project_x\Anomaly_Detection\data\synthetic_network_traffic.csv")
    return df



from imblearn.over_sampling import SMOTE

def preprocess_data(df):
    """Handle missing values, normalize data, balance dataset, and split into train-test sets."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Drop duplicate rows if any
    df = df.drop_duplicates()

    # Replace NaN and infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    # Extract features and labels
    X = df.drop(columns=['IsAnomaly'])  # Features
    y = df['IsAnomaly']  # Labels

    # Balance dataset using SMOTE
    smote = SMOTE(sampling_strategy=0.1, random_state=42)  # Adjust strategy as needed
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)

    # Split into training and test sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.3, random_state=42)

    print(f"Balanced dataset: {dict(pd.Series(y_resampled).value_counts())}")

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # Load and preprocess data
    file_path = r"A:\project_x\Anomaly_Detection\data\synthetic_network_traffic.csv"  # Update path if needed
    df = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Print dataset details
    print(f"Train set size: {X_train.shape}, Test set size: {X_test.shape}")
