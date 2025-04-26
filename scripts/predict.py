import sys
import os
import numpy as np

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add 'src' directory to Python's module search path
sys.path.insert(0, os.path.join(project_root, 'src'))

# Now import the modules
from data_preprocessing import load_data, preprocess_data
from isolation_forest import load_model as load_if_model
from dnn_classifier import load_model as load_dnn_model

def predict_anomalies(X_new):
    """Predict whether new data points are anomalies using the hybrid model."""

    # Load trained models
    print("Loading trained models...")
    isolation_forest = load_if_model()
    dnn = load_dnn_model()

    # Get anomaly scores from Isolation Forest
    print("Computing anomaly scores using Isolation Forest...")
    anomaly_scores = isolation_forest.decision_function(X_new).reshape(-1, 1)

    # Append anomaly scores as a feature
    X_new_hybrid = np.hstack((X_new, anomaly_scores))

    # Use the DNN model to make final predictions
    print("Making predictions with DNN...")
    predictions = dnn.predict(X_new_hybrid)
    predictions = (predictions > 0.2).astype(int)  # Convert probabilities to binary labels

    return predictions

if __name__ == '__main__':
    print("Loading and preprocessing data...")

    # Ensure project_root is defined before using it
    file_path = os.path.join(project_root, 'data', 'synthetic_network_traffic.csv')

    # Load and preprocess data (new incoming network traffic)
    df = load_data(file_path)
    X_train, X_test, _, _ = preprocess_data(df)

    print("Predicting anomalies on test data...")

    # Run predictions on test set
    predictions = predict_anomalies(X_test)

    # Display the first 20 predictions
    print("\n📌 Predictions (First 20 Samples):")
    print(predictions[:20])

    # Count anomalies
    num_anomalies = np.sum(predictions)
    print(f"\n🔍 Total Anomalies Detected: {num_anomalies} out of {len(predictions)} samples")
    print("\n✅ Prediction completed successfully.")
