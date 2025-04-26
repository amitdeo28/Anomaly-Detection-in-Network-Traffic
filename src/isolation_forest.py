import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from data_preprocessing import load_data, preprocess_data

# Train Isolation Forest
def train_isolation_forest(X_train):
    """Train Isolation Forest model on training data."""
    model = IsolationForest(n_estimators=300, contamination=0.1, random_state=42)
    model.fit(X_train)
    return model

# Save the model
def save_model(model, filename= r"A:\project_x\Anomaly_Detection\models\isolation_forest.pkl"):
    """Save trained model to file."""
    joblib.dump(model, filename)
    print(f"Model saved: {filename}")

# Load the model
def load_model(filename= r"A:\project_x\Anomaly_Detection\models\isolation_forest.pkl"):
    """Load trained model from file."""
    return joblib.load(filename)

if __name__ == '__main__':
    # Load and preprocess data
    file_path = r"A:\project_x\Anomaly_Detection\data\synthetic_network_traffic.csv"
    df = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Train Isolation Forest
    model = train_isolation_forest(X_train)

    # Save model
    save_model(model)
