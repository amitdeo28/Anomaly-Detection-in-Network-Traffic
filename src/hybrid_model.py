import numpy as np
from data_preprocessing import load_data, preprocess_data
from isolation_forest import train_isolation_forest, save_model as save_if_model, load_model as load_if_model
from dnn_classifier import build_dnn, train_dnn, save_model as save_dnn_model, load_model as load_dnn_model

# Train Hybrid Model (Isolation Forest + DNN)
def train_hybrid_model(X_train, X_test, y_train, y_test):
    """Train Isolation Forest and Deep Neural Network as a hybrid model."""
    
    # Step 1: Train Isolation Forest for anomaly detection
    isolation_forest = train_isolation_forest(X_train)
    save_if_model(isolation_forest)

    # Step 2: Use Isolation Forest to predict anomaly scores
    anomaly_scores_train = isolation_forest.decision_function(X_train).reshape(-1, 1)
    anomaly_scores_test = isolation_forest.decision_function(X_test).reshape(-1, 1)

    # Step 3: Append anomaly scores to features
    X_train_hybrid = np.hstack((X_train, anomaly_scores_train))
    X_test_hybrid = np.hstack((X_test, anomaly_scores_test))

    # Step 4: Train DNN with augmented feature set
    dnn = build_dnn(X_train_hybrid.shape[1])
    dnn = train_dnn(dnn, X_train_hybrid, y_train, X_test_hybrid, y_test)
    save_dnn_model(dnn)

    return isolation_forest, dnn

if __name__ == '__main__':
    # Load and preprocess data
    file_path = r'A:\project_x\Anomaly_Detection\data\synthetic_network_traffic.csv'
    df = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Train Hybrid Model
    train_hybrid_model(X_train, X_test, y_train, y_test)
