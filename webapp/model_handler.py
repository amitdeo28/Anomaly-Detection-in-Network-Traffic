# import sys
# import os
# import pandas as pd
# import numpy as np

# # Get the absolute path of the project root
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# # Add 'src' directory to Python's module search path
# sys.path.insert(0, os.path.join(project_root, 'src'))

# # Import from src correctly
# from isolation_forest import load_model as load_if_model
# from dnn_classifier import load_model as load_dnn_model

# # Set the results folder
# RESULTS_FOLDER = os.path.join(os.getcwd(), 'results')

# def detect_anomalies(file_path):
#     """
#     This function loads the trained models (Isolation Forest & DNN),
#     applies them to the uploaded dataset, and detects anomalies.
#     """
#     try:
#         # Load dataset
#         df = pd.read_csv(file_path)

#         # Select relevant features (Adjust based on dataset structure)
#         X_new = df.iloc[:, :-1].values  

#         # Load trained models
#         isolation_forest = load_if_model()
#         dnn = load_dnn_model()

#         # Get anomaly scores from Isolation Forest
#         anomaly_scores = isolation_forest.decision_function(X_new).reshape(-1, 1)

#         # Append anomaly scores as an additional feature
#         X_hybrid = np.hstack((X_new, anomaly_scores))

#         # Predict anomalies using the DNN model
#         predictions = dnn.predict(X_hybrid)
#         predictions = (predictions > 0.2).astype(int)  # Convert probabilities to binary labels

#         # Save results with anomaly labels
#         df['Anomaly'] = predictions
#         result_filename = f"anomalies_{os.path.basename(file_path)}"
#         result_path = os.path.join(RESULTS_FOLDER, result_filename)
#         df.to_csv(result_path, index=False)

#         return result_filename

#     except Exception as e:
#         print(f"❌ Error in detect_anomalies: {e}")
#         return None

# def detect_anomaly_single(features):
#     """Process single-row input and detect anomaly."""
#     isolation_forest = load_if_model()
#     dnn = load_dnn_model()

#     features = np.array(features).reshape(1, -1)
#     anomaly_score = isolation_forest.decision_function(features).reshape(-1, 1)
#     X_hybrid = np.hstack((features, anomaly_score))

#     prediction = dnn.predict(X_hybrid)
#     return int(prediction > 0.2)

import sys
import os
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add 'src' directory to Python's module search path
sys.path.insert(0, os.path.join(project_root, 'src'))

# Import custom modules (if any exist in 'src')
# If you have custom code in src, import as needed.

# Set the results folder
RESULTS_FOLDER = os.path.join(os.getcwd(), 'results')

def load_if_model():
    """Loads the Isolation Forest model from the 'models' directory using joblib."""
    try:
        model_path = os.path.join(project_root, 'models', 'isolation_forest.pkl')
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"❌ Error loading Isolation Forest model: {e}")
        return None

def load_dnn_model():
    """Loads the DNN model from the 'models' directory."""
    try:
        model_path = os.path.join(project_root, 'models', 'dnn_model.h5')
        model = keras.models.load_model(model_path)
        model.compile()  # Compile the model to avoid warning
        return model
    except Exception as e:
        print(f"❌ Error loading DNN model: {e}")
        return None

def detect_anomalies(file_path):
    """
    This function loads the trained models (Isolation Forest & DNN),
    applies them to the uploaded dataset, and detects anomalies.
    """
    try:
        # Load dataset
        df = pd.read_csv(file_path)

        # Select relevant features (Adjust based on dataset structure)
        X_new = df.iloc[:, :-1].values  

        # Load trained models
        isolation_forest = load_if_model()
        dnn = load_dnn_model()

        # Check if models were loaded successfully
        if isolation_forest is None or dnn is None:
            print("❌ One or more models could not be loaded.")
            return None

        # Get anomaly scores from Isolation Forest
        anomaly_scores = isolation_forest.decision_function(X_new).reshape(-1, 1)

        # Append anomaly scores as an additional feature
        X_hybrid = np.hstack((X_new, anomaly_scores))

        # Predict anomalies using the DNN model
        predictions = dnn.predict(X_hybrid)
        predictions = (predictions > 0.2).astype(int)  # Convert probabilities to binary labels

        # Save results with anomaly labels
        df['Anomaly'] = predictions
        result_filename = f"anomalies_{os.path.basename(file_path)}"
        result_path = os.path.join(RESULTS_FOLDER, result_filename)
        df.to_csv(result_path, index=False)

        return result_filename

    except Exception as e:
        print(f"❌ Error in detect_anomalies: {e}")
        return None

def detect_anomaly_single(features):
    """
    Detect anomalies for a single row of input features using the trained models.
    """
    try:
        # Load models
        isolation_forest = load_if_model()
        dnn = load_dnn_model()

        # Check if models were loaded successfully
        if isolation_forest is None or dnn is None:
            print("❌ One or more models could not be loaded.")
            return None

        # Reshape features for single-row input
        features = np.array(features).reshape(1, -1)

        # Get anomaly score from Isolation Forest
        anomaly_score = isolation_forest.decision_function(features).reshape(-1, 1)

        # Combine features and anomaly score
        X_hybrid = np.hstack((features, anomaly_score))

        # Predict anomaly using DNN
        prediction = dnn.predict(X_hybrid)
        return int(prediction > 0.2)

    except Exception as e:
        print(f"❌ Error in detect_anomaly_single: {e}")
        return None
