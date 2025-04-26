import sys
import os

# Add the 'src' directory to Python's module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), r'A:\project_x\Anomaly_Detection\src')))

from data_preprocessing import load_data, preprocess_data
from hybrid_model import train_hybrid_model

if __name__ == '__main__':
    print("Loading and preprocessing data...")
    
    # Load and preprocess data
    file_path = r'A:\project_x\Anomaly_Detection\data\synthetic_network_traffic.csv'
    df = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("Training Hybrid Model...")
    
    # Train hybrid model (Isolation Forest + DNN)
    train_hybrid_model(X_train, X_test, y_train, y_test)

    print("Training completed. Models saved successfully!")
