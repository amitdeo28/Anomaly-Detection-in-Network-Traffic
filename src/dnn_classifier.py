import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from data_preprocessing import load_data, preprocess_data

# Build Deep Neural Network model
def build_dnn(input_shape):
    """Create a Deep Neural Network model."""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification (0 = normal, 1 = anomaly)
    ])
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Train the model
def train_dnn(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=64):
    """Train the deep neural network model."""
    model.fit(X_train, y_train, 
              validation_data=(X_test, y_test),
              epochs=epochs, batch_size=batch_size, verbose=1)
    return model

# Save the model
def save_model(model, filename= r"A:\project_x\Anomaly_Detection\models\dnn_model.h5"):
    """Save trained DNN model to file."""
    model.save(filename)
    print(f"Model saved: {filename}")

# Load the model
def load_model(filename= r"A:\project_x\Anomaly_Detection\models\dnn_model.h5"):
    """Load trained DNN model from file."""
    return tf.keras.models.load_model(filename)

if __name__ == '__main__':
    # Load and preprocess data
    file_path = r'A:\project_x\Anomaly_Detection\data\synthetic_network_traffic.csv'
    df = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Build and train DNN model
    model = build_dnn(X_train.shape[1])
    model = train_dnn(model, X_train, y_train, X_test, y_test)

    # Save model
    save_model(model)
