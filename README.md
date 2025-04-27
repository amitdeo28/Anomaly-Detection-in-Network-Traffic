# Anomaly Detection in Network Traffic

A **Hybrid Approach** combining **Isolation Forest** and a **Deep Neural Network (DNN)** to detect anomalies in network traffic data.

---

## 🚀 Project Overview

This project implements a two-stage anomaly detection system:
1. **Isolation Forest** - for fast, unsupervised anomaly detection.
2. **Deep Neural Network** - for fine-tuned anomaly classification.

It aims to improve the detection of malicious or unusual network behavior by combining the strengths of traditional machine learning and deep learning models.

---

## 📂 Project Structure

```
models/          # Saved machine learning and deep learning models
scripts/         # Preprocessing and training scripts
src/             # Core source code for model building and evaluation
webapp/          # Web interface for real-time anomaly detection
split_dataset.py # Utility to split dataset into train/test
requirements.txt # Python package dependencies
environment.yml  # Conda environment setup
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/amitdeo28/Anomaly-Detection-in-Network-Traffic.git
cd Anomaly-Detection-in-Network-Traffic

# Create a conda environment
conda env create -f environment.yml
conda activate anomaly_detection

# Alternatively, install dependencies manually
pip install -r requirements.txt
```

---

## 🏃‍♂️ How to Run

1. **Data Preparation**
   - Use `data_preprocessing.py` to prepare your train/test datasets.

2. **Model Training**
   - Run the training scripts inside the `scripts/` directory to train Isolation Forest and DNN models.

3. **Web Interface**
   - Navigate to `webapp/` and launch the web application to perform real-time anomaly detection.

---

## 📈 Results

- Improved anomaly detection by combining the efficiency of Isolation Forest with the deep learning capabilities of a DNN.
- Evaluation metrics include **Precision**, **Recall**, and **F1-Score**.

---

## 🧠 Tech Stack

- Python
- Scikit-learn
- TensorFlow / Keras
- Flask (for the web application)
- Pandas, NumPy, Matplotlib

---

## 📬 Contact

- [Amit Deo](https://github.com/amitdeo28)

---

> *"Detection is not enough; understanding the anomalies leads to better security."*
