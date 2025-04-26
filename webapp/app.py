from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
from model_handler import detect_anomalies, detect_anomaly_single  # Import model execution function

app = Flask(__name__)

# Define folders for uploads & results
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
RESULTS_FOLDER = os.path.join(os.getcwd(), 'results')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Ensure required directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html', manual_result=None)  # ✅ Ensure manual_result is None initially

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Run anomaly detection
        result_file = detect_anomalies(file_path)
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_file)

        # ✅ Load results and filter anomalies
        df_results = pd.read_csv(result_path)

        # Count total detected anomalies
        total_anomalies = df_results[df_results["Anomaly"] == 1].shape[0]

        # Filter detected anomalies for display
        df_anomalies = df_results[df_results["Anomaly"] == 1]

        # Convert to dictionary format for HTML display
        table_data = df_anomalies.head(20).to_dict(orient="records")  # Show first 20 anomalies

        # ✅ Pass filtered anomaly data & count to results.html
        return render_template('results.html', 
                               file_url=url_for('download_file', filename=result_file), 
                               table_data=table_data, 
                               total_anomalies=total_anomalies)

@app.route('/manual_check', methods=['POST'])
def manual_check():
    # Extract values from form input
    features = [
        float(request.form['SourceIP']),
        float(request.form['DestinationIP']),
        float(request.form['SourcePort']),
        float(request.form['DestinationPort']),
        float(request.form['Protocol']),
        float(request.form['BytesSent']),
        float(request.form['BytesReceived']),
        float(request.form['PacketsSent']),
        float(request.form['PacketsReceived']),
        float(request.form['Duration']),
    ]

    # Run anomaly detection
    anomaly_result = detect_anomaly_single(features)

    return render_template('index.html', manual_result=anomaly_result)

@app.route('/results/<filename>')
def results(filename):
    result_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    return render_template('results.html', file_url=url_for('download_file', filename=filename))

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['RESULTS_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
