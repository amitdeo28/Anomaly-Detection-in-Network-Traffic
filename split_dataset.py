import pandas as pd
import os
import zipfile

# Load dataset
file_path = r"A:\project_x\Anomaly_Detection\data\synthetic_network_traffic.csv"  # Change this to your actual dataset path
df = pd.read_csv(file_path)

# Define chunk size
chunk_size = 10000
num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size != 0 else 0)

# Create directory for chunked datasets
output_dir = "chunked_datasets"
os.makedirs(output_dir, exist_ok=True)

# Split dataset into chunks
chunk_files = []
for i, chunk in enumerate(range(0, len(df), chunk_size)):
    chunk_df = df.iloc[chunk:chunk + chunk_size]
    chunk_filename = f"synthetic_network_traffic_part_{i+1}.csv"
    chunk_filepath = os.path.join(output_dir, chunk_filename)
    chunk_df.to_csv(chunk_filepath, index=False)
    chunk_files.append(chunk_filepath)

# Create a zip file containing all chunks
zip_filepath = r"A:\project_x\Anomaly_Detection\uploads\synthetic_network_traffic_chunks.zip"
with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file in chunk_files:
        zipf.write(file, os.path.basename(file))

print(f"✅ Dataset split into {num_chunks} chunks and saved as a ZIP file: {zip_filepath}")
