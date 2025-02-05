import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create the output directory if it doesn't exist
output_dir = "dashboard_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --------------------------
# 1. Anomaly Visualization
# --------------------------
# Read anomaly detection results CSV
results_df = pd.read_csv('results/anomaly_detection_results.csv')

plt.figure(figsize=(10, 6))
plt.plot(results_df['reconstruction_error'], label="Reconstruction Error")
plt.axhline(y=20.0, color='red', linestyle='--', label="Critical Threshold")
plt.axhline(y=10.0, color='yellow', linestyle='--', label="Warning Threshold")
plt.title("Reconstruction Error Timeline")
plt.xlabel("Sample Index")
plt.ylabel("Reconstruction Error")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "anomaly_visualization.png"))
plt.close()

# ------------------------------------
# 2. Feature Correlation Matrix
# ------------------------------------
# Read network data CSV
network_data = pd.read_csv('Dataset/network_data.csv')
# Select the desired features
features = ['bandwidth', 'packets_rate', 'delay', 'jitter', 'loss_rate']
corr = network_data[features].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_correlation_matrix.png"))
plt.close()

# --------------------------
# 3. Traffic Metrics Plot
# --------------------------
plt.figure(figsize=(10, 6))
# Use timestamp if available, otherwise use the index.
if 'timestamp' in network_data.columns:
    x = pd.to_datetime(network_data['timestamp'])
else:
    x = network_data.index
plt.plot(x, network_data['bandwidth'], label='Bandwidth')
plt.plot(x, network_data['packets_rate'], label='Packets Rate')
if 'throughput' in network_data.columns:
    plt.plot(x, network_data['throughput'], label='Throughput')
plt.title("Traffic Metrics Over Time")
plt.xlabel("Time" if 'timestamp' in network_data.columns else "Sample Index")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "traffic_metrics.png"))
plt.close()

# -------------------------------
# 4. Performance Metrics Plot
# -------------------------------
plt.figure(figsize=(10, 6))
if 'timestamp' in network_data.columns:
    x = pd.to_datetime(network_data['timestamp'])
else:
    x = network_data.index
plt.plot(x, network_data['delay'], label='Delay')
plt.plot(x, network_data['jitter'], label='Jitter')
plt.plot(x, network_data['loss_rate'], label='Loss Rate')
plt.title("Network Performance Metrics Over Time")
plt.xlabel("Time" if 'timestamp' in network_data.columns else "Sample Index")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "performance_metrics.png"))
plt.close()

# --------------------------
# 5. Anomalies List
# --------------------------
# Filter rows where anomalies are detected (assuming 'is_anomaly' is 1 or True)
if results_df['is_anomaly'].dtype == 'bool':
    anomalies = results_df[results_df['is_anomaly']]
else:
    anomalies = results_df[results_df['is_anomaly'] == 1]

# Generate a table plot using matplotlib
plt.figure(figsize=(10, 6))
plt.axis('off')
table = plt.table(cellText=anomalies.values,
                  colLabels=anomalies.columns,
                  loc='center',
                  cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
plt.title("Detected Anomalies")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "anomalies_list.png"))
plt.close()

print("Dashboard images generated and saved in the folder:", output_dir) 