import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

class AnomalyDetector:
    def __init__(self, model_path, threshold_path):
        self.model = keras.models.load_model(model_path)
        self.threshold = np.load(threshold_path)
    
    def detect_anomalies(self, data):
        """Detect anomalies in the data"""
        # Get reconstruction error
        predictions = self.model.predict(data)
        mse = np.mean(np.power(data - predictions, 2), axis=1)
        
        # Classify as anomaly if error > threshold
        anomalies = mse > self.threshold
        
        return anomalies, mse
    
    def plot_reconstruction_error(self, mse, anomalies):
        """Plot reconstruction error and threshold"""
        plt.figure(figsize=(12, 6))
        plt.hist(mse, bins=50)
        plt.axvline(x=self.threshold, color='r', linestyle='--', label='Threshold')
        plt.title('Reconstruction Error Distribution')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig('models/reconstruction_error_dist.png')
        plt.close()
    
    def analyze_results(self, data, anomalies, mse):
        """Analyze and visualize detection results"""
        results = pd.DataFrame({
            'reconstruction_error': mse,
            'is_anomaly': anomalies
        })
        
        print("\nAnomaly Detection Results:")
        print(f"Total samples: {len(data)}")
        print(f"Normal samples: {sum(~anomalies)}")
        print(f"Anomalies detected: {sum(anomalies)}")
        print(f"Anomaly percentage: {(sum(anomalies)/len(data))*100:.2f}%")
        
        return results

def main():
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Set paths for local computer
    MODEL_PATH = 'models/autoencoder.keras'
    THRESHOLD_PATH = 'models/reconstruction_threshold.npy'
    DATA_PATH = 'data'
    
    # Load test data
    print("Loading test data...")
    X_test = np.load(f'{DATA_PATH}/X_test.npy')
    
    # Initialize detector
    detector = AnomalyDetector(
        model_path=MODEL_PATH,
        threshold_path=THRESHOLD_PATH
    )
    
    # Detect anomalies
    print("\nDetecting anomalies...")
    anomalies, mse = detector.detect_anomalies(X_test)
    
    # Plot reconstruction error distribution
    print("\nPlotting reconstruction error distribution...")
    detector.plot_reconstruction_error(mse, anomalies)
    
    # Analyze results
    results = detector.analyze_results(X_test, anomalies, mse)
    
    # Save results
    results.to_csv('results/anomaly_detection_results.csv', index=False)
    print("\nResults saved to: results/anomaly_detection_results.csv")
    
    # Additional visualization of anomalies
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=results, x=range(len(results)), y='reconstruction_error', 
                   hue='is_anomaly', style='is_anomaly')
    plt.axhline(y=detector.threshold, color='r', linestyle='--', label='Threshold')
    plt.title('Reconstruction Error Over Time')
    plt.xlabel('Sample Index')
    plt.ylabel('Reconstruction Error')
    plt.legend()
    
    plt.savefig('results/anomaly_visualization.png')
    plt.close()

if __name__ == "__main__":
    main() 