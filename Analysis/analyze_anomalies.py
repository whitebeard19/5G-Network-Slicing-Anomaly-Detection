import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class AnomalyAnalyzer:
    def __init__(self):
        # Load the results and original data
        self.results = pd.read_csv('results/anomaly_detection_results.csv')
        self.original_data = pd.read_csv('data/preprocessed_data.csv')
        
        # Convert is_anomaly to boolean
        self.results['is_anomaly'] = self.results['is_anomaly'].astype(bool)
        
    def analyze_anomalies(self):
        """Analyze detected anomalies"""
        # Combine results with original data
        self.original_data['is_anomaly'] = self.results['is_anomaly']
        self.original_data['reconstruction_error'] = self.results['reconstruction_error']
        
        # Get anomaly samples (using boolean comparison)
        anomalies = self.original_data[self.original_data['is_anomaly'] == 1]
        normal = self.original_data[self.original_data['is_anomaly'] == 0]
        
        # Print basic statistics
        print("\nAnomaly Analysis Summary:")
        print("-" * 50)
        print(f"Total samples analyzed: {len(self.original_data)}")
        print(f"Number of anomalies detected: {len(anomalies)}")
        print(f"Anomaly rate: {(len(anomalies)/len(self.original_data))*100:.2f}%")
        
        # Analyze feature distributions for anomalies
        print("\nFeature Statistics for Anomalies:")
        print("-" * 50)
        features = ['bandwidth', 'packets_rate', 'delay', 'jitter', 
                   'loss_rate', 'bandwidth_change', 'throughput']
        
        for feature in features:
            normal_mean = normal[feature].mean()
            anomaly_mean = anomalies[feature].mean()
            print(f"\n{feature.capitalize()}:")
            print(f"Normal mean: {normal_mean:.4f}")
            print(f"Anomaly mean: {anomaly_mean:.4f}")
            print(f"Difference: {((anomaly_mean-normal_mean)/normal_mean)*100:.2f}%")
    
    def plot_anomalies(self):
        """Create visualizations for anomaly analysis"""
        # Create directory for plots
        os.makedirs('results/plots', exist_ok=True)
        
        # 1. Reconstruction Error Distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(data=self.results, x='reconstruction_error', hue='is_anomaly', bins=50)
        plt.title('Reconstruction Error Distribution')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Count')
        plt.savefig('results/plots/reconstruction_error_dist.png')
        plt.close()
        
        # 2. Feature Distributions
        features = ['bandwidth', 'packets_rate', 'delay', 'jitter', 
                   'loss_rate', 'bandwidth_change', 'throughput']
        
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(features, 1):
            plt.subplot(3, 3, i)
            sns.boxplot(data=self.original_data, x='is_anomaly', y=feature)
            plt.title(f'{feature.capitalize()} Distribution')
        plt.tight_layout()
        plt.savefig('results/plots/feature_distributions.png')
        plt.close()
        
        # 3. Anomaly Timeline
        plt.figure(figsize=(15, 6))
        plt.scatter(range(len(self.results)), 
                   self.results['reconstruction_error'],
                   c=self.results['is_anomaly'],
                   alpha=0.5)
        plt.axhline(y=self.results[self.results['is_anomaly']]['reconstruction_error'].min(),
                   color='r', linestyle='--', label='Anomaly Threshold')
        plt.title('Anomaly Timeline')
        plt.xlabel('Sample Index')
        plt.ylabel('Reconstruction Error')
        plt.legend()
        plt.savefig('results/plots/anomaly_timeline.png')
        plt.close()
    
    def export_detailed_report(self):
        """Export detailed analysis report"""
        # Get anomalies with all features (using explicit comparison)
        anomalies = self.original_data[self.original_data['is_anomaly'] == 1].copy()
        anomalies['reconstruction_error'] = self.results[self.results['is_anomaly'] == 1]['reconstruction_error']
        
        # Sort by reconstruction error
        anomalies = anomalies.sort_values('reconstruction_error', ascending=False)
        
        # Save detailed report
        report_path = 'results/anomaly_analysis_report.csv'
        anomalies.to_csv(report_path, index=False)
        print(f"\nDetailed analysis report saved to: {report_path}")

def main():
    # Create analyzer instance
    analyzer = AnomalyAnalyzer()
    
    # Run analysis
    print("Starting anomaly analysis...")
    analyzer.analyze_anomalies()
    
    # Create visualizations
    print("\nGenerating visualizations...")
    analyzer.plot_anomalies()
    
    # Export detailed report
    analyzer.export_detailed_report()
    
    print("\nAnalysis completed! Check the results directory for detailed reports and visualizations.")

if __name__ == "__main__":
    main() 