import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

class ModelEvaluator:
    def __init__(self):
        self.results = pd.read_csv('results/anomaly_detection_results.csv')
        self.analysis = pd.read_csv('results/anomaly_analysis_report.csv')
        
    def calculate_metrics(self):
        """Calculate performance metrics"""
        print("\nModel Performance Metrics:")
        print("-" * 50)
        
        # Calculate basic metrics
        total_samples = len(self.results)
        anomaly_count = sum(self.results['is_anomaly'] == 1)
        normal_count = total_samples - anomaly_count
        
        print(f"Total Samples: {total_samples}")
        print(f"Normal Samples: {normal_count}")
        print(f"Anomaly Samples: {anomaly_count}")
        print(f"Anomaly Rate: {(anomaly_count/total_samples)*100:.2f}%")
        
        # Calculate reconstruction error statistics
        print("\nReconstruction Error Statistics:")
        print("-" * 50)
        print(f"Mean Error: {self.results['reconstruction_error'].mean():.4f}")
        print(f"Std Error: {self.results['reconstruction_error'].std():.4f}")
        print(f"Max Error: {self.results['reconstruction_error'].max():.4f}")
        print(f"Min Error: {self.results['reconstruction_error'].min():.4f}")
        
        # Feature importance analysis
        print("\nFeature Importance Analysis:")
        print("-" * 50)
        features = ['bandwidth', 'packets_rate', 'delay', 'jitter', 
                   'loss_rate', 'bandwidth_change', 'throughput']
        
        for feature in features:
            normal_std = self.analysis[self.analysis['is_anomaly'] == 0][feature].std()
            anomaly_std = self.analysis[self.analysis['is_anomaly'] == 1][feature].std()
            importance = abs(anomaly_std - normal_std) / normal_std
            print(f"{feature}: {importance:.4f}")
    
    def plot_evaluation_metrics(self):
        """Create evaluation visualizations"""
        os.makedirs('results/evaluation', exist_ok=True)
        
        # 1. Error Distribution Comparison
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=self.results, x='reconstruction_error', hue='is_anomaly')
        plt.title('Reconstruction Error Distribution by Class')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Density')
        plt.savefig('results/evaluation/error_distribution.png')
        plt.close()
        
        # 2. Feature Correlations
        plt.figure(figsize=(12, 8))
        features = ['bandwidth', 'packets_rate', 'delay', 'jitter', 
                   'loss_rate', 'bandwidth_change', 'throughput']
        correlation_matrix = self.analysis[features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Feature Correlations')
        plt.tight_layout()
        plt.savefig('results/evaluation/feature_correlations.png')
        plt.close()
    
    def generate_report(self):
        """Generate evaluation report"""
        report = {
            'timestamp': pd.Timestamp.now(),
            'total_samples': len(self.results),
            'anomaly_count': sum(self.results['is_anomaly'] == 1),
            'normal_count': sum(self.results['is_anomaly'] == 0),
            'mean_error': self.results['reconstruction_error'].mean(),
            'std_error': self.results['reconstruction_error'].std(),
            'max_error': self.results['reconstruction_error'].max(),
            'min_error': self.results['reconstruction_error'].min()
        }
        
        # Save report
        pd.DataFrame([report]).to_csv('results/evaluation/evaluation_report.csv', index=False)
        print("\nEvaluation report saved to: results/evaluation/evaluation_report.csv")

def main():
    # Create evaluator instance
    evaluator = ModelEvaluator()
    
    # Calculate metrics
    print("Calculating performance metrics...")
    evaluator.calculate_metrics()
    
    # Create visualizations
    print("\nGenerating evaluation plots...")
    evaluator.plot_evaluation_metrics()
    
    # Generate report
    print("\nGenerating evaluation report...")
    evaluator.generate_report()
    
    print("\nEvaluation completed! Check the results/evaluation directory for outputs.")

if __name__ == "__main__":
    main() 