# Network Anomaly Detection System

## Project Overview
This project implements an autoencoder-based anomaly detection system for network traffic analysis. It uses deep learning to identify unusual patterns in network behavior that might indicate security issues or performance problems.

## System Architecture

### Components
1. **Data Collection** (`datacollection.py`)
   - Collects network traffic data from multiple sources
   - Processes raw network packets into feature vectors
   - Handles data cleaning and initial formatting

2. **Data Preprocessing** (`preprocess_data.py`)
   - Scales and normalizes features
   - Handles missing values
   - Splits data into training, validation, and test sets

3. **Autoencoder Model** (`autoencoder.py`)
   - Implements deep learning-based anomaly detection
   - Trains on normal network behavior
   - Computes reconstruction error for anomaly detection

4. **Anomaly Detection** (`anomaly_detector.py`)
   - Applies trained model to identify anomalies
   - Uses reconstruction error threshold
   - Generates anomaly reports

5. **Analysis** (`analyze_anomalies.py`)
   - Analyzes detected anomalies
   - Generates visualizations
   - Provides detailed statistics

6. **Evaluation** (`model_evaluation.py`)
   - Evaluates model performance
   - Generates performance metrics
   - Creates evaluation visualizations

## Installation

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- networkx

### Setup
