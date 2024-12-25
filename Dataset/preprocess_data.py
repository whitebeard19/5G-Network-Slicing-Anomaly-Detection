import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import os

class DataPreprocessor:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.scaler = StandardScaler()
        
        # Create directories if they don't exist
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
    
    def load_data(self):
        """Load data from CSV file"""
        print("Loading data...")
        df = pd.read_csv(self.data_path)
        print(f"Loaded {len(df)} records")
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        print("\nHandling missing values...")
        print("Missing values before:", df.isnull().sum())
        
        # Fill missing values with appropriate methods
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        print("Missing values after:", df.isnull().sum())
        return df
    
    def scale_features(self, df):
        """Scale numerical features"""
        print("\nScaling features...")
        
        # Select features for scaling (excluding categorical and temporal columns)
        features_to_scale = ['bandwidth', 'packets_rate', 'delay', 'jitter', 
                           'loss_rate', 'bandwidth_change', 'throughput']
        
        # Scale the selected features
        scaled_features = self.scaler.fit_transform(df[features_to_scale])
        
        # Create new dataframe with scaled features
        scaled_df = pd.DataFrame(scaled_features, columns=features_to_scale)
        
        # Add back the non-scaled columns
        for col in df.columns:
            if col not in features_to_scale:
                scaled_df[col] = df[col]
        
        print("Features scaled successfully")
        return scaled_df
    
    def prepare_training_data(self, df):
        """Prepare data for training"""
        print("\nPreparing training data...")
        
        # Select features for training (excluding metadata columns)
        training_features = ['bandwidth', 'packets_rate', 'delay', 'jitter', 
                           'loss_rate', 'bandwidth_change', 'throughput']
        
        X = df[training_features].values
        
        # First split: separate test set
        X_train_val, X_test = train_test_split(X, test_size=0.2, random_state=42)
        
        # Second split: separate train and validation sets
        X_train, X_val = train_test_split(X_train_val, test_size=0.2, random_state=42)
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Validation set shape: {X_val.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        return X_train, X_val, X_test
    
    def process_data(self):
        """Complete data preprocessing pipeline"""
        # Load data
        df = self.load_data()
        
        # Print initial statistics
        print("\nInitial data statistics:")
        print(df.describe())
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Scale features
        df_scaled = self.scale_features(df)
        
        # Prepare training data
        X_train, X_val, X_test = self.prepare_training_data(df_scaled)
        
        # Save processed data
        np.save('data/X_train.npy', X_train)
        np.save('data/X_val.npy', X_val)
        np.save('data/X_test.npy', X_test)
        
        # Save scaler for later use
        import joblib
        joblib.dump(self.scaler, 'models/scaler.save')
        
        # Save preprocessed DataFrame for reference
        df_scaled.to_csv('data/preprocessed_data.csv', index=False)
        
        return X_train, X_val, X_test

def main():
    # Initialize preprocessor
    preprocessor = DataPreprocessor("network_data.csv")
    
    # Process data
    X_train, X_val, X_test = preprocessor.process_data()
    
    print("\nPreprocessing completed successfully!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Print sample statistics
    print("\nTraining set statistics:")
    print(pd.DataFrame(X_train).describe())

if __name__ == "__main__":
    main() 