import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os

class NetworkAutoencoder:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.encoding_dim = 4  # Reduced dimensionality
        self.autoencoder = self.build_autoencoder()
        
    def build_autoencoder(self):
        """Build the autoencoder architecture"""
        # Encoder
        input_layer = Input(shape=(self.input_dim,))
        encoder = Dense(8, activation='relu')(input_layer)
        encoder = Dense(self.encoding_dim, activation='relu')(encoder)
        
        # Decoder
        decoder = Dense(8, activation='relu')(encoder)
        decoder = Dense(self.input_dim, activation='sigmoid')(decoder)
        
        # Autoencoder
        autoencoder = Model(input_layer, decoder)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    
    def train(self, X_train, X_val, epochs=100, batch_size=32):
        """Train the autoencoder"""
        # Create directories for model checkpoints
        os.makedirs('Models/checkpoints', exist_ok=True)
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        checkpoint = ModelCheckpoint(
            'Models/checkpoints/autoencoder_{epoch:02d}.keras',
            save_best_only=True,
            monitor='val_loss'
        )
        
        # Train the model
        history = self.autoencoder.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(X_val, X_val),
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )
        
        return history
    
    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        # Plot training & validation loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig('Models/training_history.png')
        plt.close()
    
    def compute_reconstruction_error(self, X):
        """Compute reconstruction error"""
        X_pred = self.autoencoder.predict(X)
        mse = np.mean(np.power(X - X_pred, 2), axis=1)
        return mse
    
    def save_model(self, path='Models/autoencoder.keras'):
        """Save the trained model"""
        self.autoencoder.save(path)
    
    def load_model(self, path='Models/autoencoder.keras'):
        """Load a trained model"""
        self.autoencoder = tf.keras.Models.load_model(path)

def main():
    # Load preprocessed data
    X_train = np.load('data/X_train.npy')
    X_val = np.load('data/X_val.npy')
    
    print("Training data shape:", X_train.shape)
    print("Validation data shape:", X_val.shape)
    
    # Initialize and train autoencoder
    autoencoder = NetworkAutoencoder(input_dim=X_train.shape[1])
    
    # Train the model
    history = autoencoder.train(X_train, X_val)
    
    # Plot training history
    autoencoder.plot_training_history(history)
    
    # Save the model
    autoencoder.save_model()
    
    # Compute reconstruction error
    train_error = autoencoder.compute_reconstruction_error(X_train)
    val_error = autoencoder.compute_reconstruction_error(X_val)
    
    # Print error statistics
    print("\nReconstruction Error Statistics:")
    print("Training set - Mean: {:.6f}, Std: {:.6f}".format(
        np.mean(train_error), np.std(train_error)))
    print("Validation set - Mean: {:.6f}, Std: {:.6f}".format(
        np.mean(val_error), np.std(val_error)))
    
    # Save error thresholds
    threshold = np.mean(train_error) + 2 * np.std(train_error)
    np.save('Models/reconstruction_threshold.npy', threshold)
    
    print("\nModel training completed!")
    print("Threshold for anomaly detection:", threshold)

if __name__ == "__main__":
    main() 