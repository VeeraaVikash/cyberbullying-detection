"""
Training Configuration
All hyperparameters and settings for model training
"""

class TrainingConfig:
    """Configuration for training the cyberbullying detection model"""
    
    # Model settings
    MODEL_NAME = 'bert-base-uncased'  # Pre-trained BERT model
    NUM_CLASSES = 2                    # Binary: 0=not cyberbullying, 1=cyberbullying
    MAX_LENGTH = 128                   # Maximum text length (tokens)
    
    # Training hyperparameters
    BATCH_SIZE = 16                    # Batch size (reduce if out of memory)
    NUM_EPOCHS = 3                     # Number of training epochs
    LEARNING_RATE = 2e-5               # Learning rate
    WARMUP_STEPS = 500                 # Warmup steps for learning rate
    
    # Dropout
    DROPOUT = 0.3                      # Dropout rate
    
    # Data paths
    TRAIN_DATA = 'data/processed_augmented/train.csv'
    VAL_DATA = 'data/processed_augmented/val.csv'
    TEST_DATA = 'data/processed_augmented/test.csv'
    
    # Model save path
    MODEL_SAVE_DIR = 'models/saved_models'
    MODEL_NAME_SAVE = 'bert_cyberbullying_model.pth'
    
    # Device
    DEVICE = 'cuda'  # Use 'cuda' for GPU, 'cpu' for CPU
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    # Early stopping
    PATIENCE = 2  # Stop if no improvement after N epochs
    
    def __str__(self):
        """Print configuration"""
        return f"""
Training Configuration:
-----------------------
Model: {self.MODEL_NAME}
Classes: {self.NUM_CLASSES}
Max Length: {self.MAX_LENGTH}
Batch Size: {self.BATCH_SIZE}
Epochs: {self.NUM_EPOCHS}
Learning Rate: {self.LEARNING_RATE}
Device: {self.DEVICE}
        """


# Create default config instance
config = TrainingConfig()


if __name__ == "__main__":
    # Test config
    print(config)
    print("\n✓ Configuration loaded successfully!")
