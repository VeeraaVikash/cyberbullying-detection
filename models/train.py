"""
Training Script
Train BERT model on cyberbullying data
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path

from bert_classifier import BERTClassifier, get_tokenizer
from config import config


class CyberbullyingDataset(Dataset):
    """
    PyTorch Dataset for cyberbullying detection
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Args:
            texts: List of text strings
            labels: List of labels (0 or 1)
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_data(train_path, val_path):
    """
    Load training and validation data
    
    Returns:
        train_df, val_df
    """
    print("\n📂 Loading data...")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    print(f"✓ Train samples: {len(train_df):,}")
    print(f"✓ Val samples: {len(val_df):,}")
    
    return train_df, val_df


def create_data_loaders(train_df, val_df, tokenizer, batch_size, max_length):
    """
    Create PyTorch DataLoaders
    
    Returns:
        train_loader, val_loader
    """
    print("\n📦 Creating data loaders...")
    
    # Create datasets
    train_dataset = CyberbullyingDataset(
        texts=train_df['text'].values,
        labels=train_df['label'].values,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    val_dataset = CyberbullyingDataset(
        texts=val_df['text'].values,
        labels=val_df['label'].values,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


def train_epoch(model, data_loader, optimizer, criterion, device):
    """
    Train for one epoch
    
    Returns:
        avg_loss, accuracy
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(data_loader, desc='Training')
    
    for batch in progress_bar:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}'
        })
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(model, data_loader, criterion, device):
    """
    Validate model
    
    Returns:
        avg_loss, accuracy
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Validation'):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def train():
    """
    Main training function
    """
    print("\n" + "="*70)
    print(" "*20 + "TRAINING BERT CLASSIFIER")
    print("="*70)
    
    # Print configuration
    print(config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and config.DEVICE == 'cuda' else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    if device.type == 'cpu':
        print("⚠️  Warning: Training on CPU will be slow!")
        print("   Consider using GPU for faster training")
    
    # Set random seed
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    # Load data
    train_df, val_df = load_data(config.TRAIN_DATA, config.VAL_DATA)
    
    # Get tokenizer
    print("\n🔤 Loading tokenizer...")
    tokenizer = get_tokenizer(config.MODEL_NAME)
    print("✓ Tokenizer loaded")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_df, val_df, tokenizer,
        config.BATCH_SIZE, config.MAX_LENGTH
    )
    
    # Create model
    print("\n🧠 Creating model...")
    model = BERTClassifier(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT
    )
    model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created")
    print(f"  Total parameters: {num_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    # Training loop
    print("\n" + "="*70)
    print("🚀 STARTING TRAINING")
    print("="*70)
    
    best_val_accuracy = 0
    patience_counter = 0
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\n📍 Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Print metrics
        print(f"\n📊 Results:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            patience_counter = 0
            
            # Save model
            save_dir = Path(config.MODEL_SAVE_DIR)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / config.MODEL_NAME_SAVE
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_accuracy': best_val_accuracy,
                'config': config
            }, save_path)
            
            print(f"  ✓ Best model saved! (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement (patience: {patience_counter}/{config.PATIENCE})")
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\n⏹️  Early stopping triggered!")
            break
    
    # Training complete
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📊 Best Validation Accuracy: {best_val_accuracy:.4f}")
    print(f"💾 Model saved to: {config.MODEL_SAVE_DIR}/{config.MODEL_NAME_SAVE}")
    print(f"\n📝 Next step: Evaluate model on test set")
    print(f"   Run: python models/evaluate.py")


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during training: {str(e)}")
        raise
