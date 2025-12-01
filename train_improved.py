"""
IMPROVED TRAINING SCRIPT
Handles class imbalance with:
1. Class weights
2. Focal loss
3. Stratified sampling
4. Better metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import BertTokenizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import sys
import os

# Add models directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from bert_classifier import BERTClassifier


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Focuses more on hard-to-classify examples
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class CyberbullyingDataset(Dataset):
    """Custom dataset with tokenization"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_data_stratified(data_path):
    """Load data with stratified split"""
    print(f"Loading data from {data_path}...")
    
    df = pd.read_csv(data_path, encoding='utf-8')
    print(f"✓ Loaded {len(df):,} samples")
    
    # Class distribution
    class_counts = df['label'].value_counts()
    print(f"\nClass distribution:")
    for label, count in class_counts.items():
        label_name = "Cyberbullying" if label == 1 else "Not Cyberbullying"
        print(f"  {label_name}: {count:,} ({count/len(df)*100:.1f}%)")
    
    return df


def compute_class_weights(labels):
    """Compute class weights for imbalanced data"""
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    print(f"\nComputed class weights:")
    print(f"  Not CB (0): {class_weights[0]:.4f}")
    print(f"  CB (1): {class_weights[1]:.4f}")
    print(f"  Ratio: {class_weights[1]/class_weights[0]:.2f}x more weight on CB")
    
    return torch.tensor(class_weights, dtype=torch.float)


def create_weighted_sampler(labels):
    """Create weighted sampler for balanced batches"""
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = class_weights[labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    print("✓ Created weighted sampler for balanced batches")
    return sampler


def train_epoch(model, dataloader, optimizer, criterion, device, use_focal_loss=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    # Calculate F1 score
    from sklearn.metrics import f1_score, precision_score, recall_score
    f1 = f1_score(all_labels, all_preds, average='binary', pos_label=1)
    precision = precision_score(all_labels, all_preds, average='binary', pos_label=1)
    recall = recall_score(all_labels, all_preds, average='binary', pos_label=1)
    
    return avg_loss, accuracy, f1, precision, recall


def main():
    """Main training pipeline with improvements"""
    
    print("="*70)
    print("     IMPROVED TRAINING WITH CLASS IMBALANCE HANDLING")
    print("="*70)
    
    # Configuration
    TRAIN_DATA = 'data/processed_with_edge_cases/train.csv'
    VAL_DATA = 'data/processed_with_edge_cases/val.csv'
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    USE_FOCAL_LOSS = True
    USE_CLASS_WEIGHTS = True
    USE_WEIGHTED_SAMPLER = True
    
    # Try alternative paths
    if not os.path.exists(TRAIN_DATA):
        TRAIN_DATA = 'data/processed_augmented/train.csv'
        VAL_DATA = 'data/processed_augmented/val.csv'
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    train_df = load_data_stratified(TRAIN_DATA)
    val_df = load_data_stratified(VAL_DATA)
    
    # Initialize tokenizer
    print("\nLoading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = CyberbullyingDataset(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        tokenizer
    )
    val_dataset = CyberbullyingDataset(
        val_df['text'].tolist(),
        val_df['label'].tolist(),
        tokenizer
    )
    
    # Compute class weights
    class_weights = None
    if USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(train_df['label'].values)
        class_weights = class_weights.to(device)
    
    # Create sampler
    sampler = None
    if USE_WEIGHTED_SAMPLER:
        sampler = create_weighted_sampler(train_df['label'].values)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler if sampler else None,
        shuffle=False if sampler else True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    # Initialize model
    print("\nInitializing BERT model...")
    model = BERTClassifier(num_classes=2, dropout=0.3)
    model.to(device)
    
    # Setup loss function
    if USE_FOCAL_LOSS:
        print("Using Focal Loss (better for imbalanced data)")
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    elif USE_CLASS_WEIGHTS:
        print("Using CrossEntropy with class weights")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        print("Using standard CrossEntropy")
        criterion = nn.CrossEntropyLoss()
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    best_f1 = 0.0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, USE_FOCAL_LOSS
        )
        
        # Validate
        val_loss, val_acc, val_f1, val_precision, val_recall = validate(
            model, val_loader, criterion, device
        )
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Val F1:     {val_f1:.4f} ⭐ (MOST IMPORTANT)")
        print(f"  Val Precision: {val_precision:.4f}")
        print(f"  Val Recall:    {val_recall:.4f} (Catching {val_recall*100:.1f}% of CB)")
        
        # Save best model based on F1 score (not accuracy!)
        if val_f1 > best_f1:
            best_f1 = val_f1
            print(f"\n✓ New best F1 score! Saving model...")
            
            save_path = 'models/saved_models/bert_cyberbullying_improved.pth'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  Saved to: {save_path}")
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)
    print(f"\nBest validation F1 score: {best_f1:.4f}")
    print(f"Model saved to: models/saved_models/bert_cyberbullying_improved.pth")
    print("\nNext steps:")
    print("  1. Run: python evaluate_comprehensive.py")
    print("  2. Compare with old model")
    print("  3. Check if F1, Precision, Recall improved")


if __name__ == "__main__":
    main()
