"""
Evaluation Script
Test trained model on test set
"""

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from pathlib import Path
from tqdm import tqdm

from bert_classifier import BERTClassifier, get_tokenizer
from config import config
from train import CyberbullyingDataset
from torch.utils.data import DataLoader


def load_model(model_path, device):
    """
    Load trained model
    
    Args:
        model_path: Path to saved model
        device: Device to load model on
        
    Returns:
        model: Loaded model
    """
    print(f"\n📥 Loading model from: {model_path}")
    
    # Create model
    model = BERTClassifier(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded")
    print(f"  Best validation accuracy: {checkpoint['best_val_accuracy']:.4f}")
    
    return model


def predict(model, data_loader, device):
    """
    Make predictions on dataset
    
    Returns:
        predictions, true_labels
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Predicting'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels)


def evaluate():
    """
    Main evaluation function
    """
    print("\n" + "="*70)
    print(" "*20 + "EVALUATING MODEL")
    print("="*70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and config.DEVICE == 'cuda' else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Check if model exists
    model_path = Path(config.MODEL_SAVE_DIR) / config.MODEL_NAME_SAVE
    if not model_path.exists():
        print(f"\n❌ Error: Model not found at {model_path}")
        print(f"   Please train the model first: python models/train.py")
        return
    
    # Load model
    model = load_model(model_path, device)
    
    # Load test data
    print(f"\n📂 Loading test data...")
    test_df = pd.read_csv(config.TEST_DATA)
    print(f"✓ Test samples: {len(test_df):,}")
    
    # Get tokenizer
    print("\n🔤 Loading tokenizer...")
    tokenizer = get_tokenizer(config.MODEL_NAME)
    print("✓ Tokenizer loaded")
    
    # Create test dataset and loader
    print("\n📦 Creating test loader...")
    test_dataset = CyberbullyingDataset(
        texts=test_df['text'].values,
        labels=test_df['label'].values,
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    print(f"✓ Test batches: {len(test_loader)}")
    
    # Make predictions
    print("\n🔮 Making predictions...")
    predictions, true_labels = predict(model, test_loader, device)
    
    # Calculate metrics
    print("\n" + "="*70)
    print("📊 EVALUATION RESULTS")
    print("="*70)
    
    # Accuracy
    accuracy = accuracy_score(true_labels, predictions)
    print(f"\n✓ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Precision, Recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, average='binary'
    )
    print(f"✓ Precision: {precision:.4f}")
    print(f"✓ Recall: {recall:.4f}")
    print(f"✓ F1-Score: {f1:.4f}")
    
    # Per-class metrics
    print("\n" + "-"*70)
    print("Per-Class Metrics:")
    print("-"*70)
    class_names = ['Not Cyberbullying', 'Cyberbullying']
    print(classification_report(true_labels, predictions, target_names=class_names))
    
    # Confusion Matrix
    print("-"*70)
    print("Confusion Matrix:")
    print("-"*70)
    cm = confusion_matrix(true_labels, predictions)
    print(f"\n              Predicted")
    print(f"              Not CB  |  CB")
    print(f"Actual  Not CB  {cm[0][0]:5d}  | {cm[0][1]:5d}")
    print(f"        CB      {cm[1][0]:5d}  | {cm[1][1]:5d}")
    
    # Additional statistics
    print("\n" + "-"*70)
    print("Additional Statistics:")
    print("-"*70)
    tn, fp, fn, tp = cm.ravel()
    
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)  # Same as recall
    
    print(f"True Negatives:  {tn:5d}")
    print(f"False Positives: {fp:5d}")
    print(f"False Negatives: {fn:5d}")
    print(f"True Positives:  {tp:5d}")
    print(f"\nSpecificity: {specificity:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    
    # Final summary
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    print(f"\n📈 Summary:")
    print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Model: {model_path}")
    
    print(f"\n💡 Model Performance:")
    if accuracy >= 0.90:
        print("  🌟 Excellent! Model performs very well.")
    elif accuracy >= 0.85:
        print("  ✅ Good! Model performs well.")
    elif accuracy >= 0.80:
        print("  👍 Decent. Room for improvement.")
    else:
        print("  ⚠️  Needs improvement. Consider:")
        print("     - Training for more epochs")
        print("     - Adjusting hyperparameters")
        print("     - Using more data")


if __name__ == "__main__":
    try:
        evaluate()
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        raise
