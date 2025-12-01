"""
COMPREHENSIVE MODEL EVALUATION
Proper metrics for imbalanced cyberbullying detection
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)
from transformers import BertTokenizer
import sys
import os

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from bert_classifier import BERTClassifier

def load_model(model_path, device):
    """Load trained model from checkpoint"""
    print(f"Loading model from {model_path}...")
    
    model = BERTClassifier(num_classes=2, dropout=0.3)
    
    # Load checkpoint (contains model_state_dict, optimizer, etc.)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract just the model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Loaded from checkpoint format")
    else:
        # If it's just weights directly
        model.load_state_dict(checkpoint)
        print("✓ Loaded from weights format")
    
    model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully")
    return model


def load_test_data(test_path):
    """Load test dataset"""
    print(f"\nLoading test data from {test_path}...")
    
    test_df = pd.read_csv(test_path, encoding='utf-8')
    print(f"✓ Loaded {len(test_df):,} test samples")
    
    return test_df


def get_predictions(model, tokenizer, texts, device, batch_size=16):
    """Get model predictions with probabilities"""
    print("\nGenerating predictions...")
    
    all_preds = []
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            encoding = tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)
            
            # Get predictions
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    print(f"✓ Generated predictions for {len(all_preds):,} samples")
    
    return np.array(all_preds), np.array(all_probs)


def calculate_metrics(y_true, y_pred, y_probs):
    """Calculate comprehensive metrics"""
    print("\n" + "="*70)
    print("CALCULATING METRICS")
    print("="*70)
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='binary', pos_label=1)
    metrics['recall'] = recall_score(y_true, y_pred, average='binary', pos_label=1)
    metrics['f1'] = f1_score(y_true, y_pred, average='binary', pos_label=1)
    
    # ROC-AUC
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
    metrics['roc_auc'] = auc(fpr, tpr)
    
    # Precision-Recall AUC
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_probs[:, 1])
    metrics['pr_auc'] = average_precision_score(y_true, y_probs[:, 1])
    
    # Per-class metrics
    report = classification_report(y_true, y_pred, 
                                   target_names=['Not CB', 'CB'],
                                   output_dict=True)
    
    metrics['class_0_precision'] = report['Not CB']['precision']
    metrics['class_0_recall'] = report['Not CB']['recall']
    metrics['class_0_f1'] = report['Not CB']['f1-score']
    
    metrics['class_1_precision'] = report['CB']['precision']
    metrics['class_1_recall'] = report['CB']['recall']
    metrics['class_1_f1'] = report['CB']['f1-score']
    
    return metrics, report


def print_metrics(metrics, report):
    """Print metrics in readable format"""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    print("\n📊 OVERALL METRICS:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f} (⚠️  Can be misleading with imbalance!)")
    print(f"  Precision:   {metrics['precision']:.4f} ⭐")
    print(f"  Recall:      {metrics['recall']:.4f} ⭐")
    print(f"  F1-Score:    {metrics['f1']:.4f} ⭐ (MOST IMPORTANT)")
    print(f"  ROC-AUC:     {metrics['roc_auc']:.4f} ⭐")
    print(f"  PR-AUC:      {metrics['pr_auc']:.4f} ⭐")
    
    print("\n📊 PER-CLASS METRICS:")
    print("\n  NOT CYBERBULLYING (Class 0):")
    print(f"    Precision: {metrics['class_0_precision']:.4f}")
    print(f"    Recall:    {metrics['class_0_recall']:.4f}")
    print(f"    F1-Score:  {metrics['class_0_f1']:.4f}")
    
    print("\n  CYBERBULLYING (Class 1):")
    print(f"    Precision: {metrics['class_1_precision']:.4f} (How many flagged are actually CB)")
    print(f"    Recall:    {metrics['class_1_recall']:.4f} (How many CB messages we catch)")
    print(f"    F1-Score:  {metrics['class_1_f1']:.4f}")
    
    print("\n" + "="*70)
    print("INTERPRETATION:")
    print("="*70)
    print(f"""
✓ F1-Score {metrics['f1']:.4f}: {'EXCELLENT ⭐⭐⭐⭐⭐' if metrics['f1'] > 0.95 else 'VERY GOOD ⭐⭐⭐⭐' if metrics['f1'] > 0.90 else 'GOOD ⭐⭐⭐' if metrics['f1'] > 0.85 else 'NEEDS IMPROVEMENT ⚠️'}
✓ Recall {metrics['recall']:.4f}: Catching {metrics['recall']*100:.1f}% of cyberbullying
✓ Precision {metrics['precision']:.4f}: {metrics['precision']*100:.1f}% of flagged messages are actually CB
✓ ROC-AUC {metrics['roc_auc']:.4f}: {'EXCELLENT' if metrics['roc_auc'] > 0.95 else 'VERY GOOD' if metrics['roc_auc'] > 0.90 else 'GOOD' if metrics['roc_auc'] > 0.85 else 'NEEDS IMPROVEMENT'}

⚠️  CRITICAL: For cyberbullying, HIGH RECALL is most important!
   Missing actual cyberbullying (false negatives) is DANGEROUS.
   False positives (flagging normal text) are annoying but safer.
""")


def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    print("\nCreating confusion matrix...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not CB', 'CB'],
                yticklabels=['Not CB', 'CB'],
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add percentages
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    
    plt.text(0.5, -0.15, f'TN: {tn:,} ({tn/total*100:.1f}%)', 
             transform=plt.gca().transAxes, ha='center')
    plt.text(1.5, -0.15, f'FP: {fp:,} ({fp/total*100:.1f}%)', 
             transform=plt.gca().transAxes, ha='center')
    plt.text(0.5, -0.20, f'FN: {fn:,} ({fn/total*100:.1f}%) ⚠️  DANGEROUS!', 
             transform=plt.gca().transAxes, ha='center', color='red')
    plt.text(1.5, -0.20, f'TP: {tp:,} ({tp/total*100:.1f}%) ✓', 
             transform=plt.gca().transAxes, ha='center', color='green')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved confusion matrix to {save_path}")
    plt.close()


def plot_roc_curve(y_true, y_probs, save_path='roc_curve.png'):
    """Plot ROC curve"""
    print("Creating ROC curve...")
    
    fpr, tpr, thresholds = roc_curve(y_true, y_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier (AUC = 0.5)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('ROC Curve - Model Performance', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved ROC curve to {save_path}")
    plt.close()


def plot_precision_recall_curve(y_true, y_probs, save_path='pr_curve.png'):
    """Plot Precision-Recall curve"""
    print("Creating Precision-Recall curve...")
    
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_probs[:, 1])
    pr_auc = average_precision_score(y_true, y_probs[:, 1])
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall_vals, precision_vals, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved PR curve to {save_path}")
    plt.close()


def analyze_errors(test_df, y_true, y_pred, y_probs, num_examples=10):
    """Analyze misclassified examples"""
    print("\n" + "="*70)
    print("ERROR ANALYSIS")
    print("="*70)
    
    # Find false positives and false negatives
    fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]
    
    print(f"\nFalse Positives: {len(fp_indices)} (normal text flagged as CB)")
    print(f"False Negatives: {len(fn_indices)} (CB missed) ⚠️  DANGEROUS!")
    
    # Show examples
    print("\n🔴 FALSE NEGATIVES (Missed Cyberbullying - Most Critical!):")
    print("-" * 70)
    for i, idx in enumerate(fn_indices[:num_examples]):
        text = test_df.iloc[idx]['text']
        prob_cb = y_probs[idx][1]
        print(f"\n{i+1}. Text: {text[:100]}...")
        print(f"   Probability CB: {prob_cb:.4f} (threshold: 0.5)")
        print(f"   Status: MISSED ⚠️")
    
    print("\n\n🟡 FALSE POSITIVES (Normal flagged as CB):")
    print("-" * 70)
    for i, idx in enumerate(fp_indices[:num_examples]):
        text = test_df.iloc[idx]['text']
        prob_cb = y_probs[idx][1]
        print(f"\n{i+1}. Text: {text[:100]}...")
        print(f"   Probability CB: {prob_cb:.4f}")
        print(f"   Status: FALSE ALARM")


def save_results(metrics, report, output_file='evaluation_results.txt'):
    """Save results to file"""
    print(f"\nSaving results to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("COMPREHENSIVE EVALUATION RESULTS\n")
        f.write("="*70 + "\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write(f"  Accuracy:    {metrics['accuracy']:.4f}\n")
        f.write(f"  Precision:   {metrics['precision']:.4f}\n")
        f.write(f"  Recall:      {metrics['recall']:.4f}\n")
        f.write(f"  F1-Score:    {metrics['f1']:.4f}\n")
        f.write(f"  ROC-AUC:     {metrics['roc_auc']:.4f}\n")
        f.write(f"  PR-AUC:      {metrics['pr_auc']:.4f}\n\n")
        
        f.write("DETAILED CLASSIFICATION REPORT:\n")
        f.write(str(report))
    
    print(f"✓ Results saved to {output_file}")


def main():
    """Main evaluation pipeline"""
    
    print("="*70)
    print("     COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Paths
    model_path = 'models/saved_models/bert_cyberbullying_improved.pth'
    test_path = 'data/processed_augmented/test.csv'
    
    # Try alternative paths if not found
    if not os.path.exists(test_path):
        test_path = 'data/processed_augmented/test.csv'
    if not os.path.exists(test_path):
        test_path = 'data/processed/test.csv'
    
    # Load model
    model = load_model(model_path, device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load test data
    test_df = load_test_data(test_path)
    
    # Get predictions
    texts = test_df['text'].tolist()
    y_true = test_df['label'].values
    
    y_pred, y_probs = get_predictions(model, tokenizer, texts, device)
    
    # Calculate metrics
    metrics, report = calculate_metrics(y_true, y_pred, y_probs)
    
    # Print metrics
    print_metrics(metrics, report)
    
    # Create visualizations
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(y_true, y_probs)
    plot_precision_recall_curve(y_true, y_probs)
    
    # Error analysis
    analyze_errors(test_df, y_true, y_pred, y_probs)
    
    # Save results
    save_results(metrics, report)
    
    print("\n" + "="*70)
    print("✓ EVALUATION COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - confusion_matrix.png")
    print("  - roc_curve.png")
    print("  - pr_curve.png")
    print("  - evaluation_results.txt")


if __name__ == "__main__":
    main()
