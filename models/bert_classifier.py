"""
BERT Classifier for Cyberbullying Detection
Simple BERT-based binary classifier
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class BERTClassifier(nn.Module):
    """
    BERT-based text classifier
    
    Architecture:
        Input Text
            ↓
        BERT Encoder (bert-base-uncased)
            ↓
        [CLS] Token Output (768 dimensions)
            ↓
        Dropout (0.3)
            ↓
        Linear Layer (768 → 2)
            ↓
        Output (2 classes: 0 or 1)
    """
    
    def __init__(self, model_name='bert-base-uncased', num_classes=2, dropout=0.3):
        """
        Initialize BERT classifier
        
        Args:
            model_name: Pre-trained BERT model name
            num_classes: Number of output classes (2 for binary)
            dropout: Dropout rate
        """
        super(BERTClassifier, self).__init__()
        
        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained(model_name)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Classification layer
        # BERT outputs 768 dimensions, we map to num_classes
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Args:
            input_ids: Token IDs [batch_size, max_length]
            attention_mask: Attention mask [batch_size, max_length]
            
        Returns:
            logits: Class predictions [batch_size, num_classes]
        """
        
        # Pass through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get [CLS] token output (first token)
        pooled_output = outputs.pooler_output  # [batch_size, 768]
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)  # [batch_size, num_classes]
        
        return logits


def get_tokenizer(model_name='bert-base-uncased'):
    """
    Get BERT tokenizer
    
    Args:
        model_name: Model name
        
    Returns:
        tokenizer: BERT tokenizer
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return tokenizer


def tokenize_text(text, tokenizer, max_length=128):
    """
    Tokenize text for BERT
    
    Args:
        text: Input text string
        tokenizer: BERT tokenizer
        max_length: Maximum sequence length
        
    Returns:
        encoding: Dictionary with input_ids and attention_mask
    """
    encoding = tokenizer(
        text,
        add_special_tokens=True,      # Add [CLS] and [SEP]
        max_length=max_length,         # Truncate to max_length
        padding='max_length',          # Pad to max_length
        truncation=True,               # Truncate if longer
        return_attention_mask=True,    # Return attention mask
        return_tensors='pt'            # Return PyTorch tensors
    )
    
    return encoding


if __name__ == "__main__":
    print("Testing BERT Classifier...")
    print("=" * 60)
    
    # Create model
    model = BERTClassifier(num_classes=2, dropout=0.3)
    print("✓ Model created")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Get tokenizer
    tokenizer = get_tokenizer()
    print("✓ Tokenizer loaded")
    
    # Test with sample text
    sample_text = "You are stupid and ugly"
    encoding = tokenize_text(sample_text, tokenizer)
    print(f"✓ Tokenized sample text: '{sample_text}'")
    print(f"  Input shape: {encoding['input_ids'].shape}")
    
    # Forward pass
    with torch.no_grad():
        logits = model(
            encoding['input_ids'],
            encoding['attention_mask']
        )
    print(f"✓ Forward pass successful")
    print(f"  Output shape: {logits.shape}")
    print(f"  Output logits: {logits}")
    
    # Get prediction
    prediction = torch.argmax(logits, dim=1)
    print(f"  Prediction: {prediction.item()} (0=not cyberbullying, 1=cyberbullying)")
    
    print("\n" + "=" * 60)
    print("✓ BERT Classifier working correctly!")
