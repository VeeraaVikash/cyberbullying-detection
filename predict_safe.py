"""
Prediction Script - Loads only model weights (no config dependency)
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import sys
import os

# Add models directory to Python path so torch.load can find config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))


class BERTClassifier(nn.Module):
    """BERT model for classification"""
    def __init__(self, model_name='bert-base-uncased', num_classes=2, dropout=0.3):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def load_model_safe(model_path, device):
    """
    Load model safely - extracts only the model weights
    """
    print("Loading model (safe mode)...")
    
    # Create model
    model = BERTClassifier(
        model_name='bert-base-uncased',
        num_classes=2,
        dropout=0.3
    )
    
    try:
        # Try loading with custom unpickler that ignores config
        import pickle
        import io
        
        class ConfiglessUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Ignore config module
                if module == 'config':
                    # Return a dummy class
                    class DummyConfig:
                        pass
                    return DummyConfig
                return super().find_class(module, name)
        
        # Load file
        with open(model_path, 'rb') as f:
            # Read as bytes
            buffer = io.BytesIO(f.read())
            checkpoint = torch.load(buffer, map_location=device, weights_only=False)
        
        # Load only model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print("✓ Model loaded successfully!")
        if 'best_val_accuracy' in checkpoint:
            print(f"  Validation accuracy: {checkpoint['best_val_accuracy']:.4f}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTrying alternative method...")
        
        # Alternative: Just load state dict directly
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print("✓ Model loaded!")
    
    return model


def predict_text(model, tokenizer, text, device):
    """Predict single text"""
    
    # Tokenize
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
        prob_not_cb = probabilities[0][0].item()
        prob_cb = probabilities[0][1].item()
    
    return prediction, confidence, prob_not_cb, prob_cb


def main():
    print("\n" + "="*70)
    print(" "*20 + "CYBERBULLYING DETECTION")
    print("="*70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load model
    model_path = 'models/saved_models/bert_cyberbullying_model.pth'
    model = load_model_safe(model_path, device)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("✓ Tokenizer loaded\n")
    
    # Test texts
    test_texts = [
        "You are stupid and ugly",
        "Great work! Keep it up!",
        "I hate you so much, loser",
        "Have a wonderful day!",
        "Nobody likes you, go away",
        "Your presentation was excellent",
        "You're so dumb, can't believe you did that",
        "Thanks for helping me today"
    ]
    
    print("="*70)
    print("TESTING WITH SAMPLE TEXTS")
    print("="*70)
    
    for i, text in enumerate(test_texts, 1):
        prediction, confidence, prob_not_cb, prob_cb = predict_text(
            model, tokenizer, text, device
        )
        
        label = "🚨 CYBERBULLYING" if prediction == 1 else "✅ NOT CYBERBULLYING"
        
        print(f"\n[{i}] '{text}'")
        print(f"    → {label} ({confidence*100:.1f}% confident)")
        print(f"    Probabilities: Not CB={prob_not_cb*100:.1f}% | CB={prob_cb*100:.1f}%")
    
    # Interactive mode
    print("\n" + "="*70)
    print("INTERACTIVE MODE - Type your own text!")
    print("="*70)
    print("(Type 'quit' to exit)\n")
    
    while True:
        try:
            text = input("Enter text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q', '']:
                if text.lower() in ['quit', 'exit', 'q']:
                    print("\n✓ Goodbye!")
                break
            
            prediction, confidence, prob_not_cb, prob_cb = predict_text(
                model, tokenizer, text, device
            )
            
            label = "🚨 CYBERBULLYING" if prediction == 1 else "✅ NOT CYBERBULLYING"
            print(f"→ {label} ({confidence*100:.1f}% confident)")
            print(f"  Probabilities: Not CB={prob_not_cb*100:.1f}% | CB={prob_cb*100:.1f}%\n")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted")
            break
        except EOFError:
            print("\n✓ Goodbye!")
            break


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
