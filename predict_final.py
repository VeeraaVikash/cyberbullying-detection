"""
Simple Prediction Script
Test your trained model on custom texts
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


def load_model(model_path, device):
    """Load trained model"""
    print("Loading model...")
    
    # Create model
    model = BERTClassifier(
        model_name='bert-base-uncased',
        num_classes=2,
        dropout=0.3
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("✓ Model loaded")
    print(f"  Best validation accuracy: {checkpoint['best_val_accuracy']:.4f}")
    
    return model


def predict_texts(model, tokenizer, texts, device):
    """
    Predict cyberbullying for list of texts
    
    Args:
        model: Trained model
        tokenizer: BERT tokenizer
        texts: List of text strings
        device: Device to run on
    """
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    for i, text in enumerate(texts, 1):
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
        
        # Display result
        label = "🚨 CYBERBULLYING" if prediction == 1 else "✅ NOT CYBERBULLYING"
        
        print(f"\n[{i}] Text: '{text}'")
        print(f"    Prediction: {label}")
        print(f"    Confidence: {confidence*100:.1f}%")
        print(f"    Probabilities: Not CB={probabilities[0][0].item()*100:.1f}% | CB={probabilities[0][1].item()*100:.1f}%")
    
    print("\n" + "="*70)


def main():
    """Main function"""
    
    print("\n" + "="*70)
    print(" "*20 + "CYBERBULLYING DETECTION")
    print("="*70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load model
    model_path = 'models/saved_models/bert_cyberbullying_model.pth'
    
    if not os.path.exists(model_path):
        print(f"\n❌ Error: Model not found at {model_path}")
        print("   Please train the model first: py models/train.py")
        return
    
    model = load_model(model_path, device)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("✓ Tokenizer loaded")
    
    # Test texts
    test_texts = [
        "Fuck you bitch",
        "Great work! Keep it up!",
        "I hate you so much, loser",
        "Have a wonderful day!",
        "Nobody likes you, go away",
        "Your presentation was excellent",
        "You're so dumb, can't believe you did that",
        "Thanks for helping me today"
    ]
    
    print("\n🤖 Testing model with sample texts...")
    predict_texts(model, tokenizer, test_texts, device)
    
    # Interactive mode
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("\nEnter your own texts to test (or 'quit' to exit):\n")
    
    while True:
        try:
            text = input("Enter text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("\n✓ Goodbye!")
                break
            
            if not text:
                print("⚠️  Please enter some text!")
                continue
            
            predict_texts(model, tokenizer, [text], device)
            print()
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            break
        except EOFError:
            print("\n\n✓ Goodbye!")
            break


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
