"""
Comprehensive Edge Case Handler
Fixes: Negation, Slang, Celebrity Names, Context Issues
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import sys
import os
import re

# Add models directory to Python path
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


# ============================================================================
# EDGE CASE DETECTION RULES
# ============================================================================

def detect_double_negative(text):
    """
    Detect 'not + negative word' patterns
    Examples: "not a bad guy", "not ugly", "not terrible"
    """
    negative_words = [
        'bad', 'terrible', 'awful', 'horrible', 'stupid', 'dumb', 
        'ugly', 'hate', 'mean', 'cruel', 'evil', 'nasty', 'trash', 
        'loser', 'worthless', 'useless', 'pathetic', 'disgusting'
    ]
    
    text_lower = text.lower()
    words = text_lower.split()
    
    for i in range(len(words)-1):
        if words[i] in ['not', "isn't", "isnt", "aren't", "arent"]:
            # Check next 1-3 words
            window = words[i+1:min(i+4, len(words))]
            
            for neg_word in negative_words:
                if neg_word in window:
                    # Double negative detected
                    return True, "double_negative"
    
    return False, None


def detect_positive_slang(text):
    """
    Detect positive slang that might be misclassified
    Examples: "GOAT", "beast", "sick", "fire", "killed it"
    """
    text_lower = text.lower()
    
    # Modern positive slang
    positive_slang = {
        'goat': 'greatest of all time',
        'beast': 'very skilled',
        'sick': 'awesome',
        'fire': 'excellent',
        'lit': 'amazing',
        'savage': 'impressive',
        'legend': 'highly respected',
        'king': 'best',
        'queen': 'best'
    }
    
    # Check for positive slang
    for slang in positive_slang.keys():
        if slang in text_lower:
            # Check if it's actually positive context
            positive_indicators = [
                'is', 'are', 'was', 'were', 'the', 'a', 'absolute',
                'total', 'complete', 'such', 'real'
            ]
            
            if any(ind in text_lower for ind in positive_indicators):
                return True, f"positive_slang:{slang}"
    
    # Check for "killed it" pattern (means "did great")
    if 'killed it' in text_lower or 'killing it' in text_lower:
        return True, "positive_expression:killed_it"
    
    return False, None


def detect_celebrity_only(text):
    """
    Detect if text is ONLY a celebrity name with no context
    """
    celebrity_names = [
        'virat', 'kohli', 'dhoni', 'rohit', 'sachin', 'bumrah',
        'messi', 'ronaldo', 'neymar', 'lebron', 'curry',
        'trump', 'biden', 'musk', 'bezos', 'gates'
    ]
    
    text_lower = text.lower().strip()
    words = text_lower.split()
    
    # Check if text is ONLY 1-2 words and matches celebrity
    if len(words) <= 2:
        for name in celebrity_names:
            if name in text_lower:
                return True, f"celebrity_only:{name}"
    
    return False, None


def detect_insufficient_context(text):
    """
    Detect if text is too short to make confident prediction
    """
    words = text.split()
    
    # Too short
    if len(words) <= 2:
        return True, "too_short"
    
    # Only contains neutral words
    neutral_only = all(word.lower() in ['is', 'are', 'the', 'a', 'an', 'it', 'he', 'she'] 
                      for word in words)
    if neutral_only:
        return True, "neutral_only"
    
    return False, None


def detect_positive_context(text):
    """
    Detect overall positive context that might have negative-sounding words
    """
    positive_indicators = [
        'love', 'great', 'amazing', 'awesome', 'excellent', 'wonderful',
        'fantastic', 'brilliant', 'good', 'nice', 'beautiful', 'perfect',
        'thank', 'thanks', 'appreciate', 'grateful', 'happy', 'proud',
        'congratulations', 'congrats', 'well done', 'good job'
    ]
    
    text_lower = text.lower()
    
    # Count positive indicators
    positive_count = sum(1 for word in positive_indicators if word in text_lower)
    
    if positive_count >= 2:
        return True, "positive_context"
    
    return False, None


# ============================================================================
# COMPREHENSIVE PREDICTION WITH ALL FIXES
# ============================================================================

def load_model_safe(model_path, device):
    """Load model safely"""
    print("Loading model...")
    model = BERTClassifier()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("✓ Model loaded\n")
    return model


def predict_text_basic(model, tokenizer, text, device):
    """Basic prediction without any fixes"""
    encoding = tokenizer(
        text, max_length=128, padding='max_length',
        truncation=True, return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()
        prob_not_cb = probs[0][0].item()
        prob_cb = probs[0][1].item()
    
    return pred, conf, prob_not_cb, prob_cb


def predict_with_all_fixes(model, tokenizer, text, device, verbose=True):
    """
    Predict with ALL edge case fixes applied
    
    Returns:
        prediction: 0 (Not CB) or 1 (CB) or 2 (Insufficient Context)
        confidence: float
        prob_not_cb: float
        prob_cb: float
        adjustments: list of applied fixes
    """
    
    # Get base prediction
    pred, conf, prob_not_cb, prob_cb = predict_text_basic(model, tokenizer, text, device)
    
    adjustments = []
    original_pred = pred
    original_conf = conf
    
    # ========================================================================
    # RULE 1: Insufficient Context
    # ========================================================================
    is_insufficient, reason = detect_insufficient_context(text)
    if is_insufficient:
        adjustments.append(f"insufficient_context:{reason}")
        if verbose:
            return 2, 0.0, 0.5, 0.5, adjustments  # Special code: 2 = Insufficient
    
    # ========================================================================
    # RULE 2: Celebrity Name Only
    # ========================================================================
    is_celebrity_only, reason = detect_celebrity_only(text)
    if is_celebrity_only:
        adjustments.append(reason)
        if verbose:
            return 2, 0.0, 0.5, 0.5, adjustments  # Insufficient context
    
    # ========================================================================
    # RULE 3: Strong Positive Context
    # ========================================================================
    is_positive_ctx, reason = detect_positive_context(text)
    if is_positive_ctx and pred == 1:
        adjustments.append(reason)
        pred = 0  # Force Not CB
        conf = 1 - conf
    
    # ========================================================================
    # RULE 4: Positive Slang
    # ========================================================================
    is_slang, reason = detect_positive_slang(text)
    if is_slang and pred == 1 and conf < 0.80:
        adjustments.append(reason)
        pred = 0  # Flip to Not CB
        conf = max(0.55, 1 - conf)  # At least 55% confidence
    
    # ========================================================================
    # RULE 5: Double Negative
    # ========================================================================
    is_double_neg, reason = detect_double_negative(text)
    if is_double_neg and pred == 1 and conf < 0.75:
        adjustments.append(reason)
        pred = 0  # Flip to Not CB
        conf = 1 - conf
    
    return pred, conf, prob_not_cb, prob_cb, adjustments


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    print("\n" + "="*80)
    print(" "*20 + "COMPREHENSIVE EDGE CASE HANDLER")
    print(" "*25 + "All Fixes Applied")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    model = load_model_safe('models/saved_models/bert_cyberbullying_model.pth', device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Comprehensive test cases
    test_cases = [
        # Category 1: Double Negatives
        ("he is not a bad guy", "Should be: Not CB (positive)"),
        ("she is not ugly", "Should be: Not CB (compliment)"),
        ("not bad at all", "Should be: Not CB (praise)"),
        ("you are not stupid", "Should be: Not CB (reassurance)"),
        
        # Category 2: Positive Slang
        ("Virat is GOAT", "Should be: Not CB (sports praise)"),
        ("You're a beast at coding", "Should be: Not CB (compliment)"),
        ("That presentation was sick", "Should be: Not CB (positive slang)"),
        ("You killed it today", "Should be: Not CB (did great)"),
        ("He's an absolute legend", "Should be: Not CB (high praise)"),
        
        # Category 3: Celebrity Names Only
        ("virat", "Should be: Insufficient Context"),
        ("kohli", "Should be: Insufficient Context"),
        ("messi", "Should be: Insufficient Context"),
        
        # Category 4: Actual Cyberbullying (should stay CB)
        ("you are stupid and ugly", "Should be: CB (clear insult)"),
        ("I hate you so much", "Should be: CB (hateful)"),
        ("you are a worthless loser", "Should be: CB (severe insult)"),
        
        # Category 5: Clear Not Cyberbullying (should stay Not CB)
        ("thank you for your help", "Should be: Not CB (gratitude)"),
        ("you did an amazing job", "Should be: Not CB (praise)"),
        ("have a great day", "Should be: Not CB (friendly)"),
        
        # Category 6: Ambiguous/Negative but not CB
        ("it is not good", "Should be: CB (criticism)"),
        ("you are not smart", "Should be: CB (insult)"),
    ]
    
    print("="*80)
    print("TESTING ALL EDGE CASES")
    print("="*80)
    
    stats = {"fixed": 0, "unchanged": 0, "insufficient": 0}
    
    for i, (text, expected) in enumerate(test_cases, 1):
        pred, conf, prob_not_cb, prob_cb, adjustments = predict_with_all_fixes(
            model, tokenizer, text, device, verbose=True
        )
        
        if pred == 2:
            label = "⚠️  INSUFFICIENT CONTEXT"
            stats["insufficient"] += 1
        elif pred == 1:
            label = "🚨 CYBERBULLYING"
        else:
            label = "✅ NOT CYBERBULLYING"
        
        adjustment_str = ""
        if adjustments:
            adjustment_str = f" [FIXED: {', '.join(adjustments)}]"
            stats["fixed"] += 1
        else:
            stats["unchanged"] += 1
        
        print(f"\n[{i}] '{text}'")
        print(f"    → {label} ({conf*100:.1f}% confident){adjustment_str}")
        print(f"    {expected}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total tests: {len(test_cases)}")
    print(f"Fixed by rules: {stats['fixed']}")
    print(f"Unchanged: {stats['unchanged']}")
    print(f"Insufficient context: {stats['insufficient']}")
    print(f"\nEdge case handling: {stats['fixed']/len(test_cases)*100:.1f}% of test cases adjusted")
    
    # Interactive mode
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("(Type 'quit' to exit)\n")
    
    while True:
        try:
            text = input("Enter text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q', '']:
                if text.lower() in ['quit', 'exit', 'q']:
                    print("\n✓ Goodbye!")
                break
            
            pred, conf, prob_not_cb, prob_cb, adjustments = predict_with_all_fixes(
                model, tokenizer, text, device, verbose=True
            )
            
            if pred == 2:
                label = "⚠️  INSUFFICIENT CONTEXT"
            elif pred == 1:
                label = "🚨 CYBERBULLYING"
            else:
                label = "✅ NOT CYBERBULLYING"
            
            adjustment_str = ""
            if adjustments:
                adjustment_str = f"\n  Fixes applied: {', '.join(adjustments)}"
            
            print(f"→ {label} ({conf*100:.1f}% confident){adjustment_str}\n")
            
        except KeyboardInterrupt:
            print("\n\n✓ Goodbye!")
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
