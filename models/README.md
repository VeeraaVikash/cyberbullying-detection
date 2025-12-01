# 🧠 MODELS FOLDER

Complete model training and evaluation for cyberbullying detection.

---

## 📁 Structure

```
models/
├── bert_classifier.py       # BERT model architecture
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
├── config.py                # Training configuration
├── README.md                # This file
└── saved_models/            # Trained models saved here
    └── bert_cyberbullying_model.pth (created after training)
```

---

## 🚀 Quick Start

### 1. Train Model

```bash
py models/train.py
```

This will:
- Load training and validation data
- Train BERT model for 3 epochs
- Save best model to `saved_models/`

**Time**: 2-4 hours (GPU) or 10-12 hours (CPU)

---

### 2. Evaluate Model

```bash
py models/evaluate.py
```

This will:
- Load trained model
- Test on test set
- Show accuracy, F1-score, confusion matrix

**Time**: ~5 minutes

---

## 📊 Model Architecture

### BERT Classifier

```
Input Text: "You are stupid"
    ↓
BERT Tokenizer
    ↓
Token IDs: [101, 2017, 2024, 7978, 102]
    ↓
BERT Encoder (bert-base-uncased)
    ├── 12 Transformer layers
    ├── 768 hidden dimensions
    └── 12 attention heads
    ↓
[CLS] Token Output (768-dim vector)
    ↓
Dropout (0.3)
    ↓
Linear Layer (768 → 2)
    ↓
Output Logits: [0.2, 0.8]
    ↓
Prediction: 1 (Cyberbullying)
```

**Model Size**: ~110 million parameters

---

## ⚙️ Configuration

Edit `config.py` to change settings:

```python
# Model
MODEL_NAME = 'bert-base-uncased'
NUM_CLASSES = 2
MAX_LENGTH = 128

# Training
BATCH_SIZE = 16       # Reduce if out of memory
NUM_EPOCHS = 3        # Increase for better accuracy
LEARNING_RATE = 2e-5
DROPOUT = 0.3

# Device
DEVICE = 'cuda'       # Use 'cpu' if no GPU
```

---

## 📈 Expected Performance

| Metric | Expected Value |
|--------|---------------|
| Accuracy | 88-92% |
| Precision | 85-90% |
| Recall | 87-91% |
| F1-Score | 86-90% |

**Training Time**:
- GPU (NVIDIA): 2-4 hours
- CPU: 10-12 hours

---

## 🔍 Files Explained

### 1. `config.py`
Training configuration and hyperparameters.

```python
from models.config import config
print(config)
```

---

### 2. `bert_classifier.py`
BERT model architecture.

```python
from models.bert_classifier import BERTClassifier, get_tokenizer

# Create model
model = BERTClassifier(num_classes=2)

# Get tokenizer
tokenizer = get_tokenizer()
```

---

### 3. `train.py`
Complete training script.

```bash
py models/train.py
```

**What it does**:
1. Load train.csv and val.csv
2. Create BERT model
3. Train for N epochs
4. Validate after each epoch
5. Save best model
6. Early stopping if no improvement

**Output**:
```
Epoch 1/3
Training: 100%|████████| 2083/2083 [15:23<00:00]
Validation: 100%|████████| 447/447 [01:45<00:00]
  Train Loss: 0.3245 | Train Acc: 0.8756
  Val Loss:   0.2891 | Val Acc:   0.8924
  ✓ Best model saved! (Val Acc: 0.8924)

Epoch 2/3
...
```

---

### 4. `evaluate.py`
Test model on test set.

```bash
py models/evaluate.py
```

**Output**:
```
📊 EVALUATION RESULTS
======================================================================

✓ Accuracy: 0.9124 (91.24%)
✓ Precision: 0.8956
✓ Recall: 0.9087
✓ F1-Score: 0.9021

Per-Class Metrics:
                    precision    recall  f1-score   support
Not Cyberbullying       0.92      0.89      0.90      1190
Cyberbullying           0.90      0.91      0.90      5950

Confusion Matrix:
              Predicted
              Not CB  |  CB
Actual  Not CB  1059  |  131
        CB       538  | 5412
```

---

## 💾 Saved Model Format

After training, model is saved as:

```
saved_models/bert_cyberbullying_model.pth
```

**Contains**:
- Model weights (state_dict)
- Optimizer state
- Best validation accuracy
- Configuration

**Size**: ~420 MB

---

## 🔄 Training Workflow

```
1. Load data
   └── data/processed/train.csv (33,320 samples)
   └── data/processed/val.csv (7,140 samples)

2. Create model
   └── BERT-base-uncased + Classification head

3. Training loop (3 epochs)
   For each epoch:
   ├── Train on train set
   ├── Validate on val set
   ├── Save if best model
   └── Early stop if no improvement

4. Save best model
   └── saved_models/bert_cyberbullying_model.pth
```

---

## 🎯 Usage Examples

### Train with Custom Config

```python
from models.config import config

# Modify settings
config.BATCH_SIZE = 8  # For low memory
config.NUM_EPOCHS = 5  # Train longer
config.DEVICE = 'cpu'  # Use CPU

# Train
from models.train import train
train()
```

---

### Load Trained Model

```python
import torch
from models.bert_classifier import BERTClassifier

# Load model
model = BERTClassifier()
checkpoint = torch.load('models/saved_models/bert_cyberbullying_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
from models.bert_classifier import get_tokenizer, tokenize_text

tokenizer = get_tokenizer()
text = "You are stupid"
encoding = tokenize_text(text, tokenizer)

with torch.no_grad():
    logits = model(encoding['input_ids'], encoding['attention_mask'])
    prediction = torch.argmax(logits, dim=1)

print(f"Prediction: {prediction.item()}")  # 0 or 1
```

---

## ⚠️ Troubleshooting

### Out of Memory Error
```python
# Reduce batch size in config.py
BATCH_SIZE = 8  # or 4
```

### CUDA Not Available
```python
# Use CPU in config.py
DEVICE = 'cpu'
```

### Slow Training
- Use GPU if available
- Reduce NUM_EPOCHS for quick test
- Reduce MAX_LENGTH to 64

---

## 📝 Training Tips

1. **Start Small**: Test with 1 epoch first
2. **Monitor**: Watch train vs val accuracy
3. **Early Stopping**: Stops if no improvement
4. **Save Best**: Only best model is kept
5. **GPU**: Use GPU for 5-10x speedup

---

## 🎓 What You'll Learn

- ✅ How to train BERT
- ✅ How PyTorch training works
- ✅ How to evaluate models
- ✅ How to interpret metrics

---

## 🚀 Next Steps

After training:
1. ✅ Evaluate on test set
2. ✅ Analyze errors
3. ✅ Deploy model
4. ✅ Create API

---

**Ready to train? Run:**
```bash
py models/train.py
```

---

**Last Updated**: December 1, 2024
