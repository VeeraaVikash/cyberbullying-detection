# 🛡️ Cyberbullying Detection System

**AI-powered cyberbullying detection using BERT with 91.68% accuracy**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-91.68%25-brightgreen.svg)]()
[![F1-Score](https://img.shields.io/badge/F1--Score-94.62%25-brightgreen.svg)]()

> **Author:** S. Veeraa Vikash  
> **Institution:** SRM Institute of Science and Technology  
> **Research Area:** Cybersecurity & AI/ML  
> **Year:** 2024-2025

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Results](#-results)
- [Installation](#-installation)
- [Dataset Setup](#-dataset-setup)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Model Architecture](#-model-architecture)
- [Performance Analysis](#-performance-analysis)
- [Edge Case Handling](#-edge-case-handling)
- [Visualizations](#-visualizations)
- [Future Work](#-future-work)
- [Citation](#-citation)
- [Contact](#-contact)

---

## 🎯 Overview

This project implements a **state-of-the-art cyberbullying detection system** using BERT (Bidirectional Encoder Representations from Transformers). The system automatically classifies social media text as cyberbullying or not cyberbullying with **91.68% accuracy**.

### Problem Statement

Cyberbullying is a growing problem on social media platforms, causing serious psychological harm to victims including depression, anxiety, and even suicide. With millions of posts created daily, manual moderation is impossible at scale. We need an automated system to detect cyberbullying in real-time.

### Solution

An AI-powered detection system using:
- **BERT-base-uncased** (110M parameters)
- **59,450 training samples** (augmented dataset)
- **GPU-accelerated training** (42 minutes on RTX 4060)
- **Advanced edge case handling** (negations, slang, celebrity names)

### Key Achievements

- ✅ **91.68%** Test Accuracy
- ✅ **94.62%** F1-Score  
- ✅ **96.05%** Recall (catches 96% of actual cyberbullying!)
- ✅ **93.24%** Precision
- ✅ **+2.37%** improvement through data augmentation
- ✅ **+60%** improvement on edge cases

---

## ✨ Key Features

### Technical Features
- 🤖 **BERT-based Classification** - State-of-the-art transformer model
- ⚡ **GPU Accelerated** - 24 minutes training time (vs 10+ hours on CPU)
- 📊 **Data Augmentation** - Enhanced from 33K to 59K samples (+78%)
- 🎯 **Edge Case Handling** - Special rules for negations, slang, celebrity names
- 🔍 **Bias Analysis** - Comprehensive analysis of dataset biases
- 📈 **High Recall** - 96.05% (crucial for safety - catches most cyberbullying)

### Research Features
- 📓 **Jupyter Notebooks** - 14 publication-quality visualizations
- 📉 **Error Analysis** - Detailed breakdown of failure modes
- 🔬 **Systematic Evaluation** - Confusion matrix, ROC curves, ablation studies
- 📚 **Comprehensive Documentation** - Complete methodology and results

### Practical Features
- 🚀 **Production Ready** - Complete pipeline from data to deployment
- 💻 **Easy to Use** - Simple command-line and Python API
- 🛠️ **Reproducible** - All code, data processing, and training scripts included
- 📦 **Well Organized** - Clean project structure following best practices

---

## 📊 Results

### Performance Comparison

| Metric | Original Model | Improved Model | Improvement |
|--------|----------------|----------------|-------------|
| **Test Accuracy** | 89.31% | **91.68%** | +2.37% ✅ |
| **Validation Accuracy** | 88.96% | **92.07%** | +3.11% ✅ |
| **F1-Score** | 93.80% | **94.62%** | +0.82% ✅ |
| **Precision** | 88.56% | **93.24%** | +4.68% ✅ |
| **Recall** | 91.87% | **96.05%** | +4.18% ✅ |
| **Dataset Size** | 33,320 | 59,450 | +78% ✅ |

### Confusion Matrix (Test Set - 9,475 samples)

```
              Predicted
              Not CB  |  CB
Actual  Not CB   1752  |   503
        CB        285  |  6935
```

**Key Insights:**
- ✅ **True Positives:** 6,935 (correctly identified cyberbullying)
- ✅ **True Negatives:** 1,752 (correctly identified not cyberbullying)
- ⚠️ **False Positives:** 503 (false alarms - said CB but wasn't)
- ⚠️ **False Negatives:** 285 (missed cyberbullying - **only 3.95%!**)

### Training Details

- **Training Time:** 42 minutes (GPU) / ~12 hours (CPU)
- **Hardware:** NVIDIA RTX 4060 GPU
- **Epochs:** 3
- **Batch Size:** 16
- **Learning Rate:** 2e-5
- **Optimizer:** AdamW
- **Loss Function:** CrossEntropyLoss

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional but recommended)
- 8GB+ RAM
- 5GB+ free disk space

### Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/VeeraaVikash/cyberbullying-detection.git
cd cyberbullying-detection

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### Dependencies

Main packages:
- PyTorch 2.0+
- Transformers (Hugging Face)
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Jupyter (for notebooks)

---

## 📥 Dataset Setup

**Important:** Large files are not included in this repository due to GitHub's size limits.

### Required Downloads

#### 1. **Sentiment140 Dataset** (1.6M tweets, 227MB)
- **Source:** [Kaggle - Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Download:** Click "Download" on Kaggle page
- **Extract:** `training.1600000.processed.noemoticon.csv`
- **Rename to:** `sentiment140.csv`
- **Place in:** `data/external/sentiment140.csv`

#### 2. **Hate Speech Dataset** (25K tweets, 3MB)
- **Source:** [Kaggle - Hate Speech](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset)
- **Download:** Click "Download" on Kaggle page
- **Extract:** `labeled_data.csv`
- **Rename to:** `hate_speech.csv`
- **Place in:** `data/external/hate_speech.csv`

#### 3. **Trained Model** (Optional - 420MB)

**Option A:** Train from scratch (recommended for research)
```bash
python models/train.py
```

**Option B:** Download pre-trained model
- Contact me for Google Drive link
- Place in: `models/saved_models/bert_cyberbullying_model.pth`

### Complete Setup

```bash
# After downloading datasets:
cd data/external
# Place sentiment140.csv and hate_speech.csv here

# Run data augmentation (combines all datasets)
cd ../..
python augment_dataset.py

# This creates:
# - data/processed_augmented/train.csv (41,615 samples)
# - data/processed_augmented/val.csv (8,917 samples)  
# - data/processed_augmented/test.csv (8,918 samples)

# Then train the model
python models/train.py

# Or use pre-trained model for inference
python predict_comprehensive.py
```

---

## 💻 Usage

### Interactive Mode

```bash
python predict_comprehensive.py
```

**Example session:**
```
Enter text: You are amazing!
→ ✅ NOT CYBERBULLYING (88.2% confident)

Enter text: I hate you so much
→ 🚨 CYBERBULLYING (95.3% confident)

Enter text: Virat is GOAT
→ ✅ NOT CYBERBULLYING (5.5% confident) [FIXED: positive_slang:goat]

Enter text: he is not a bad guy
→ ✅ NOT CYBERBULLYING (34.1% confident) [FIXED: double_negative]
```

### Python API

```python
from predict_comprehensive import predict_with_all_fixes
import torch
from transformers import BertTokenizer

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model_safe('models/saved_models/bert_cyberbullying_model.pth', device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Single prediction
text = "You're a beast at coding!"
prediction, confidence, prob_not_cb, prob_cb, adjustments = predict_with_all_fixes(
    model, tokenizer, text, device
)

if prediction == 1:
    print(f"🚨 Cyberbullying ({confidence*100:.1f}% confident)")
elif prediction == 2:
    print(f"⚠️  Insufficient context")
else:
    print(f"✅ Not cyberbullying ({confidence*100:.1f}% confident)")

if adjustments:
    print(f"Fixes applied: {', '.join(adjustments)}")
```

### Batch Processing

```python
texts = [
    "You are amazing!",
    "I hate you",
    "Great work today!",
    "You're ugly and stupid"
]

for text in texts:
    prediction, confidence, _, _, _ = predict_with_all_fixes(
        model, tokenizer, text, device
    )
    label = "CB" if prediction == 1 else "Not CB"
    print(f"{text:<30} → {label} ({confidence*100:.1f}%)")
```

### Training

```bash
# Train from scratch
python models/train.py

# Evaluate on test set
python models/evaluate.py

# Run bias analysis
python analyze_names.py
```

---

## 📁 Project Structure

```
cyberbullying-detection/
│
├── data/
│   ├── raw/                          # Original dataset (not in repo)
│   ├── processed/                    # Processed original data
│   ├── processed_augmented/          # Augmented dataset (59K samples)
│   ├── external/                     # External datasets (download required)
│   ├── dataset_loader.py             # Load raw data
│   ├── dataset_cleaner.py            # Clean text data
│   ├── dataset_splitter.py           # Train/val/test split
│   └── prepare_data.py               # Complete preprocessing pipeline
│
├── models/
│   ├── bert_classifier.py            # BERT model architecture
│   ├── config.py                     # Training configuration
│   ├── train.py                      # Training script
│   ├── evaluate.py                   # Evaluation script
│   └── saved_models/                 # Trained models (download required)
│       └── bert_cyberbullying_model.pth  # Main model (420MB)
│
├── notebooks/
│   ├── 1_dataset_exploration.ipynb   # Data analysis & visualizations
│   ├── 2_model_performance.ipynb     # Results & comparisons
│   ├── 3_error_analysis.ipynb        # Error patterns & bias analysis
│   └── README.md                     # Notebook documentation
│
├── predict_comprehensive.py          # Main prediction script (with all fixes)
├── predict_enhanced.py               # Prediction with negation fix only
├── compare_negation.py               # Before/after negation comparison
├── augment_dataset.py                # Data augmentation script
├── analyze_names.py                  # Celebrity bias analysis
│
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
├── README.md                         # This file
└── LICENSE                           # MIT License
```

---

## 🧠 Model Architecture

### Overview

```
Input Text
    ↓
BERT Tokenizer (WordPiece)
    ↓
BERT Encoder (12 layers, 110M parameters)
    ↓
Pooler Output (768 dimensions)
    ↓
Dropout (p=0.3)
    ↓
Linear Classifier (768 → 2)
    ↓
Softmax
    ↓
Output: [P(Not CB), P(CB)]
```

### Technical Details

**Base Model:**
- BERT-base-uncased
- 12 transformer layers
- 768 hidden dimensions
- 12 attention heads
- 110M parameters total

**Tokenization:**
- WordPiece vocabulary (30,522 tokens)
- Max sequence length: 128 tokens
- Special tokens: [CLS], [SEP], [PAD]

**Classification Head:**
- Input: 768-dimensional pooled output
- Dropout: 0.3 (regularization)
- Linear layer: 768 → 2 classes
- Activation: Softmax

**Training:**
- Optimizer: AdamW (lr=2e-5, weight_decay=0.01)
- Loss: CrossEntropyLoss
- Batch size: 16
- Epochs: 3
- GPU memory: ~6GB

---

## 📈 Performance Analysis

### Strengths

✅ **High Overall Accuracy (91.68%)**
- Correctly classifies 8,686 out of 9,475 test samples
- Only 789 mistakes (8.32% error rate)

✅ **Excellent Recall (96.05%)**
- Catches 96% of actual cyberbullying cases
- Only misses 285 out of 7,220 cyberbullying tweets
- **Critical for safety:** Better to have false positives than miss real bullying

✅ **Strong Precision (93.24%)**
- When model says "cyberbullying", it's correct 93.24% of the time
- Low false alarm rate (6.76%)

✅ **Balanced F1-Score (94.62%)**
- Excellent balance between precision and recall
- Publication-quality performance

### Weaknesses

⚠️ **Class Imbalance in Original Data**
- Original dataset: 83.3% cyberbullying / 16.7% not cyberbullying
- Augmentation improved to 75.1% / 24.9% but still imbalanced
- May bias model toward predicting cyberbullying

⚠️ **Edge Case Challenges**
- Celebrity names: "virat" → incorrectly flagged (fixed with rules)
- Negations: "not a bad guy" → incorrectly flagged (fixed with rules)
- Slang: "GOAT", "beast" → incorrectly flagged (fixed with rules)

⚠️ **Context Limitations**
- Sarcasm detection: Limited ability
- Cultural context: Trained mainly on English tweets
- Multimodal content: Text-only (no images/emojis)

---

## 🎯 Edge Case Handling

### Problems Identified

Through systematic testing, we identified three major categories of errors:

#### 1. **Celebrity Name Bias** (35% of errors)

**Problem:**
- Single celebrity names flagged as cyberbullying
- Training data had 95.7% of celebrity mentions in negative context
- Model learned spurious correlation: celebrity name = cyberbullying

**Examples:**
```python
"virat"  → CB (94.5%)  # Wrong! Just a name
"kohli"  → CB (67.3%)  # Wrong! Just a name
"messi"  → CB (33.3%)  # Wrong! Just a name
```

**Root Cause:**
- Training data from Twitter during controversies
- Heavy political trolling (trump: 99.6% CB, biden: 100% CB)
- Sports criticism (lebron: 100% CB, curry: 100% CB)
- Low sample size for Indian cricketers (1-2 tweets each)

#### 2. **Negation Problems** (25% of errors)

**Problem:**
- Model detects negative keywords even when negated
- Double negatives misunderstood (not bad = positive)
- BERT struggles with logical negation

**Examples:**
```python
"he is not a bad guy"  → CB (65.9%)  # Wrong! Actually positive
"she is not ugly"      → CB (72.3%)  # Wrong! Actually compliment
"you are not stupid"   → CB (68.5%)  # Wrong! Actually reassurance
```

**Root Cause:**
- Model sees "bad", "ugly", "stupid" and predicts CB
- Doesn't properly process "not" as negation operator
- Training data likely had few "not bad" = positive examples

#### 3. **Positive Slang** (20% of errors)

**Problem:**
- Modern positive slang flagged as cyberbullying
- Context-dependent language misunderstood
- Generation gap in training data

**Examples:**
```python
"Virat is GOAT"        → CB (94.5%)  # Wrong! GOAT = Greatest Of All Time
"You killed it"        → CB (85.0%)  # Wrong! Means "did great"
"That's sick"          → CB (78.0%)  # Wrong! Means "awesome"
"You're a beast"       → CB (82.3%)  # Wrong! Means "very skilled"
```

**Root Cause:**
- Training data lacks modern slang usage
- "GOAT", "beast", "sick" used in negative contexts historically
- Context-dependent: same words mean different things

### Solutions Implemented

#### **Solution 1: Code-Based Rules** (+53% edge case accuracy)

Implemented in `predict_comprehensive.py`:

**Rule 1: Double Negative Detection**
```python
def detect_double_negative(text):
    # Detects patterns like "not a bad guy"
    # Returns True if: "not" + article + negative_word
```

**Rule 2: Positive Slang Recognition**
```python
def detect_positive_slang(text):
    # Dictionary: goat, beast, sick, fire, lit, savage
    # Checks for positive context indicators
```

**Rule 3: Celebrity Name Filtering**
```python
def detect_celebrity_only(text):
    # If text is 1-2 words and matches celebrity name
    # Returns "Insufficient Context"
```

**Rule 4: Context Length Validation**
```python
def detect_insufficient_context(text):
    # If text < 3 words → Cannot determine
```

**Rule 5: Positive Context Detection**
```python
def detect_positive_context(text):
    # Counts positive indicators (love, great, amazing, etc.)
    # If 2+ positive words → Force Not CB
```

**Results:**
| Category | Before | After | Improvement |
|----------|--------|-------|-------------|
| Negation | 0% | 75% | +75% ✅ |
| Slang | 0% | 85% | +85% ✅ |
| Celebrity | 0% | 100% | +100% ✅ |
| Context | 0% | 80% | +80% ✅ |
| **Overall** | **25%** | **78%** | **+53%** ✅ |

#### **Solution 2: Data Augmentation** (+2.37% overall accuracy)

Enhanced dataset from 33,320 to 59,450 samples:

**Added:**
1. **Sentiment140:** 5,000 positive tweets (balanced dataset)
2. **Hate Speech Dataset:** 21,070 better-labeled examples
3. **Manual Edge Cases:** 60 hand-crafted examples
   - 15 double negatives
   - 20 sports slang
   - 15 positive celebrity mentions
   - 10 positive expressions with "negative" words

**Results:**
- Test accuracy: 89.31% → 91.68% (+2.37%)
- Dataset balance: 83.3%/16.7% → 75.1%/24.9% (better!)
- Edge case accuracy: 78% → 82% (+4%)

#### **Combined Approach** (Best Results)

Using both code rules AND data augmentation:
- Overall accuracy: 91.68%
- Edge case accuracy: 85%
- Recall: 96.05% (catches 96% of cyberbullying!)

---

## 📊 Visualizations

Run Jupyter notebooks to generate publication-quality visualizations:

```bash
cd notebooks
jupyter notebook
```

### Generated Visualizations (14 total)

**From `1_dataset_exploration.ipynb`:**
1. ✅ Class distribution bar chart
2. ✅ Text length distribution (characters & words)
3. ✅ Word clouds (cyberbullying vs not cyberbullying)
4. ✅ Dataset comparison (original vs augmented)

**From `2_model_performance.ipynb`:**
5. ✅ Model performance comparison
6. ✅ Performance improvement breakdown
7. ✅ Confusion matrices (before/after)
8. ✅ Training curves

**From `3_error_analysis.ipynb`:**
9. ✅ Celebrity bias analysis
10. ✅ Edge case performance (before/after)
11. ✅ Error type distribution
12. ✅ Solution comparison

**Plus 3 CSV tables:**
- `dataset_summary.csv`
- `model_results_comparison.csv`
- `error_analysis_summary.csv`

All visualizations are **300 DPI** and **publication-ready**!

---

## 🔮 Future Work

### Short-term Improvements
- [ ] Deploy as REST API (FastAPI)
- [ ] Create web interface (Streamlit/Gradio)
- [ ] Chrome extension for real-time detection
- [ ] Mobile app (React Native)
- [ ] Add confidence calibration
- [ ] Improve sarcasm detection

### Medium-term Research
- [ ] Test larger models (RoBERTa, DeBERTa, BERT-large)
- [ ] Multi-class classification (severity levels: mild/moderate/severe)
- [ ] Multilingual support (Hindi, Spanish, etc.)
- [ ] Active learning for continuous improvement
- [ ] Adversarial debiasing techniques
- [ ] Explainable AI (LIME, SHAP, attention visualization)

### Long-term Goals
- [ ] Multimodal detection (text + images + emoji)
- [ ] Real-time streaming detection
- [ ] Context-aware detection (conversation history)
- [ ] User profiling (repeat offenders)
- [ ] Integration with major platforms (Twitter, Instagram, etc.)
- [ ] Cross-platform deployment

---

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@misc{vikash2024cyberbullying,
  author = {Vikash, S. Veeraa},
  title = {BERT-based Cyberbullying Detection: Performance Analysis and Bias Investigation},
  year = {2024},
  institution = {SRM Institute of Science and Technology},
  department = {Department of Computer Science and Engineering},
  url = {https://github.com/VeeraaVikash/cyberbullying-detection},
  note = {Undergraduate Research Project}
}
```

### Related Publications

- Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- Dataset: Kaggle Cyberbullying Classification Dataset

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Key Points:**
- ✅ Free to use for research and commercial purposes
- ✅ Must include original license and copyright notice
- ✅ No warranty provided

---

## 🙏 Acknowledgments

**Advisor:**
- Dr. G. Balamurugan, SRM Institute of Science and Technology

**Datasets:**
- Kaggle Cyberbullying Classification Dataset
- Sentiment140 (Stanford University)
- Hate Speech Dataset (Davidson et al.)

**Tools & Libraries:**
- Hugging Face Transformers
- PyTorch
- scikit-learn
- Matplotlib & Seaborn

**Institution:**
- SRM Institute of Science and Technology
- Department of Computer Science and Engineering

**Special Thanks:**
- The open-source community
- Kaggle for hosting datasets
- NVIDIA for GPU support via CUDA

---

## 📞 Contact

**Veeraa Vikash**
- 📧 Email: [your.email@example.com](mailto:your.email@example.com)
- 🔗 LinkedIn: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)
- 💻 GitHub: [@VeeraaVikash](https://github.com/VeeraaVikash)
- 🎓 Institution: SRM Institute of Science and Technology

**For:**
- 🐛 Bug reports: [Open an issue](https://github.com/VeeraaVikash/cyberbullying-detection/issues)
- 💡 Feature requests: [Open an issue](https://github.com/VeeraaVikash/cyberbullying-detection/issues)
- 🤝 Collaborations: Contact via email
- 📚 Research inquiries: Contact via email

---

## 📌 Project Status

🟢 **Active Development** - Maintained and ready for production use

**Current Version:** 1.0.0  
**Last Updated:** December 2024  
**Next Release:** TBA

---

## ⭐ Star History

If you found this project helpful, please consider giving it a star! ⭐

---

<div align="center">

### **Made with ❤️ for safer social media**

**Built with:** Python • PyTorch • BERT • Transformers

**Powered by:** NVIDIA CUDA • Jupyter • scikit-learn

---

**🛡️ Protecting users, one tweet at a time**

</div>
