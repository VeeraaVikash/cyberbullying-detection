# 🛡️ Cyberbullying Detection AI

BERT-based deep learning system for detecting cyberbullying, sarcasm, and toxic content with **94.5% recall**.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Features

- ✅ **94.5% Recall** on direct cyberbullying detection
- ✅ **72% Confidence** on edge cases (sarcasm, negation patterns, indirect insults)
- ✅ **Real-time Detection** with <500ms response time
- ✅ **Beautiful Web Interface** with interactive UI
- ✅ **REST API** for easy integration
- ✅ **Batch Processing** support
- ✅ **Edge Case Detection** - handles sarcasm, indirect insults, negation patterns

## 📸 Demo

![Demo Screenshot](screenshots/demo.png)

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/veeravikash/cyberbullying-detection.git
cd cyberbullying-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Model

Download the pre-trained model:
- **[Google Drive Link](YOUR_GOOGLE_DRIVE_LINK)** (Recommended)
- **[Hugging Face](YOUR_HUGGINGFACE_LINK)** (Alternative)

Place the model file in: `models/saved_models/bert_cyberbullying_model.pth`

### Run Web Application

```bash
python app.py
```

Open browser: **http://localhost:5000**

### Quick Test

```bash
# Test single text
python predict_bert.py --text "I'm not saying you're dumb, but you need help"

# Interactive mode
python predict_bert.py

# Test edge cases
python test_custom_cases.py
```

## 📊 Performance Metrics

### Overall Performance

| Metric | Score |
|--------|-------|
| **Recall** | 94.5% |
| **Precision** | 93.9% |
| **F1 Score** | 94.2% |
| **Accuracy** | 94.3% |

### Edge Case Performance

| Edge Case Type | Recall | Confidence |
|----------------|--------|------------|
| **Direct Insults** | 94.5% | 97%+ |
| **Sarcasm** | 68% | 67% |
| **Negation Patterns** | 75% | 72% |
| **Indirect Insults** | 65% | 63% |

## 🎯 Example Results

```python
Input: "You're so stupid"
Output: 🚨 Cyberbullying (97.5% confidence)

Input: "I'm not saying you're dumb, but you need help"
Output: 🚨 Cyberbullying (72.1% confidence)

Input: "Wow you're SO smart 🙄"
Output: 🚨 Cyberbullying (67.3% confidence)

Input: "I disagree with your opinion"
Output: ✅ Not Cyberbullying (91.2% confidence)
```

## 🤖 Model Architecture

- **Base Model:** BERT-base-uncased (110M parameters)
- **Framework:** PyTorch + Transformers
- **Training Data:** 47,560 samples
- **Fine-tuning Data:** 120,000+ edge case examples
- **Optimizer:** AdamW
- **Learning Rate:** 2e-5 (base), 1e-5 (fine-tuning)
- **Epochs:** 3 (base), 2 (fine-tuning)

## 🌐 Web Application

### Features

- **Real-time Analysis:** Instant predictions on user input
- **Pre-loaded Examples:** Click-to-test example cases
- **Adjustable Sensitivity:** Threshold slider (0.3 - 0.7)
- **Statistics Dashboard:** Track detection metrics
- **Prediction History:** View and export past predictions
- **CSV Export:** Download results for analysis

### Screenshots

| Homepage | Results |
|----------|---------|
| ![Homepage](screenshots/ui.png) | ![Results](screenshots/results.png) |

## 📡 API Usage

### Python

```python
import requests

# Single prediction
response = requests.post('http://localhost:5000/predict', 
    json={'text': 'Your text here', 'threshold': 0.5})

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.1%}")
```

### cURL

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text", "threshold": 0.5}'
```

### JavaScript

```javascript
fetch('http://localhost:5000/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({text: 'Your text', threshold: 0.5})
})
.then(res => res.json())
.then(data => console.log(data));
```

See [API Documentation](API.md) for complete reference.

## 🔧 Training Your Own Model

### Step 1: Prepare Data

```bash
python prepare_edge_case_training.py
```

### Step 2: Train Base Model

```bash
python train_bert.py
```

Training time: ~60 minutes on RTX 4060

### Step 3: Fine-tune on Edge Cases

```bash
python fine_tune_edge_cases.py
```

Fine-tuning time: ~40 minutes

### Step 4: Evaluate

```bash
python test_custom_cases.py --compare
```

## 📂 Project Structure

```
cyberbullying-detection/
├── app.py                          # Flask web application
├── models/
│   ├── bert_classifier.py          # BERT model architecture
│   └── saved_models/               # Trained models (download separately)
├── templates/
│   └── index.html                  # Web UI
├── train_bert.py                   # Training script
├── fine_tune_edge_cases.py         # Fine-tuning script
├── predict_bert.py                 # Prediction script
├── test_custom_cases.py            # Testing suite
├── quick_test.py                   # Quick testing
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── SETUP.md                        # Setup guide
└── API.md                          # API documentation
```

## 📊 Datasets Used

Training combined multiple datasets:

1. **Cyberbullying Tweets** (47K samples)
   - Balanced dataset with multiple categories

2. **Sarcasm Tweets** (30K samples)
   - Labeled sarcasm, irony, and regular tweets

3. **News Headlines Sarcasm** (28K samples)
   - High-quality sarcastic vs. non-sarcastic headlines

4. **Hate Speech Dataset** (25K samples)
   - Distinguishes offensive language from hate speech

**Total:** 120,000+ samples across diverse contexts

## 🔬 Research

This project is part of research on edge case detection in cyberbullying at **SRM Institute of Science and Technology**.

### Research Supervisor
- **Dr. G. Balamurugan**
- Cybersecurity and Disruptive Technology Domain

### Citation

If you use this work in your research, please cite:

```bibtex
@article{veeraa2025cyberbullying,
  title={BERT-based Cyberbullying Detection with Edge Case Support},
  author={Veeraa Vikash, S.},
  institution={SRM Institute of Science and Technology},
  year={2025}
}
```

## 🛠️ Technologies

- **Deep Learning:** PyTorch, Transformers (Hugging Face)
- **NLP Model:** BERT-base-uncased
- **Web Framework:** Flask
- **Frontend:** HTML5, CSS3, JavaScript
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

## 📈 Future Improvements

- [ ] Multi-language support
- [ ] Real-time streaming detection
- [ ] Context-aware detection
- [ ] Ensemble models
- [ ] Mobile application
- [ ] Browser extension
- [ ] Integration with social media platforms

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 👤 Author

**Veeraa Vikash S.**

- 🎓 Computer Science Undergraduate, SRM Institute of Science and Technology
- 💼 Software Testing Intern, Interain AI
- 🔬 Research: Cyberbullying Detection and Prevention
- GitHub: [@veeravikash](https://github.com/veeravikash)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

## 🙏 Acknowledgments

- BERT model by Google Research
- Hugging Face Transformers library
- Dataset providers and contributors
- Research supervisor: Dr. G. Balamurugan
- SRM Institute of Science and Technology

## 📚 Additional Resources

- [Setup Guide](SETUP.md) - Detailed installation instructions
- [API Documentation](API.md) - Complete API reference
- [Training Guide](TRAINING.md) - How to train your own model
- [Contributing Guidelines](CONTRIBUTING.md) - How to contribute

## 📞 Support

If you have any questions or issues:

- 📧 Email: your.email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/veeravikash/cyberbullying-detection/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/veeravikash/cyberbullying-detection/discussions)

## ⭐ Show Your Support

Give a ⭐️ if this project helped you!

---

**Made with ❤️ by Veeraa Vikash**

**For research purposes and social good** 🛡️
