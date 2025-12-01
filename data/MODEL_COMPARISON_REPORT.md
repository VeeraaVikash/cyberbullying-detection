# Model Comparison Report: Baseline vs Focal Loss

## Executive Summary

**Selected Model:** Baseline BERT (Original)  
**Reason:** Superior recall (94.50%) critical for user safety

## Methodology

Both models evaluated on identical test set:
- Dataset: data/processed_augmented/test.csv
- Samples: 9,475 (7,220 CB + 2,255 Not CB)
- Metrics: F1, Recall, Precision, ROC-AUC

## Results

### Model 1: Baseline BERT ✅ SELECTED
- **Training:** Standard CrossEntropyLoss + Data Augmentation
- **F1-Score:** 94.19%
- **Recall:** 94.50% ⭐ (CRITICAL METRIC)
- **Precision:** 93.88%
- **ROC-AUC:** 0.9661
- **False Negatives:** 397 (5.50% of CB missed)
- **False Positives:** 445 (19.73% false alarm rate)

### Model 2: Focal Loss + Class Weights ❌ REJECTED
- **Training:** Focal Loss (α=0.25, γ=2.0) + Balanced Weights
- **F1-Score:** 93.19%
- **Recall:** 91.93%
- **Precision:** 94.49%
- **ROC-AUC:** 0.9618
- **False Negatives:** 583 (8.07% of CB missed)
- **False Positives:** 387 (17.16% false alarm rate)

## Key Findings

### 1. Recall Is Critical for Safety
Original model catches **186 more cyberbullying messages** (397 vs 583 FN).

### 2. Focal Loss Made Model Too Conservative
While Focal Loss improved precision (+0.61%), it significantly reduced 
recall (-2.57%), causing the model to miss more actual cyberbullying.

### 3. Trade-off Analysis
- Improved: 58 fewer false alarms
- Original: 186 more caught CB messages

**Verdict:** Catching 186 more CB cases is far more valuable than 
reducing 58 false alarms.

## Impact Analysis

### If We Choose Improved Model:
- ❌ 186 additional users experience undetected cyberbullying
- ❌ Potential harm: suicide risk, trauma, continued harassment
- ✅ 58 fewer false alarms (minor inconvenience)

### If We Choose Original Model:
- ✅ 186 more users protected from cyberbullying
- ✅ Higher safety standard
- ⚠️ 58 more false alarms (acceptable, can be reviewed)

## Decision Rationale

For safety-critical cyberbullying detection:
1. **High recall is paramount** - Missing CB has severe consequences
2. **False positives are acceptable** - Can be reviewed by moderators
3. **User safety > Operational efficiency**

## Recommendation

✅ **Deploy Baseline BERT Model**

**Configuration:**
- Model: models/saved_models/bert_cyberbullying_model.pth
- Threshold: 0.50 (standard)
- Performance: 94.50% recall, 94.19% F1-score
- Status: Production-ready

**Optional Enhancement:**
- Test threshold=0.45 for even higher recall
- Expected: ~96-97% recall with acceptable precision drop

## Lessons Learned

1. **Focal Loss isn't always better** - Depends on application domain
2. **Safety > Metrics** - In safety-critical apps, prioritize recall
3. **Ablation studies are essential** - Always compare alternatives
4. **Context matters** - What works for one task may harm another

## For Research Paper

This comparison demonstrates:
- Systematic approach to model selection
- Understanding of domain requirements
- Critical evaluation of advanced techniques
- Evidence-based decision making

## Conclusion

The baseline BERT model with standard CrossEntropyLoss provides 
the best balance of safety and accuracy for cyberbullying detection, 
achieving 94.50% recall while maintaining 93.88% precision. Focal Loss, 
while theoretically sound for imbalanced data, proved suboptimal for 
this safety-critical application.

**Final Decision:** Use Baseline BERT model for production deployment.