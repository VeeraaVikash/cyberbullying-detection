# 📊 Notebooks - Data Analysis & Visualization

This folder contains Jupyter notebooks for exploratory data analysis, model performance visualization, and error analysis.

---

## 📓 **Notebooks Overview**

### **1. Dataset Exploration** (`1_dataset_exploration.ipynb`)
**Purpose:** Comprehensive analysis of the cyberbullying dataset

**What it generates:**
- ✅ Class distribution charts
- ✅ Text length distributions
- ✅ Word clouds (cyberbullying vs not cyberbullying)
- ✅ Dataset comparison (original vs augmented)
- ✅ Statistical summaries

**Output files:**
- `class_distribution.png`
- `text_length_distribution.png`
- `wordclouds.png`
- `dataset_comparison.png`
- `dataset_summary.csv`

**Use in paper:** Section 3 (Data & Methodology)

---

### **2. Model Performance** (`2_model_performance.ipynb`)
**Purpose:** Analyze and visualize model results

**What it generates:**
- ✅ Performance comparison charts
- ✅ Improvement breakdown
- ✅ Confusion matrices
- ✅ Training curves
- ✅ Results comparison table

**Output files:**
- `model_comparison.png`
- `performance_improvement.png`
- `confusion_matrices.png`
- `training_curves.png`
- `model_results_comparison.csv`

**Use in paper:** Section 4 (Results)

---

### **3. Error Analysis** (`3_error_analysis.ipynb`)
**Purpose:** Analyze model mistakes and edge case handling

**What it generates:**
- ✅ Celebrity bias analysis
- ✅ Edge case performance charts
- ✅ Error type distribution
- ✅ Solution comparison
- ✅ Error summary table

**Output files:**
- `celebrity_bias_analysis.png`
- `edge_case_performance.png`
- `error_types_distribution.png`
- `solution_comparison.png`
- `error_analysis_summary.csv`

**Use in paper:** Section 5 (Discussion & Error Analysis)

---

## 🚀 **How to Run**

### **Step 1: Install Dependencies**
```bash
pip install jupyter notebook matplotlib seaborn wordcloud pandas numpy scikit-learn
```

### **Step 2: Navigate to Notebooks**
```bash
cd notebooks
```

### **Step 3: Start Jupyter**
```bash
jupyter notebook
```

This will open Jupyter in your browser.

### **Step 4: Run Notebooks**
1. Click on a notebook (e.g., `1_dataset_exploration.ipynb`)
2. Click "Cell" → "Run All"
3. Wait for all cells to execute
4. Images will be saved to the `notebooks/` folder

---

## 📊 **Generated Visualizations**

After running all notebooks, you'll have **14 visualizations** ready for your research paper!

### **For Section 3 (Data):**
- Class distribution
- Text length distribution  
- Word clouds
- Dataset comparison

### **For Section 4 (Results):**
- Model comparison
- Performance improvement
- Confusion matrices
- Training curves

### **For Section 5 (Error Analysis):**
- Celebrity bias analysis
- Edge case performance
- Error types distribution
- Solution comparison

---

## 📁 **Folder Structure**

```
notebooks/
├── 1_dataset_exploration.ipynb     ← Run first
├── 2_model_performance.ipynb       ← Run second
├── 3_error_analysis.ipynb          ← Run third
├── README.md                       ← You are here
└── [generated images and CSVs]     ← Output files
```

---

## 💡 **Tips**

### **For Best Results:**
1. Run notebooks in order (1 → 2 → 3)
2. Make sure your data files exist in `../data/processed/`
3. Check that model results match your actual results
4. Adjust numbers if needed (edit cells before running)

### **Customization:**
- Change colors by modifying `color=` parameters
- Adjust figure sizes with `figsize=(width, height)`
- Add more charts by copying existing cells
- Export to PDF: File → Download as → PDF

### **Troubleshooting:**
- **"File not found"**: Make sure you're in the notebooks folder
- **"Module not found"**: Run `pip install [module]`
- **Kernel crashed**: Restart kernel and try again

---

## 🎓 **For Your Research Paper**

### **How to Use These Visualizations:**

1. **In LaTeX:**
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{figures/model_comparison.png}
\caption{Model Performance Comparison}
\label{fig:model_comparison}
\end{figure}
```

2. **In Word:**
- Insert → Pictures → Select PNG file
- Add caption below image
- Reference in text: "Figure 1 shows..."

3. **In Presentation:**
- Drag and drop PNG files into slides
- High quality (300 DPI)

---

## ✅ **Checklist**

Before writing your paper, make sure you have:

- [ ] Ran all 3 notebooks
- [ ] Generated all visualizations
- [ ] Saved CSV files for tables
- [ ] Verified results match your actual results
- [ ] Exported high-quality images (300 DPI)
- [ ] Organized files for paper writing

---

## 📚 **Next Steps**

After running notebooks:
1. Review all generated visualizations
2. Select best charts for paper
3. Write figure captions
4. Reference figures in text
5. Include CSV data as tables

---

**All notebooks ready! Run them to generate publication-quality visualizations!** 🎉
