# 🩺 Diabetes Prediction System
**Government College University Faisalabad — BS Computer Science**
*Faizan Younis | Talha Arshad | Hamza Amin*

---

## Project Structure
```
diabetes_prediction/
├── app.py                    # Main Streamlit application
├── requirements.txt
├── models/                   # Saved trained models (auto-created)
├── outputs/                  # Exported reports (auto-created)
└── utils/
    ├── preprocessing.py      # Data loading, cleaning, scaling
    ├── ml_models.py          # LR, DT, RF, SVM training & evaluation
    ├── visualizations.py     # All matplotlib/seaborn charts
    └── report.py             # PDF report generator
```

---

## Setup & Run

### 1. Install Python (3.10+)
Download from https://www.python.org/downloads/

### 2. Create virtual environment (recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```
Opens automatically at http://localhost:8501

---

## How to Use

### Step 1 — Data Upload & EDA
- Click **"Use Pima Indians Sample Dataset"** or upload your own CSV
- Explore class distribution, correlations, feature distributions

### Step 2 — Train & Evaluate Models
- Adjust test size and CV folds
- Click **"Train All Models"** — trains LR, Decision Tree, Random Forest, SVM
- Compare accuracy, precision, recall, F1-score, ROC-AUC
- View ROC curves, confusion matrices, feature importance

### Step 3 — Predict Diabetes Risk
- Enter patient health parameters
- Get prediction (Diabetic / Non-Diabetic) + probability %

### Step 4 — Export Report
- Download full PDF report with all metrics and charts
- Download metrics as CSV

---

## Dataset
Uses the **Pima Indians Diabetes Dataset** from:
- Kaggle: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- UCI ML Repository: https://archive.ics.uci.edu/dataset/34/diabetes

**Required CSV columns:**
`Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`

---

## Models Implemented
| Model | Type |
|-------|------|
| Logistic Regression | Baseline binary classification |
| Decision Tree | Rule-based interpretable |
| Random Forest | Ensemble, reduced overfitting |
| SVM (RBF kernel) | High-dimensional feature handling |

---

## ⚠️ Disclaimer
This system is for **educational and research purposes only**.
It does not constitute professional medical advice.
# Diabetes-Prediction-System-FYP
