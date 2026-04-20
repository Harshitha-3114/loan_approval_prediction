# 🏦 Loan Approval Prediction System

A machine learning web application that predicts whether a loan application will be **Approved** or **Rejected**, based on applicant financial and personal details. Built with a Random Forest Classifier and deployed as an interactive Streamlit app.

---

## 📌 Project Overview

Banks receive thousands of loan applications daily. This project automates the approval decision process using a supervised ML model trained on real-world-style applicant data, helping reduce manual effort and improve decision consistency.

---

## 🗂️ Dataset

- **File:** `loan_approval_prediction.csv`
- **Records:** 4,269 applicants
- **Features:** 12 input features + 1 target label

| Feature | Description |
|---|---|
| `no_of_dependents` | Number of financial dependents |
| `education` | Graduate / Not Graduate |
| `self_employed` | Yes / No |
| `income_annum` | Annual income (₹) |
| `loan_amount` | Requested loan amount (₹) |
| `loan_term` | Loan repayment duration (months) |
| `cibil_score` | Credit score (300–900) |
| `residential_assets_value` | Value of residential property |
| `commercial_assets_value` | Value of commercial property |
| `luxury_assets_value` | Value of luxury assets |
| `bank_asset_value` | Value of bank-held assets |
| `loan_status` | **Target** — Approved / Rejected |

**Class distribution:** 2,656 Approved | 1,613 Rejected

---

## 🔧 Project Pipeline

### 1. Data Preprocessing
- Stripped whitespace from column names and string values
- Imputed missing values using **median** for numerical columns
- Detected and capped outliers using **IQR (1.5× rule)**
- Label encoded categorical features (`education`, `self_employed`, `loan_status`)

### 2. Exploratory Data Analysis
- Missing value heatmaps (before and after cleaning)
- Boxplots for outlier detection and treatment
- Feature distribution plots with KDE curves

### 3. Model Training
- **Algorithm:** Random Forest Classifier
- **Hyperparameters:** `n_estimators=100`, `max_depth=5`, `random_state=42`, `oob_score=True`
- **Train/Test Split:** 80% / 20%

### 4. Model Evaluation
- Accuracy Score
- Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix
- ROC Curve with AUC Score
- Feature Importance Plot

### 5. Deployment
- Model serialized using `pickle` → `loan_model.pkl`
- Interactive UI built with **Streamlit** (`app.py`)

---

## 🚀 How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the Streamlit App
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
loan_approval_app/
│
├── app.py                        # Streamlit web app
├── loan_predict_project.ipynb    # ML model notebook
├── loan_approval_modified.csv    # Dataset
├── loan_model.pkl                # Trained model (serialized)
└── requirements.txt              # Dependencies
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas & NumPy | Data manipulation |
| Seaborn & Matplotlib | Data visualization |
| Scikit-learn | ML model (Random Forest) |
| Streamlit | Web app UI |
| Pickle | Model serialization |

---



---

