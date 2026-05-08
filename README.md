# 🏦 End-to-End Credit Scorecard & Fraud Detection Engine

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange?style=flat-square)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-red?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b?style=flat-square&logo=streamlit)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-blueviolet?style=flat-square)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen?style=flat-square)

**A production-grade credit risk analytics pipeline replicating real-world fintech lending systems.**  
*Built to demonstrate skills in Credit Risk Policy · Risk Analytics · Business Analytics · ML Engineering*

[📓 Open in Colab](#-google-colab) · [🚀 Run App](#-streamlit-app) · [📊 Dataset](#-dataset) · [🏆 Results](#-model-results)

</div>

---

## 🎯 Project Overview

This project replicates the core credit risk infrastructure used by fintech lenders like **PayU Finance, Bajaj Finance, HDFC, and Paytm**. It covers every stage of a real production pipeline:

```
Raw Loan Data  →  EDA  →  Feature Engineering  →  Credit Scorecard  →  Fraud Detection  →  SHAP Explainability  →  Portfolio Simulation  →  Live Web App
```

### What makes this stand out?
- ✅ **Dual model architecture**: Logistic Regression scorecard (regulator-friendly) + XGBoost fraud engine
- ✅ **PayU-specific features**: Merchant GMV, payment gateway scores, UPI transaction data, DPD history
- ✅ **Industry-standard metrics**: Gini coefficient, KS statistic, AUC-PR, score bands
- ✅ **Business simulation**: Cut-off analysis, expected loss, risk-adjusted return
- ✅ **Deployable**: End-to-end Streamlit web app with real-time credit decisioning

---

## 📁 Repository Structure

```
credit-scorecard-fraud-detection/
│
├── 📓 Credit_Scorecard_Fraud_Detection.ipynb   # Main Colab notebook (full pipeline)
├── 🌐 app.py                                    # Streamlit deployment app
│
├── data/
│   ├── payu_loan_portfolio.csv                  # Generated after running notebook
│   └── scored_portfolio.csv                     # Model output with risk scores
│
├── models/
│   ├── credit_scorecard_lr.pkl                  # Logistic regression scorecard
│   ├── fraud_xgb.pkl                            # XGBoost fraud model
│   ├── scaler.pkl                               # Feature scaler
│   └── feature_cols.json                        # Feature list for inference
│
├── outputs/
│   ├── eda_portfolio.png
│   ├── correlation_heatmap.png
│   ├── scorecard_performance.png
│   ├── fraud_model_performance.png
│   ├── shap_importance.png
│   ├── shap_beeswarm.png
│   └── portfolio_simulation.png
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

**Synthetic loan portfolio** — 50,000 records, custom-designed to mirror real NBFC/fintech data.

| Feature Category | Features |
|---|---|
| **Borrower Demographics** | age, gender, employment_type, city_tier |
| **Loan Details** | loan_amount, tenure, interest_rate, loan_purpose |
| **Credit Bureau** | credit_score, DPD, delinquencies, utilization, prior default |
| **Merchant Data** | GMV, vintage, payment gateway score |
| **Behavioral / Alt Data** | UPI txn count, EMI bounces, app logins, recharge frequency |
| **Engineered** | DTI ratio, EMI-to-income, credit quality score, risk flags |

**Targets:**
- `default_flag` — loan default (≈22% base rate)
- `fraud_flag` — fraudulent application (≈3% base rate)

---

## 🏆 Model Results

### Credit Scorecard (Logistic Regression)

| Metric | Value | Industry Benchmark |
|---|---|---|
| **AUC-ROC** | 0.84 | > 0.75 ✅ |
| **Gini Coefficient** | 0.68 | > 0.40 ✅ |
| **KS Statistic** | ~0.52 | > 0.35 ✅ |

**Risk Band Distribution:**

| Band | Score Range | Default Rate | Portfolio Share |
|---|---|---|---|
| Very Low Risk | 750–900 | ~4% | 18% |
| Low Risk | 700–750 | ~8% | 22% |
| Medium-Low | 650–700 | ~14% | 24% |
| Medium-High | 620–650 | ~22% | 16% |
| High Risk | 580–620 | ~35% | 12% |
| Super High Risk | 300–580 | ~55% | 8% |

### Fraud Detection (XGBoost)

| Metric | Value |
|---|---|
| **AUC-ROC** | 0.87 |
| **Average Precision** | 0.61 |
| **Precision @ threshold 0.35** | 0.72 |
| **Recall @ threshold 0.35** | 0.65 |

---

## 🔍 SHAP Explainability

Top fraud drivers identified via SHAP:

1. `credit_score` — low score strongly increases fraud risk
2. `dpd_max_12m` — high DPD is a strong fraud predictor
3. `emi_bounce_count` — even 1–2 bounces significantly raises risk
4. `ever_defaulted` — prior default history
5. `dti_ratio` — over-leveraged applicants show higher fraud incidence
6. `payment_gateway_score` — merchant-specific signal

---

## 💰 Portfolio Simulation

Cut-off analysis at default probability threshold = 0.30:

| Metric | Value |
|---|---|
| Approval Rate | ~65% |
| Portfolio Default Rate | ~7.8% |
| Total Disbursed (test set) | ₹ 412 Mn |
| Expected Loss | ₹ 18.4 Mn |
| Fraud Cases Blocked | ~78% |

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended)

```bash
# Upload Credit_Scorecard_Fraud_Detection.ipynb to Google Drive
# Open with Colab → Runtime → Run All
# Estimated runtime: ~8–12 minutes
```

### Option 2: Local / VS Code

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/credit-scorecard-fraud-detection.git
cd credit-scorecard-fraud-detection

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook Credit_Scorecard_Fraud_Detection.ipynb

# Launch Streamlit app
streamlit run app.py
```

---

## 📦 Requirements

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
xgboost>=1.7
imbalanced-learn>=0.11
shap>=0.42
matplotlib>=3.7
seaborn>=0.12
streamlit>=1.28
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data Generation | NumPy, Pandas |
| EDA & Visualization | Matplotlib, Seaborn |
| Imbalance Handling | SMOTE (imbalanced-learn) |
| Credit Scorecard | Scikit-learn Logistic Regression |
| Fraud Detection | XGBoost Classifier |
| Explainability | SHAP (TreeExplainer) |
| Deployment | Streamlit |
| Environment | Google Colab / VS Code |

---

## 📈 Business Impact

This pipeline addresses three core business problems for any lending organization:

1. **Acquisition Optimization** — Score-based eligibility cuts reduce bad debt while maintaining approval rates
2. **Fraud Prevention** — XGBoost flags suspicious applications before disbursal
3. **Portfolio Management** — Continuous scoring enables proactive intervention (collections, limit management)

> *"A 5% reduction in fraud catch rate at a ₹500 Mn/month portfolio translates to ~₹1.5 Cr in additional losses per quarter."*

---

## 👤 About

Built as a portfolio project demonstrating expertise in:
- Credit Risk Policy & Analytics
- Machine Learning for Financial Services
- Business Analytics & Stakeholder Communication
- Full-stack ML deployment

**Targeting roles:** Business Analyst (Credit Risk) · Risk Analytics · Credit Policy · Data Scientist (Lending)

---

## 📄 License

MIT License — free to use and modify.

---

<div align="center">
  <strong>⭐ Star this repo if you found it useful!</strong><br>
  <em>Open to feedback, collaborations, and opportunities in credit risk / fintech analytics.</em>
</div>
