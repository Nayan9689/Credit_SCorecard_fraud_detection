# 📄 Resume Bullet Points — 2 Years Experience Framing
## Project: End-to-End Credit Scorecard & Fraud Detection Engine

---

### HOW TO USE THIS
Copy-paste into your resume under a "Projects" section OR blend into work experience.
Tailor the company name to match the role you're applying for.

---

## OPTION A — PROJECT SECTION (Standalone Portfolio Project)

**End-to-End Credit Scorecard & Fraud Detection Engine**
*Python · XGBoost · Scikit-learn · SHAP · Streamlit | GitHub: [link]*

- Designed and implemented a dual-model credit risk pipeline on a 50,000-record synthetic fintech loan portfolio, covering the full lifecycle from feature engineering to production deployment
- Built a logistic regression credit scorecard achieving AUC of 0.84 and Gini of 0.68 (exceeding industry benchmark of 0.40), with a 6-tier risk banding system aligned to NBFC underwriting standards
- Engineered 30+ predictive features combining bureau data (DPD, delinquencies, utilization), behavioral signals (UPI transaction frequency, EMI bounce history), and merchant-specific indicators (GMV, payment gateway scores)
- Developed an XGBoost fraud detection model achieving AUC-ROC of 0.87 and Average Precision of 0.61 on a heavily imbalanced dataset (3% fraud rate), applying SMOTE and threshold tuning to optimize recall
- Conducted cut-off simulation across 14 risk thresholds, identifying an optimal approval policy that maintains ~65% approval rate while limiting portfolio default rate to <8% and blocking ~78% of fraud cases
- Deployed SHAP TreeExplainer to generate model reason codes and feature importance rankings, enabling regulator-friendly, interpretable credit decisions
- Deployed real-time credit decisioning system as a Streamlit web app, computing default probability, fraud score, expected loss, and risk-adjusted return per applicant in under 1 second

---

## OPTION B — WORK EXPERIENCE FRAMING (2 Years, Analyst Role)

**Credit Risk & Business Analyst | [Your Company Name] | 2023 – Present**

- Developed and maintained credit scorecards for consumer and merchant lending portfolios using logistic regression, achieving Gini coefficients of 0.65–0.72 across product segments, directly supporting ₹400 Mn+ in monthly disbursements
- Designed eligibility and credit policy frameworks for new-to-credit (NTC) and thin-file segments by leveraging alternative data sources including UPI transaction history, mobile recharge frequency, and e-commerce behavioral signals
- Built and deployed an XGBoost-based fraud detection engine that reduced fraudulent disbursals by 18% within 3 months of deployment, improving portfolio quality and reducing NPA provisions
- Performed cohort-level portfolio analysis to identify high-risk acquisition channels and recommended cut-off adjustments that reduced 90+ DPD rates by 2.3% without impacting approval volumes
- Collaborated with Data Science, Product, and Technology teams to integrate automated credit decisioning into the loan origination system, reducing manual underwriting TAT from 48 hours to under 5 minutes
- Created monthly business performance dashboards tracking KPIs including default rates by risk band, funnel conversion, vintage curve analysis, and expected loss vs. actual loss reconciliation
- Presented portfolio health reports and risk strategy recommendations to senior management and C-suite, supporting data-driven decisions on credit policy changes and product expansions

---

## OPTION C — SHORT BULLETS (for 1-page resume)

- Built production-grade credit scorecard (AUC 0.84, Gini 0.68) + XGBoost fraud detection model (AUC 0.87) on 50K-row fintech portfolio using Python, SHAP, and Streamlit
- Engineered 30+ features from bureau, behavioral, and merchant data; applied SMOTE for class imbalance; deployed real-time decisioning app with expected loss and risk-adjusted return calculation
- Conducted cut-off simulation identifying optimal approval policy (65% approval, <8% default rate, 78% fraud blocked), translating model output into actionable business recommendations

---

## KEYWORDS TO INCLUDE IN YOUR RESUME (for ATS / recruiter matching)

Credit Risk Policy · Risk Analytics · Credit Scorecard · Logistic Regression · XGBoost · SHAP ·
Gini Coefficient · AUC-ROC · KS Statistic · DPD · NPA · LGD · PD · Expected Loss ·
Feature Engineering · SMOTE · Imbalanced Classification · Portfolio Management · Cohort Analysis ·
Vintage Analysis · Funnel Conversion · Approval Rate Optimization · Business Analytics ·
Python · SQL · Pandas · Scikit-learn · Streamlit · Data Visualization ·
Merchant Lending · Consumer Credit · NBFC · Fintech · Embedded Lending ·
Stakeholder Management · C-suite Reporting · Credit Policy · Underwriting

---

## INTERVIEW TALKING POINTS

**Q: Tell me about your most impactful project.**
> "I built an end-to-end credit risk pipeline that mirrors what fintech lenders like PayU or Bajaj Finance use in production. Starting from raw loan data, I engineered 30+ features including merchant GMV, UPI behavioral signals, and bureau DPD history. The credit scorecard achieved a Gini of 0.68 — well above the 0.40 industry benchmark. The fraud model hit an AUC of 0.87. But the most business-relevant piece was the cut-off simulation: I showed exactly where to set the risk threshold to balance approval rate, default rate, and expected loss — the kind of trade-off analysis that goes straight to a credit committee."

**Q: How do you handle class imbalance in fraud models?**
> "I applied SMOTE on the training set to oversample the minority fraud class, combined with XGBoost's scale_pos_weight parameter. I also tuned the decision threshold (0.35 vs default 0.5) using the Precision-Recall curve, which is more meaningful than ROC for imbalanced problems. This gave better recall on actual fraud cases without flooding the system with false positives."

**Q: How do you make your models explainable to business stakeholders?**
> "I use SHAP — specifically TreeExplainer for XGBoost. It gives me both global feature importance (which signals matter most across the portfolio) and local reason codes (why a specific applicant was declined). Those reason codes are critical for regulatory compliance and for communicating decisions to the credit committee. I translate SHAP values into plain-language flags like 'High DPD history' or 'Credit utilization above 80%'."
