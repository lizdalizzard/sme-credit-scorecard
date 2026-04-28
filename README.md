# SME Credit Scorecard Engine (DACH/UK)

> **Automated Underwriting for the "Missing Middle"** — A logic-driven credit scoring system built to bridge the gap between traditional banking and modern SME lending.

---

### 🚀 [Click Here to View Live Dashboard](PASTE_YOUR_STREAMLIT_LINK_HERE)

---

### Project Structure
```text
sme-credit-scorecard/
├── app.py                # Live Streamlit Dashboard UI
├── model.py              # ML Engine: WOE Encoding & Logistic Regression
├── data_prep.py          # ETL: Data Cleaning & Feature Engineering
├── requirements.txt      # Dependency Mapping
├── .gitignore            # Security & Data Privacy Config
├── assets/               # Professional BI Visuals & Audits
│   ├── risk_drivers.png
│   └── portfolio_risk.png
└── models/               # Serialized Model Artifacts (.joblib)



💡 The Problem
Traditional lenders reject 60% of viable SMEs due to a lack of deep credit bureau history. This project implements Alternative Data logic to score businesses based on assets, revenue, and longevity rather than just bureau hits.

⚙️ How it Works (The Logic Pipeline)
Data Ingestion: Loads raw applicant data (300k+ records).
Feature Engineering: Computes LTV (Loan-to-Value) and Business Vintage proxies.
Risk Encoding: Implements Weight of Evidence (WOE) for Basel III compliance.
ML Inference: Uses a Logistic Regression classifier to calculate Probability of Default.
Score Calibration: Maps probabilities to a standard 300-850 Credit Score.

Regulatory & Legal Compliance
GDPR Article 22: Uses SHAP (Shapley Values) to provide clear "Reason Codes" for every automated decision.
Basel III Standards: Follows global banking requirements for model stability and transparency.
EU AI Act: Moves from "Black Box" AI to "Glass Box" interpretability.

Technical Stack
Languages: Python (Pandas, Scikit-Learn)
Explainability: SHAP
Encoding: Category Encoders (WOE)
Deployment: Streamlit Cloud


Created by Jeremiah.I as part of a Finance & AI Portfolio.

