📊 DACH & UK SME Credit Scorecard [Credit Risk & ML Engine]
A logic-driven data pipeline and automated underwriting dashboard. It transforms raw financial data into real-time creditworthiness assessments for the European SME market using Basel III-compliant scoring.

### :rocket: [Click Here to View Live Dashboard](https://sme-credit-scorecard-be8ub8urocaxwgtprzc3az.streamlit.app/)

🎯 Project Goal
To bridge the gap between "Thin File" SME data and credit decisions. This prototype demonstrates a functional path to automating 35%+ of the initial underwriting workload by replacing manual data reviews with a Machine Learning scorecard. It provides lenders with a standardized "Credit Score" (300–850) based on asset coverage and business longevity.
The Big Picture
The project is built as a three-stage modular pipeline. When the system is initialized, it processes data in the following sequence:
ETL & Cleaning (Data Prep) ➔ Risk Engine (ML Training & Scoring) ➔ Streamlit UI (Professional Dashboard)
📂 Project Structure
data_prep.py: The ETL Engine. Handles global median imputation, feature engineering (LTV & Vintage), and data cleaning.
model.py: The Risk Engine. Implements WOE (Weight of Evidence) encoding, Logistic Regression training, and SHAP explainability.
/assets: The Audit Trail. Contains high-fidelity BI visualizations and model performance reports.
/models: The Core Artifacts. Contains serialized .joblib files (the model's "brain") used for real-time inference.
System Architecture
code
Mermaid
graph TD
    A[application_train.csv] -->|Step 1| B(data_prep.py)
    B -->|Cleans & Engineers Features| C(model.py)
    C -->|Generates BI Visuals| D[assets/folder]
    C -->|Saves Model Artifacts| E[models/folder]
    D & E -->|Step 3| F[Streamlit Dashboard UI]

    style F fill:#2ecc71,stroke:#27ae60,color:#fff
    style A fill:#f0f5f5,stroke:#1b263b
    style B fill:#f0f5f5,stroke:#1b263b
    style C fill:#f0f5f5,stroke:#1b263b
🧠 Core Logic: The "Asset-Backed" Scorecard
The primary innovation of this engine is the shift from "Black Box" AI to Explainable Credit Risk. It utilizes Weight of Evidence (WoE)—a banking industry standard—to ensure the model is stable, transparent, and legally defensible.
1. The Risk Calculation
The engine performs a 3-step audit of the applicant's profile:
Asset-to-Loan Coverage: Calculates the LTV (Loan-to-Value) Ratio to determine the physical security of the loan.
Business Vintage: Converts raw operational days into a "Longevity Score" to assess historical stability.
WOE Transformation: Groups raw numbers into "Risk Bins" to ensure the model remains robust against data outliers.
2. Scoring Rubric (Calibration)
Credit Score	Risk Level	Decision Logic
> 820	🟢 Low Risk	Auto-Approve: High serviceability.
800–820	🟡 Medium Risk	Manual Review: Standard leverage.
< 800	🔴 High Risk	Decline: Protects bank capital.
🇪🇺 DACH & UK Market Specialization
This engine is specifically tuned to the regulatory requirements of the European lending market:
Basel III Alignment: Uses Logistic Regression and WOE encoding, the standard for internal ratings-based (IRB) approaches in European banking.
GDPR Compliance (Article 22): Integrated SHAP values to provide automated "Reason Codes," explaining exactly why a score was generated.
Open Banking Proxy: Mapped alternative data features to mirror the logic used by PSD2-compliant fintechs (e.g., Klarna, Iwoca, N26) to score SMEs without deep bureau history.
Tech Stack
Model Logic: Python 3.11, Scikit-Learn, Category Encoders (WOE).
Data Science: Pandas (Wrangling), SHAP (Explainability).
Visualizations: Seaborn & Matplotlib (Market Risk Drivers).
Frontend UI: Streamlit (Fintech-branded Mint & Navy aesthetic).
🚀 How to Run
Initialize Environment:
code
Bash
pip install -r requirements.txt
Train & Audit Model:
code
Bash
python model.py
Start Underwriting Portal:
code
Bash
streamlit run app.py
>
> 
Author: Jeremiah.I
