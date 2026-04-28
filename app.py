import streamlit as st
import pandas as pd
import joblib

# --- 1. CONFIG & ASSETS ---
st.set_page_config(page_title="SME Underwriting Portal", layout="wide")

@st.cache_resource
def load_model_assets():
    model = joblib.load('credit_model.joblib')
    encoder = joblib.load('woe_encoder.joblib')
    return model, encoder

try:
    credit_model, woe_encoder = load_model_assets()
except:
    st.error("Missing model files.")

# --- 2. THE "SATISFYING" UI/UX CSS ---
st.markdown("""
    <style>
    /* 1. Force the background to a soft, satisfying Mint-Blue */
    .stApp {
        background-color: #f0f5f5 !important;
    }

    /* 2. Fix the top header (the black bar) to match the background */
    header[data-testid="stHeader"] {
        background: #f0f5f5 !important;
        border-bottom: 1px solid #dbe6e6;
    }

    /* 3. Sidebar: Deep Navy for a grounded, professional feel */
    [data-testid="stSidebar"] {
        background-color: #0d1b2a !important;
        border-right: 1px solid #1b263b;
    }
    
    /* Sidebar Text & Labels */
    [data-testid="stSidebar"] .stMarkdown p, 
    [data-testid="stSidebar"] label {
        color: #e0e1dd !important;
        font-weight: 500 !important;
    }

    /* 4. Headings: High contrast Navy for legibility */
    h1 {
        color: #1b263b !important;
        font-family: 'Inter', sans-serif;
        font-weight: 800 !important;
        padding-top: 0px !important;
    }
    h5 {
        color: #415a77 !important;
        font-weight: 400 !important;
        margin-bottom: 2rem !important;
    }

    /* 5. The Result Card: White for focus with a subtle green shadow */
    .result-box {
        background-color: #ffffff;
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(46, 204, 113, 0.1);
        border-left: 10px solid #2ecc71;
        margin-top: 10px;
    }

    /* 6. Buttons: Forest Green with smooth interaction */
    .stButton>button {
        background-color: #2ecc71 !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 12px 20px !important;
        font-weight: 700 !important;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #27ae60 !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(39, 174, 96, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. THE UI LAYOUT ---
# My title area - now clearly visible with high-contrast colors
st.title("SME Underwriting Engine")
st.markdown("##### Commercial Risk Assessment | DACH & UK Markets")

# Defining the split: Input on the left (sidebar), Analysis on the right
with st.sidebar:
    st.markdown("### Application Data")
    revenue = st.number_input("Annual Revenue (€)", value=250000, step=10000)
    loan_amt = st.number_input("Loan Principal (€)", value=50000, step=5000)
    assets = st.number_input("Total Asset Value (€)", value=150000, step=10000)
    biz_age = st.slider("Business Longevity (Years)", 0, 50, 5)
    size = st.number_input("Organization Size Factor", 1, 10, 2)
    
    st.markdown("---")
    analyze_btn = st.button("Execute Credit Analysis", use_container_width=True)

# Main result column
col_spacer, col_main = st.columns([0.1, 0.9])

with col_main:
    if analyze_btn:
        # --- MY LOGIC LAYER ---
        input_data = pd.DataFrame({
            'Annual_Revenue': [revenue], 'Requested_Loan': [loan_amt],
            'Asset_Value': [assets], 'Company_Size_Factor': [size],
            'LTV_Ratio': [loan_amt / assets if assets > 0 else 0],
            'Business_Age_Years': [biz_age]
        })
        
        # Transform through my WOE pipeline
        input_woe = woe_encoder.transform(input_data)
        prob = credit_model.predict_proba(input_woe)[:, 1][0]
        score = int(300 + (1 - prob) * 550)

        # Categorize decision
        if score > 820:
            status, color, rec = "LOW RISK", "#2ecc71", "Auto-Approve"
        elif score > 800:
            status, color, rec = "MEDIUM RISK", "#f1c40f", "Manual Referral"
        else:
            status, color, rec = "HIGH RISK", "#e74c3c", "Decline"

        # --- THE RESULT CARD DISPLAY ---
        st.markdown(f"""
            <div class="result-box">
                <p style="color: #778da9; font-size: 0.85rem; letter-spacing: 1px; font-weight: 700;">AUDIT-READY SCORE</p>
                <h1 style="font-size: 5rem; margin: 0; color: #1b263b; border:none;">{score}</h1>
                <p style="color: {color}; font-weight: 800; font-size: 1.2rem; margin-top: 10px;">{status}</p>
                <hr style="margin: 30px 0; opacity: 0.1;">
                <p style="font-size: 1.2rem; color: #1b263b;"><b>Primary Recommendation:</b> {rec}</p>
                <p style="color: #778da9; font-size: 0.9rem; margin-top: 20px;">
                    Decision based on Asset-to-Loan coverage and business longevity. <br>
                    Internal Probability of Default: <b>{prob:.2%}</b>
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Placeholder that matches the theme
        st.info("👈 Enter business parameters in the sidebar to generate a new credit assessment report.")