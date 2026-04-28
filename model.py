import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np
from category_encoders import WOEEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib

# Import logic from other file
from data_prep import load_and_clean_data

#visual style 
sns.set_theme(style="whitegrid")
main_color = "#2c3e50" # Professional Navy Blue

# 1. Loading and Processing Data
df = load_and_clean_data('application_train.csv')

# 2. Setting up Features and Target
X = df.drop('Default_Label', axis=1)
y = df['Default_Label']

# 3. Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. WOE Encoding & Training
# Weight of Evidence (WOE) is the industry standard for Basel III compliance
encoder = WOEEncoder()
X_train_woe = encoder.fit_transform(X_train, y_train)
X_test_woe = encoder.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_woe, y_train)

# 5. Business Scoring Logic (300-850 Scale)
probs = model.predict_proba(X_test_woe)[:, 1]
gini = (2 * roc_auc_score(y_test, probs)) - 1

def to_credit_score(p):
    # High probability of default = Low credit score
    return 300 + (1 - p) * 550

# ==========================================
# PHASE 3: VISUALIZATIONS 
# ==========================================
print("--- Generating Professional Visuals ---")

# --- VISUAL 1: PRIMARY RISK DRIVERS (Bar Chart) ---
explainer = shap.Explainer(model, X_train_woe)
shap_v = explainer(X_test_woe.iloc[:500]) # Sample for speed

importance_df = pd.DataFrame({
    'Risk Factor': X_test.columns,
    'Impact Level': np.abs(shap_v.values).mean(0)
}).sort_values(by='Impact Level', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Impact Level', y='Risk Factor', color=main_color)
plt.title("What Drives the Credit Decision? (Market Risk Factors)", fontsize=14, fontweight='bold')
plt.xlabel("Strength of Influence on Final Score", fontsize=12)
plt.ylabel("")
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.savefig("assets/risk_drivers_analysis.png")

# --- VISUAL 2: RISK DISTRIBUTION (Dynamic Donut Chart) ---
test_scores = [to_credit_score(p) for p in probs]
risk_labels = []

for s in test_scores:
    if s > 820: risk_labels.append("Low Risk (Green)") 
    elif s > 800: risk_labels.append("Medium Risk (Amber)") 
    else: risk_labels.append("High Risk (Red)")

risk_df = pd.Series(risk_labels).value_counts().reset_index(name='Count')
risk_df.columns = ['Category', 'Count']

# Dynamic color and explode mapping
color_map = {
    "Low Risk (Green)": "#2ecc71",
    "Medium Risk (Amber)": "#f1c40f",
    "High Risk (Red)": "#e74c3c"
}
current_colors = [color_map[cat] for cat in risk_df['Category']]
current_explode = [0.05] * len(risk_df) 

plt.figure(figsize=(8, 8))
plt.pie(risk_df['Count'], labels=risk_df['Category'], 
        colors=current_colors, 
        autopct='%1.1f%%', startangle=140, pctdistance=0.85, 
        explode=current_explode)

centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title("Portfolio Risk Composition", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("assets/portfolio_risk_distribution.png")

# --- VISUAL 3: RECESSION STRESS TEST (Comparison) ---
X_stressed = X_test_woe.copy()
X_stressed['Annual_Revenue'] = X_stressed['Annual_Revenue'] * 0.80 # 20% Revenue Drop
stressed_probs = model.predict_proba(X_stressed)[:, 1]
stressed_gini = (2 * roc_auc_score(y_test, stressed_probs)) - 1

stress_df = pd.DataFrame({
    'Scenario': ['Normal Market', '20% Revenue Drop'],
    'Model Accuracy (Gini)': [gini, stressed_gini]
})

plt.figure(figsize=(8, 6))
ax = sns.barplot(data=stress_df, x='Scenario', y='Model Accuracy (Gini)', palette="Blues_d")
plt.ylim(0, 0.4)
plt.title("Model Resilience Audit", fontsize=14, fontweight='bold')
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("assets/recession_stress_test.png")

print("\n" + "="*40)
print("PROJECT 01: AUDIT COMPLETE")
print("="*40)
print(f"Base Predictive Power (Gini): {gini:.2f}")
print(f"Stress-Tested Power (Gini):  {stressed_gini:.2f}")
print("Check the 'assets' folder for 3 PNG files ")
print("="*40)

# Save the trained model and encoder
joblib.dump(model, 'models/credit_model.joblib')
joblib.dump(encoder, 'models/woe_encoder.joblib')

print("Model saved as credit_model.joblib") 