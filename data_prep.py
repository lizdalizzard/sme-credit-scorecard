import pandas as pd
import numpy as np

def load_and_clean_data(file_path):
    # 1. Load raw data
    df = pd.read_csv(file_path)
    
    # 2. Select columns & Rename them for Business Clarity
    mapping = {
        'TARGET': 'Default_Label',
        'AMT_INCOME_TOTAL': 'Annual_Revenue',
        'AMT_CREDIT': 'Requested_Loan',
        'AMT_GOODS_PRICE': 'Asset_Value',
        'DAYS_EMPLOYED': 'Days_In_Business',
        'DAYS_BIRTH': 'Applicant_Age_Days',
        'CNT_FAM_MEMBERS': 'Company_Size_Factor'
    }
    
    # Filter and rename
    df = df[list(mapping.keys())].rename(columns=mapping)

    # 3. Handle Missing Values (Global Median Imputation)
    # This ensures no 'NaN' errors during model training
    df = df.fillna(df.median())

    # 4. SME Feature Engineering
    # Loan-to-Value (LTV) 
    df['LTV_Ratio'] = df['Requested_Loan'] / df['Asset_Value']
    
    # Convert negative days to positive years for 'Business Age'
    df['Business_Age_Years'] = abs(df['Days_In_Business']) / 365
    
    # Cleanup: Remove the messy 'Days' columns
    df = df.drop(columns=['Days_In_Business', 'Applicant_Age_Days'])

    return df

if __name__ == "__main__":
    print("Data Prep Script Ready.")