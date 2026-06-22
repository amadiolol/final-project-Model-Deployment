import streamlit as st
import pandas as pd
import pickle
import numpy as np
import re

st.set_page_config(page_title="Credit Score Predictor", layout="wide")
st.title("Credit Score Prediction Deployment")
st.write("Aplikasi inferencing untuk memprediksi performa kredit nasabah langsung dari Dataset C.")

@st.cache_resource
def load_components():
    with open('best_rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return model, encoders

model, encoders = load_components()

def preprocess_input(df, encoders):
    cols_to_drop = ['Unnamed: 0', 'ID', 'Customer_ID', 'Name', 'SSN', 'Month', 'Credit_Score']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    if 'Occupation' in df.columns:
        df['Occupation'] = df['Occupation'].replace('_______', np.nan)
        
    numeric_cols = ['Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment',
                    'Changed_Credit_Limit', 'Outstanding_Debt', 'Amount_invested_monthly']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[^0-9.-]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    def extract_months(text):
        if pd.isna(text): return np.nan
        years = re.search(r'(\d+)\s*Years', str(text))
        months = re.search(r'(\d+)\s*Months', str(text))
        y = int(years.group(1)) if years else 0
        m = int(months.group(1)) if months else 0
        return (y * 12) + m

    if 'Credit_History_Age' in df.columns:
        df['Credit_History_Age_Months'] = df['Credit_History_Age'].apply(extract_months)
        df = df.drop(columns=['Credit_History_Age'])
        
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=['object']).columns:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if col in encoders:
            df[col] = df[col].apply(
                lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0]
            )
            df[col] = encoders[col].transform(df[col])
            
    return df

st.subheader("Data Nasabah:")

try:
   
    df_test = pd.read_csv('data_C.csv')
    st.write("Preview 5 Data Teratas:")
    st.dataframe(df_test.head(5))
    
    if st.button("Predict Credit Score"):
        try:
            df_clean = preprocess_input(df_test.copy(), encoders)
            
            predictions = model.predict(df_clean)
            
            target_mapping = {0: 'Poor', 1: 'Standard', 2: 'Good'}
            df_test['Prediksi_Credit_Score'] = [target_mapping[p] for p in predictions]
            
            st.success("Prediksi Berhasil Dilakukan!")
            st.write("Hasil Klasifikasi Kredit Nasabah:")
            
            display_cols = ['Customer_ID', 'Name', 'Prediksi_Credit_Score']
            cols_to_show = [col for col in display_cols if col in df_test.columns]
            
            if not cols_to_show:
                cols_to_show = ['Prediksi_Credit_Score']
                
            st.dataframe(df_test[cols_to_show])
            
        except Exception as e:
            st.error(f"Terjadi kesalahan teknis saat prediksi: {e}")

except FileNotFoundError:
    st.error("File 'data_C.csv' tidak ditemukan. Pastikan file tersebut berada di folder yang sama dengan file 'app.py' ini.")