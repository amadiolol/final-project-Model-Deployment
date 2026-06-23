import streamlit as st
import pandas as pd
import boto3
import json

st.set_page_config(page_title="Credit Score Predictor Cloud", layout="wide")
st.title("Credit Score Prediction Deployment (AWS SageMaker)")
st.write("Aplikasi inferencing ini terhubung langsung dengan AWS SageMaker Endpoint.")

# Nama endpoint HARUS sama dengan yang ada di script deployment
ENDPOINT_NAME = "credit-score-endpoint"
REGION = "us-east-1"

st.subheader("Data Nasabah:")

try:
    df_test = pd.read_csv('data_C.csv')
    st.write("Preview 5 Data Teratas:")
    st.dataframe(df_test.head(5))
    
    if st.button("Predict Credit Score"):
        with st.spinner("Memanggil AWS SageMaker Endpoint..."):
            try:
                # 1. Konversi data mentah menjadi JSON payload
                # Kita batasi 100 data pertama agar payload JSON tidak terlalu besar untuk real-time inference
                df_subset = df_test.head(100).copy() 
                payload = {
                    "instances": df_subset.to_dict(orient='records')
                }
                
                # 2. Panggil SageMaker Endpoint
                runtime = boto3.client('sagemaker-runtime', region_name=REGION)
                response = runtime.invoke_endpoint(
                    EndpointName=ENDPOINT_NAME,
                    ContentType='application/json',
                    Accept='application/json',
                    Body=json.dumps(payload)
                )
                
                # 3. Ekstrak Hasil
                result = json.loads(response['Body'].read().decode("utf-8"))
                df_subset['Prediksi_Credit_Score'] = result['labels']
                
                st.success("Prediksi Berhasil Dilakukan via SageMaker!")
                
                # 4. Tampilkan Hasil
                display_cols = ['Customer_ID', 'Name', 'Prediksi_Credit_Score']
                cols_to_show = [col for col in display_cols if col in df_subset.columns]
                
                if not cols_to_show:
                    cols_to_show = ['Prediksi_Credit_Score']
                    
                st.dataframe(df_subset[cols_to_show])
                
            except Exception as e:
                st.error(f"Terjadi kesalahan teknis saat memanggil AWS Endpoint: {e}")

except FileNotFoundError:
    st.error("File 'data_C.csv' tidak ditemukan. Pastikan file berada di direktori yang sama.")
