import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load('model_attrition_final.pkl')
scaler = joblib.load('scaler_attrition_final.pkl')

st.title("🎯 HR Attrition Prediction")
st.markdown("Masukkan data karyawan untuk memprediksi apakah ia akan keluar dari perusahaan atau tidak.")

# Input fitur
age = st.slider("Usia (Age)", 18, 60, 30)
monthly_income = st.number_input("Pendapatan Bulanan (Monthly Income)", 1000, 20000, 5000)
years_at_company = st.slider("Lama Bekerja (Years at Company)", 0, 40, 5)
job_level = st.selectbox("Tingkat Jabatan (Job Level)", [1, 2, 3, 4, 5])
education = st.selectbox("Tingkat Pendidikan (Education Level)", [1, 2, 3, 4, 5])
distance_from_home = st.slider("Jarak ke Tempat Kerja (km)", 1, 50, 10)

# Proses input
input_data = np.array([[age, monthly_income, years_at_company, job_level, education, distance_from_home]])
input_scaled = scaler.transform(input_data)

# Prediksi
if st.button("Prediksi"):
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction[0] == 1:
        st.error(f"Karyawan berpotensi keluar 😟 (Probabilitas: {probability:.2%})")
    else:
        st.success(f"Karyawan kemungkinan besar tetap bekerja 🙂 (Risiko keluar: {probability:.2%})")
