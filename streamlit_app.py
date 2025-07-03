import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load('model_attrition_fixed.pkl')
scaler = joblib.load('scaler_attrition_fixed.pkl')

st.title("Attrition Prediction - HR Analytics")

st.markdown("Masukkan data karyawan di bawah ini untuk memprediksi apakah akan keluar atau tidak:")

# Input fitur penting (bisa disesuaikan lagi)
age = st.slider("Age", 18, 60, 30)
monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)
years_at_company = st.slider("Years at Company", 0, 40, 5)
job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
education = st.selectbox("Education Level", [1, 2, 3, 4, 5])
distance_from_home = st.slider("Distance from Home (km)", 1, 50, 10)

# Simpan input ke dalam array
user_input = np.array([[age, monthly_income, years_at_company, job_level, education, distance_from_home]])

# Preprocessing input
user_input_scaled = scaler.transform(user_input)

# Prediksi
prediction = model.predict(user_input_scaled)
proba = model.predict_proba(user_input_scaled)[0][1]

# Output hasil
if st.button("Prediksi"):
    if prediction[0] == 1:
        st.error(f"Karyawan berpotensi keluar 😟 (Probabilitas: {proba:.2%})")
    else:
        st.success(f"Karyawan cenderung bertahan 🙂 (Probabilitas keluar: {proba:.2%})")
