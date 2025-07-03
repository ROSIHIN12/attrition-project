import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load('model_attrition_rf.pkl')
scaler = joblib.load('scaler_attrition_rf.pkl')

st.title("🧠 HR Attrition Prediction (Versi Random Forest)")
st.markdown("Masukkan data karyawan untuk memprediksi apakah ia akan keluar dari perusahaan atau tidak.")

# Input fitur
age = st.slider("Usia (Age)", 18, 60, 30)
monthly_income = st.number_input("Pendapatan Bulanan (Monthly Income)", 1000, 20000, 5000)
years_at_company = st.slider("Lama Bekerja (Years at Company)", 0, 40, 5)
job_level = st.selectbox("Tingkat Jabatan (Job Level)", [1, 2, 3, 4, 5])
education = st.selectbox("Tingkat Pendidikan (Education Level)", [1, 2, 3, 4, 5])
distance_from_home = st.slider("Jarak ke Tempat Kerja (km)", 1, 50, 10)

overtime = st.selectbox("Lembur (OverTime)", ["No", "Yes"])
job_satisfaction = st.slider("Kepuasan Kerja (Job Satisfaction)", 1, 4, 3)
env_satisfaction = st.slider("Kepuasan Lingkungan (Environment Satisfaction)", 1, 4, 3)
work_life_balance = st.slider("Keseimbangan Hidup-Kerja (Work Life Balance)", 1, 4, 3)
business_travel = st.selectbox("Frekuensi Dinas (Business Travel)", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])

# Encode input kategorikal
overtime_enc = 1 if overtime == "Yes" else 0
bt_map = {"Non-Travel": 0, "Travel_Rarely": 2, "Travel_Frequently": 1}
business_travel_enc = bt_map[business_travel]

# Gabungkan semua input
input_data = np.array([[age, monthly_income, years_at_company, job_level, education,
                        distance_from_home, overtime_enc, job_satisfaction,
                        env_satisfaction, work_life_balance, business_travel_enc]])

# Scaling
numerical_indices = [0,1,2,3,4,5,7,8,9]
input_data[:, numerical_indices] = scaler.transform(input_data[:, numerical_indices])

# Prediksi
if st.button("Prediksi"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.error(f"Karyawan berpotensi keluar 😟 (Probabilitas: {probability:.2%})")
    else:
        st.success(f"Karyawan kemungkinan besar tetap bekerja 🙂 (Risiko keluar: {probability:.2%})")
