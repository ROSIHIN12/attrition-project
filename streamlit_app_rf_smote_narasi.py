import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load('model_attrition_rf_smote.pkl')
scaler = joblib.load('scaler_attrition_rf_smote.pkl')

st.title("🌳 HR Attrition Prediction (Random Forest + SMOTE)")
st.markdown("Masukkan data karyawan untuk memprediksi apakah ia akan keluar dari perusahaan atau tidak.")
st.caption("Catatan: Gaji diisi dalam satuan Rupiah (misal: 5.000.000)")

# Input fitur
age = st.slider("Usia (Age)", 18, 60, 30)

monthly_income = st.number_input(
    "Pendapatan Bulanan (Monthly Income dalam Rupiah)",
    min_value=1_000_000,
    max_value=50_000_000,
    value=5_000_000,
    step=500_000
)

years_at_company = st.slider("Lama Bekerja (Years at Company)", 0, 40, 5)

job_level = st.selectbox(
    "Tingkat Jabatan (Job Level)",
    options=[1, 2, 3, 4, 5],
    format_func=lambda x: {
        1: "1 - Staff",
        2: "2 - Supervisor",
        3: "3 - Assistant Manager",
        4: "4 - Manager",
        5: "5 - Senior Manager"
    }[x]
)

education = st.selectbox(
    "Tingkat Pendidikan (Education Level)",
    options=[1, 2, 3, 4, 5],
    format_func=lambda x: {
        1: "1 - SMA",
        2: "2 - Diploma",
        3: "3 - Sarjana",
        4: "4 - Magister",
        5: "5 - Doktor"
    }[x]
)

distance_from_home = st.slider("Jarak ke Tempat Kerja (km)", 1, 50, 10)

# Normalisasi gaji
normalized_income = monthly_income / 1000
input_data = np.array([[age, normalized_income, years_at_company, job_level, education, distance_from_home]])

# Prediksi
try:
    input_scaled = scaler.transform(input_data)

    if st.button("Prediksi"):
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0][1]

        if prediction[0] == 1:
            st.error(f"Karyawan berpotensi keluar 😟 (Probabilitas: {probability:.2%})")
        else:
            st.success(f"Karyawan kemungkinan besar tetap bekerja 🙂 (Risiko keluar: {probability:.2%})")

        # Tambahkan narasi penjelas
        st.markdown("---")
        st.subheader("📝 Keterangan Prediksi:")

        if prediction[0] == 1:
            st.markdown(
                "- Karyawan ini berpotensi keluar karena kemungkinan kombinasi dari beberapa faktor seperti:
"
                "  - Usia masih muda
"
                "  - Lama bekerja masih singkat
"
                "  - Gaji relatif rendah
"
                "  - Jarak rumah yang jauh

"
                "Faktor-faktor ini dapat memicu keinginan untuk mencari pekerjaan yang lebih sesuai atau lebih dekat."
            )
        else:
            st.markdown(
                "- Karyawan ini kemungkinan tetap bertahan karena didukung oleh:
"
                "  - Masa kerja yang cukup lama
"
                "  - Usia yang matang
"
                "  - Kemungkinan telah beradaptasi dengan lingkungan kerja

"
                "Hal-hal tersebut menjadi indikasi loyalitas dan stabilitas kerja."
            )

except ValueError as e:
    st.error(f"Terjadi kesalahan input: {e}")
