# HR Attrition Prediction App

Aplikasi ini dibuat untuk memprediksi apakah seorang karyawan akan keluar dari perusahaan (attrition) atau tidak, berdasarkan data HR seperti usia, gaji, masa kerja, tingkat pendidikan, dan faktor lainnya.

Model yang digunakan adalah **Random Forest Classifier** yang telah dilatih menggunakan teknik **SMOTE (Synthetic Minority Over-sampling Technique)** untuk mengatasi ketidakseimbangan data antara karyawan yang keluar dan tetap bekerja.

## 🔍 Dataset
Dataset yang digunakan adalah:  
[IBM HR Analytics Attrition Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

## Link Google Colab 
[Attrition_RF_SMOTE.ipynb](https://drive.google.com/file/d/1meKxbr2KfPTI7uJKQrwhG7a7ot4yEbrH/view?usp=sharing)

## 🚀 Fitur Aplikasi
- Input data karyawan melalui form interaktif di Streamlit
- Prediksi risiko attrition secara real-time menggunakan model Random Forest + SMOTE
- Menampilkan probabilitas prediksi
- Menyediakan narasi otomatis penjelas hasil prediksi (alasan kemungkinan keluar/bertahan)

## 🛠 Teknologi yang Digunakan
- Python
- Streamlit
- Scikit-learn
- Pandas & NumPy
- Joblib
- Imbalanced-learn (SMOTE)

## 📁 Struktur File
```
📁 attrition-project/
├── streamlit_app_rf_smote_narasi.py     # Aplikasi Streamlit versi final + narasi
├── model_attrition_rf_smote.pkl         # Model Random Forest hasil SMOTE
├── scaler_attrition_rf_smote.pkl        # Scaler StandardScaler
├── requirements.txt                     # Daftar library yang dibutuhkan
├── README.md                            # Dokumentasi proyek
└── Attrition_RF_SMOTE.ipynb             # Notebook training model
```

## ▶️ Cara Menjalankan di Lokal
1. Install dependensi:
   ```
   pip install -r requirements.txt
   ```

2. Jalankan aplikasi Streamlit:
   ```
   streamlit run streamlit_app_rf_smote_narasi.py
   ```

## 🌐 Cara Deploy ke Streamlit Cloud
1. Upload seluruh file ke repository GitHub publik
2. Buka [https://share.streamlit.io](https://share.streamlit.io)
3. Pilih file `streamlit_app_rf_smote_narasi.py` dan klik **Deploy**

---

## 👥 Anggota Kelompok
1. **Rosihin** – 22416255201002  
2. **Duma Zindy Aritonang** – 22416255201207

📌 Proyek ini disusun untuk memenuhi tugas akhir mata kuliah **Data Science**, dengan fokus pada penerapan machine learning untuk permasalahan prediksi risiko attrition karyawan.
