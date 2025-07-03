# HR Attrition Prediction App

Aplikasi ini dibuat untuk memprediksi apakah seorang karyawan akan keluar dari perusahaan (attrition) atau tidak, berdasarkan data HR seperti usia, gaji, masa kerja, tingkat pendidikan, dan faktor lainnya.

## 🔍 Dataset
Dataset yang digunakan adalah [IBM HR Analytics Attrition Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

## 🚀 Fitur Aplikasi
- Input data karyawan melalui form di Streamlit
- Prediksi risiko attrition (keluar) menggunakan model Random Forest
- Probabilitas prediksi ditampilkan secara real-time

## 🛠 Teknologi
- Python
- Streamlit
- Scikit-learn
- Pandas, NumPy
- Joblib

## 📁 Struktur File
```
📁 attrition-project/
├── streamlit_app.py             # Aplikasi utama Streamlit
├── requirements.txt             # Daftar dependensi
├── model_attrition.pkl          # File model prediksi
├── scaler_attrition.pkl         # File scaler preprocessing
└── README.md                    # Deskripsi proyek
```

## ▶️ Cara Menjalankan di Lokal
1. Install dependensi:
   ```
   pip install -r requirements.txt
   ```

2. Jalankan aplikasi:
   ```
   streamlit run streamlit_app.py
   ```

## 🌐 Deploy ke Streamlit Cloud
Upload semua file ke GitHub, lalu deploy via https://share.streamlit.io

---

📌 Proyek ini merupakan tugas akhir dari mata kuliah **Data Science**.


# Anggota Kelompok
1. Rosihin  22416255201002
2. Duma Zindy Aritonang 22416255201207