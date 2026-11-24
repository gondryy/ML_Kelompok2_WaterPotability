# ML_Kelompok2_WaterPotability
# 💧 Prediksi Kelayakan Air Minum (Water Potability Prediction)

Aplikasi web berbasis Machine Learning untuk memprediksi kelayakan konsumsi sampel air (`Potable = 1` vs `Non-Potable = 0`) berdasarkan 9 parameter fisikokimia air. Proyek ini disusun sebagai Tugas Akhir Mata Kuliah **TEK1302 Pembelajaran Mesin**, Program Studi Teknologi Rekayasa Komputer, Sekolah Vokasi IPB University.

---

## 👥 Anggota Kelompok

| No | Nama Mahasiswa | NIM | Kontribusi Utama |
| :--- | :--- | :--- | :--- |
| 1 | Juan Benedict Tanoto | J0404241001 | Modeling (5-Fold CV), OOF Threshold Tuning & Evaluasi |
| 2 | Phiello Abraham | J0404241025 | Dataset, EDA & Preprocessing |
| 3 | Muhamad Raditya | J0404241096 | Pengembangan Aplikasi Web (`app.py`) & Deployment |

---

## 🎯 Fitur Utama & Keunggulan Metodologi

1. **Pipeline Bebas *Data Leakage***: Dataset dibagi secara *stratified* (80% Train / 20% Test) terlebih dahulu. *Median Imputation* dan *Standard Scaling* di-fit secara eksklusif di dalam tiap *fold* pelatihan sehingga distribusi data uji tidak bocor ke proses pelatihan.
2. **Evaluasi Adil dengan 5-Fold Stratified Cross-Validation**: Membandingkan 5 algoritma klasifikasi (`SVM (RBF)`, `Random Forest`, `Gradient Boosting`, `KNN`, dan `Logistic Regression`) menggunakan metrik **ROC-AUC**.
3. **Optimasi Threshold Berbasis Keselamatan (*Safety-First Threshold Tuning*)**: Dalam domain air minum, kesalahan **False Positive (FP)** (air berbahaya diprediksi layak minum) memiliki risiko kesehatan tertinggi. Melalui kalibrasi probabilitas dan *Out-of-Fold (OOF) Threshold Tuning*, ambang batas dinaikkan dari `0.50` menjadi **`0.60`** untuk menekan FP secara signifikan.
4. **Aplikasi Web Interaktif (`Streamlit`)**: Mendukung prediksi sampel tunggal (termasuk simulasi imputasi otomatis untuk nilai kosong) maupun prediksi massal (*batch prediction*) melalui unggahan file CSV.

---

## 📊 Ringkasan Hasil Eksperimen

* **Dataset**: `water_potability.csv` (3.276 sampel, 9 parameter fisikokimia, rasio kelas `60.99%` Tidak Layak vs `39.01%` Layak).
* **Model Terpilih**: **Calibrated SVM (RBF Kernel)** — unggul pada 5-Fold CV ROC-AUC (`0.7019 ± 0.0179`) tanpa mengalami *overfitting* ekstrem seperti Random Forest (`Train AUC = 0.9760` vs `CV AUC = 0.6724`).
* **Performa Akhir pada Data Test (656 sampel, Threshold = `0.60`)**:
  * **ROC-AUC**: `0.6487`
  * **Precision (Akurasi Prediksi Layak)**: `0.7033`
  * **Specificity**: `0.9325`
  * **Accuracy**: `0.6662`
  * **Confusion Matrix**: `TP = 64`, `TN = 373`, `FP = 27`, `FN = 192`
* **Fitur Paling Berpengaruh (*Permutation Importance*)**: `Sulfate`, `ph`, dan `Hardness`.

---

## 🗂️ Struktur Direktori Proyek

```text
projectML2/
├── data/
│   └── water_potability.csv             # Dataset utama (3.276 sampel)
├── model/
│   ├── model_info.json                  # Metadata model, threshold rekomendasi, & statistik fitur
│   ├── model_pipeline.joblib            # Pipeline end-to-end (Imputer + Scaler + Calibrated SVM)
│   └── preprocessor_blueprint.joblib    # Blueprint preprocessing untuk validasi silang
├── notebooks/
│   └── 02+03_preprocess_train_eval.ipynb # Eksperimen interaktif CV, OOF tuning, & GridSearchCV
├── output/
│   ├── class_balance.csv                # Distribusi kelas target
│   ├── corr_with_target.csv             # Korelasi Pearson fitur terhadap Potability
│   ├── correlation_matrix.csv           # Matriks korelasi antar fitur
│   ├── descriptive_stats.csv            # Statistik deskriptif dataset
│   ├── feature_cols.json                # Urutan kolom fitur standar
│   ├── feature_stats.json               # Nilai min, max, dan median tiap fitur
│   ├── missing_stats.csv                # Statistik nilai hilang (missing values)
│   ├── model_comparison.csv             # Perbandingan Train, CV, dan Test 5 algoritma
│   ├── permutation_importance.csv       # Skor Permutation Importance pada data Test
│   ├── threshold_tuning.csv             # Evaluasi trade-off metrik pada threshold 0.30–0.80
│   ├── train.csv                        # Data latih hasil split 80%
│   └── test.csv                         # Data uji hasil split 20%
├── scripts/
│   ├── 01_eda.py                        # Script Exploratory Data Analysis
│   ├── 02_preprocess.py                 # Script Stratified Split & Blueprint Preprocessing
│   ├── 03_train_evaluate.py             # Script 5-Fold CV, OOF Tuning, & Export Model
│   └── cek_cepat.py                     # Script pemeriksaan skewness & missing values
├── app.py                               # Aplikasi Web Streamlit utama
├── predict_utils.py                     # Modul inferensi & validasi input aplikasi
├── requirements.txt                     # Daftar dependensi Python
└── README.md                            # Dokumentasi proyek
