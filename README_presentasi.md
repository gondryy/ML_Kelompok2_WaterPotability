# Panduan Presentasi & Rangkuman Proyek Machine Learning: Prediksi Kualitas Air Minum (Water Potability)

Dokumen ini disusun sebagai panduan lengkap untuk bahan presentasi, video demonstrasi, diskusi tim, dan sesi tanya jawab Tugas Akhir Mata Kuliah **TEK1302 Pembelajaran Mesin**, Program Studi Teknologi Rekayasa Komputer, Sekolah Vokasi IPB University.

---

## 👥 Identitas Kelompok (ML_Kelompok2_WaterPotability)

| No | Nama Mahasiswa | NIM | Kontribusi Utama |
| :--- | :--- | :--- | :--- |
| 1 | Juan Benedict Tanoto | J0404241001 | Modeling (5-Fold CV), OOF Threshold Tuning & Evaluasi |
| 2 | Phiello Abraham | J0404241025 | Dataset, EDA & Preprocessing |
| 3 | Muhamad Raditya | J0404241096 | Pengembangan Aplikasi Web (`app.py`, `predict_utils.py`) & Deployment |

---

## 📌 Ringkasan Eksekutif (Executive Summary)

* **Topik Utama**: Klasifikasi biner untuk memprediksi apakah suatu sampel air layak konsumsi (`Potability = 1`) atau tidak layak konsumsi (`Potability = 0`).
* **Dataset**: `water_potability.csv` terdiri dari **3.276 sampel air** dan **9 parameter fisikokimia** (`ph`, `Hardness`, `Solids`, `Chloramines`, `Sulfate`, `Conductivity`, `Organic_carbon`, `Trihalomethanes`, `Turbidity`).
* **Model yang Dibandingkan (5 Algoritma)**: `SVM (RBF)`, `Random Forest`, `CatBoost`, `KNN`, dan `Logistic Regression` (`CatBoost` menggantikan `GradientBoostingClassifier` standar).
* **Tiga Perbaikan Metodologi Utama**:
  1. **Menghilangkan *Data Leakage* pada Preprocessing**: Dataset dibagi terlebih dahulu menjadi *Train* (80% = 2.620 sampel) dan *Test* (20% = 656 sampel) secara *stratified*. `SimpleImputer(strategy="median")` dan `StandardScaler()` dikemas dalam `preprocessor_blueprint` dan **di-fit ulang di dalam setiap *fold* pelatihan**, sehingga informasi distribusi data uji tidak pernah bocor ke proses pelatihan.
  2. **Pemilihan Model Berbasis 5-Fold Stratified CV (Mencegah *Overfitting*)**: Alih-alih memilih model hanya dari satu kali evaluasi *Test*, pemilihan model dilakukan menggunakan **5-Fold Stratified Cross-Validation** berbasis **ROC-AUC** untuk membandingkan stabilitas generalisasi dan mendeteksi *overfitting* (`Train ROC-AUC` vs `CV ROC-AUC`) antar kelima model.
  3. **Pengurangan *False Positive* (FP) via *Out-of-Fold (OOF) Threshold Tuning***: Dalam konteks keselamatan air minum, *False Positive* (air berbahaya diprediksi layak minum) memiliki risiko kesehatan tertinggi. Pencarian ambang batas dilakukan secara adil menggunakan prediksi **Out-of-Fold (OOF)** data *Train* (tanpa menyentuh data *Test*) dengan target penurunan FP $\ge 40\%$.
* **Fitur Paling Berpengaruh (*Permutation Importance*)**: `Sulfate`, `ph`, dan `Hardness` (diikuti `Solids` dan `Chloramines`).

---

## 1. Latar Belakang & Karakteristik Dataset

### 1.1 Deskripsi Dataset
Dataset `water_potability.csv` berisi indikator kualitas fisikokimia air dari berbagai sumber air:
* **Jumlah Sampel**: 3.276 baris (2.011 baris lengkap jika *missing values* dihapus langsung, sehingga imputasi wajib dilakukan agar tidak kehilangan ~38.6% data).
* **Jumlah Kolom**: 10 (9 Fitur numerik + 1 Target biner `Potability`).
* **Keseimbangan Kelas (*Class Balance*)**:
  * `0` (Tidak Layak / *Non-Potable*): **1.998 sampel (60.99%)**
  * `1` (Layak Minum / *Potable*): **1.278 sampel (39.01%)**
* **Nilai Hilang (*Missing Values*)**:
  * `Sulfate`: **781 nilai hilang (23.84%)**
  * `ph`: **491 nilai hilang (14.99%)**
  * `Trihalomethanes`: **162 nilai hilang (4.95%)**
  * 6 fitur lainnya (`Hardness`, `Solids`, `Chloramines`, `Conductivity`, `Organic_carbon`, `Turbidity`): **0 nilai hilang (0.00%)**.

### 1.2 Statistik Deskriptif & Korelasi Linear dengan Target

| Fitur | Mean | Std | Median | Min | Max | Korelasi Pearson vs `Potability` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Solids` | 22014.09 | 8768.57 | 20927.83 | 320.94 | 61227.20 | `+0.0337` |
| `Organic_carbon` | 14.28 | 3.31 | 14.22 | 2.20 | 28.30 | `-0.0300` |
| `Chloramines` | 7.12 | 1.58 | 7.13 | 0.35 | 13.13 | `+0.0238` |
| `Sulfate` | 333.78 | 41.42 | 333.07 | 129.00 | 481.03 | `-0.0236` |
| `Hardness` | 196.37 | 32.88 | 196.97 | 47.43 | 323.12 | `-0.0138` |
| `Conductivity` | 426.21 | 80.82 | 421.88 | 181.48 | 753.34 | `-0.0081` |
| `Trihalomethanes` | 66.40 | 16.18 | 66.62 | 0.74 | 124.00 | `+0.0071` |
| `ph` | 7.08 | 1.59 | 7.04 | 0.00 | 14.00 | `-0.0036` |
| `Turbidity` | 3.97 | 0.78 | 3.96 | 1.45 | 6.74 | `+0.0016` |

### 1.3 Pemeriksaan Duplikat, Outlier & Interpretasi Visualisasi EDA

* **Data Duplikat**: Pemeriksaan `df.duplicated().sum()` menghasilkan **`0` baris duplikat** dari 3.276 sampel, sehingga seluruh baris merupakan observasi unik.
* **Analisis Outlier (Metode IQR $1.5\times$)**: Proporsi nilai ekstrem relatif kecil di seluruh fitur—tertinggi pada `Hardness` (**83 sampel / 2.53%**), `Chloramines` (**61 sampel / 1.86%**), `Solids` (**47 sampel / 1.43%**), `ph` (**46 sampel / 1.65%**), dan `Sulfate` (**41 sampel / 1.64%**). Seluruh nilai ekstrem ini **dipertahankan** karena merepresentasikan kondisi pencemaran air ekstrem di dunia nyata (misalnya air sangat asam/basa atau kadar mineral tinggi) yang krusial untuk dideteksi model, serta dampaknya sudah diredam oleh *Median Imputation* dan *Standard Scaling*.
* **Interpretasi 5 Visualisasi Utama (`output/eda_01` – `eda_05`)**:
  1. **Grafik Distribusi Kelas (`eda_01_class_balance.png`)**: Menunjukkan ketidakseimbangan kelas moderat (**60.99%** kelas `0` vs **39.01%** kelas `1`). Hal ini mendasari penggunaan *Stratified Split*, evaluasi berbasis **ROC-AUC** (bukan *Accuracy* semata), serta *Threshold Tuning*.
  2. **Grafik Missing Values (`eda_02_missing_values.png`)**: Nilai hilang hanya terkonsentrasi pada 3 parameter (`Sulfate` 23.84%, `ph` 14.99%, `Trihalomethanes` 4.95%). Menghapus baris kosong akan membuang **38.6%** dataset (tersisa 2.011 baris), sehingga imputasi median di dalam *fold* pelatihan menjadi solusi terbaik.
  3. **Heatmap Korelasi (`eda_03_correlation_heatmap.png`)**: Korelasi linear antarfitur maupun terhadap `Potability` sangat lemah ($\vert{}r\vert{} \le 0.17$ antarfitur dan $\vert{}r\vert{} \le 0.034$ terhadap target), membuktikan tidak ada multikolinearitas berat dan relasi menuju kelayakan air bersifat non-linear.
  4. **Boxplot 9 Parameter (`eda_04_feature_boxplots.png`)**: Median antara kelas `0` dan `1` tampak hampir berimpit secara univariat pada tiap fitur, yang menjelaskan mengapa pemisahan kelas membutuhkan kombinasi interaksi multi-parameter non-linear.
  5. **Histogram & KDE Top-3 Fitur (`eda_05_top_features_hist.png`)**: Memperlihatkan perbandingan sebaran probabilitas (`density`) untuk `Sulfate`, `ph`, dan `Hardness` antara kelas `0` (Tidak Layak) dan `1` (Layak).

---

## 2. Permasalahan Pipeline Lama & Solusi Anti-Leakage

### 2.1 Dua Sumber *Data Leakage* pada Pendekatan Konvensional
1. **Preprocessing Leakage**: Menghitung nilai median imputasi dan parameter *scaling* pada keseluruhan 3.276 baris sebelum pembagian data (*split*), sehingga statistik data *Test* bocor ke data *Train*.
2. **Threshold & Selection Leakage**: Memilih model terbaik dan mencari *decision threshold* langsung menggunakan performa pada data *Test*, sehingga metrik evaluasi akhir tidak lagi mencerminkan generalisasi murni pada data baru.

### 2.2 Arsitektur Pipeline Bebas Leakage pada `projectML2`
$$\text{Dataset Utuh (3.276)} \xrightarrow{\text{Stratified Split (80/20)}} \begin{cases} \text{Train (2.620)} \xrightarrow{\text{5-Fold CV (Fit Imputer+Scaler per Fold)}} \text{Seleksi Model \& OOF Threshold} \\ \text{Test (656)} \xrightarrow{\text{Evaluasi Akhir Sekali Pakai}} \text{Laporan Performa Murni} \end{cases}$$

---

## 3. Model yang Dibandingkan & Alasannya

Eksperimen membandingkan 5 algoritma klasifikasi dengan pendekatan matematis yang berbeda:

1. **Calibrated Support Vector Machine (`SVM RBF Kernel`)**: Mencari *hyperplane* pemisah non-linear optimal pada ruang fitur berdimensi tinggi menggunakan kernel RBF, dibungkus dengan `CalibratedClassifierCV` (*Platt scaling / sigmoid*, `cv=5`) agar menghasilkan probabilitas terkalibrasi secara akurat untuk kebutuhan *threshold tuning*.
2. **Random Forest Classifier (`n_estimators=100, max_depth=10`)**: Model *ensemble bagging* berbasis pohon keputusan yang mampu menangkap interaksi non-linear antar fitur dan mereduksi varians.
3. **CatBoost Classifier (`iterations=300, depth=5, learning_rate=0.05, auto_class_weights="Balanced"`)**: Model *gradient boosting* berbasis *symmetric/oblivious trees* yang menggantikan `GradientBoostingClassifier` standar untuk memodelkan pola non-linear dengan penyeimbangan bobot kelas.
4. **K-Nearest Neighbors (`KNN, n_neighbors=5`)**: Algoritma *instance-based* berbasis jarak pada ruang fitur yang telah distandardisasi.
5. **Logistic Regression**: Model klasifikasi linear sebagai acuan dasar (*baseline*) untuk membandingkan batas keputusan linear vs. non-linear.

---

## 4. Metrik Evaluasi & Alasannya

### 4.1 Mengapa Menggunakan **5-Fold CV ROC-AUC** untuk Seleksi Model?
* **Accuracy** pada threshold default `0.50` rentan bias terhadap kelas mayoritas (`60.99%` tidak layak).
* **ROC-AUC** mengukur kemampuan diskriminasi model di seluruh kemungkinan *decision threshold* (`0.0` hingga `1.0`). Dengan menghitung rata-rata dan standar deviasi ROC-AUC pada **5-Fold Stratified Cross-Validation**, stabilitas model dan risiko *overfitting* (selisih `Train AUC` vs `CV AUC`) dapat terukur secara objektif.

### 4.2 Mengapa Prioritas *Threshold Tuning* Adalah Menekan **False Positive (FP)**?
* **False Positive (FP — Bahaya Kesehatan)**: Air sebenarnya **TIDAK LAYAK / TERKONTAMINASI (`0`)**, tetapi diprediksi **LAYAK MINUM (`1`)**. Kesalahan ini berisiko langsung membahayakan kesehatan konsumen.
* **False Negative (FN — Konservatif / Aman)**: Air sebenarnya **LAYAK (`1`)**, tetapi diprediksi **TIDAK LAYAK (`0`)**. Dampaknya hanya memerlukan penyaringan ulang atau pemanfaatan non-konsumsi, tanpa risiko keracunan.

---

## 5. Struktur Proyek & Cara Menjalankan Pipeline

```text
projectML2/
├── data/
│   └── water_potability.csv              # Dataset utama (3.276 sampel)
├── model/
│   ├── model_info.json                   # Metadata model, threshold rekomendasi, & statistik fitur
│   ├── model_pipeline.joblib             # Pipeline end-to-end (Imputer + Scaler + Model Terbaik)
│   └── preprocessor_blueprint.joblib     # Blueprint preprocessing untuk validasi silang
├── notebooks/
│   └── 02+03_preprocess_train_eval.ipynb # Eksperimen interaktif CV & OOF tuning
├── output/
│   ├── class_balance.csv                 # Distribusi kelas target
│   ├── corr_with_target.csv              # Korelasi Pearson fitur terhadap Potability
│   ├── correlation_matrix.csv            # Matriks korelasi antar fitur
│   ├── descriptive_stats.csv             # Statistik deskriptif dataset
│   ├── eda_01_class_balance.png          # Grafik distribusi kelas
│   ├── eda_02_missing_values.png         # Grafik missing values
│   ├── eda_03_correlation_heatmap.png      # Heatmap korelasi Pearson
│   ├── eda_04_feature_boxplots.png       # Boxplot 9 fitur berdasarkan Potability
│   ├── eda_05_top_features_hist.png      # Histogram & KDE fitur utama
│   ├── feature_cols.json                 # Urutan kolom fitur standar
│   ├── feature_stats.json                # Nilai min, max, dan median tiap fitur
│   ├── missing_stats.csv                 # Statistik nilai hilang (missing values)
│   ├── model_comparison.csv              # Perbandingan Train, CV, dan Test 5 algoritma
│   ├── outlier_stats.csv                 # Statistik outlier metode IQR
│   ├── permutation_importance.csv        # Skor Permutation Importance pada data Test
│   ├── threshold_tuning.csv              # Evaluasi trade-off OOF pada threshold 0.30–0.80
│   ├── train.csv                         # Data latih hasil split 80% (2.620 baris)
│   └── test.csv                          # Data uji hasil split 20% (656 baris)
├── scripts/
│   ├── 01_eda.py                         # Script EDA & 5 visualisasi
│   ├── 02_preprocess.py                  # Script Stratified Split & Blueprint Preprocessing
│   ├── 03_train_evaluate.py              # Script 5-Fold CV (SVM, RF, CatBoost, KNN, LR), OOF Tuning, & Export
│   └── cek_cepat.py                      # Script pemeriksaan cepat skewness & missing values
├── app.py                                # Aplikasi Web Streamlit 5-halaman (termasuk SHAP)
├── predict_utils.py                      # Modul inferensi & validasi input aplikasi
├── requirements.txt                      # Daftar dependensi Python
├── README.md                             # Dokumentasi utama repositori
└── README_presentasi.md                  # Panduan lengkap presentasi & diskusi tim

```

### Perintah Eksekusi Pipeline (`python3`)

```bash
python3 -m pip install --user --break-system-packages -r requirements.txt catboost
python3 scripts/01_eda.py && python3 scripts/02_preprocess.py && python3 scripts/03_train_evaluate.py
python3 -m streamlit run app.py

```

---

## 6. Hasil Eksperimen & Analisis Output

### 6.1 Perbandingan 5 Model (5-Fold Stratified CV & Evaluasi Test pada Threshold `0.50`)

Isi tabel di bawah ini diambil langsung dari `output/model_comparison.csv` setelah menjalankan `python3 scripts/03_train_evaluate.py`:

| Model | Train ROC-AUC | **CV ROC-AUC (Mean ± Std)** | Test ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1 | TP | TN | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **SVM (RBF)** | *output/model_comparison.csv* | *output/model_comparison.csv* | *output/model_comparison.csv* | *output/model_comparison.csv* | *output/model_comparison.csv* | *output/model_comparison.csv* | *output/model_comparison.csv* | — | — | — | — |
| **Random Forest** | *output/model_comparison.csv* | *output/model_comparison.csv* | *output/model_comparison.csv* | *output/model_comparison.csv* | *output/model_comparison.csv* | *output/model_comparison.csv* | *output/model_comparison.csv* | — | — | — | — |
| **CatBoost** | *output/model_comparison.csv* | *output/model_comparison.csv* | *output/model_comparison.csv* | *output/model_comparison.csv* | *output/model_comparison.csv* | *output/model_comparison.csv* | *output/model_comparison.csv* | — | — | — | — |
| **KNN** | *output/model_comparison.csv* | *output/model_comparison.csv* | *output/model_comparison.csv* | *output/model_comparison.csv* | *output/model_comparison.csv* | *output/model_comparison.csv* | *output/model_comparison.csv* | — | — | — | — |
| **Logistic Regression** | *output/model_comparison.csv* | *output/model_comparison.csv* | *output/model_comparison.csv* | *output/model_comparison.csv* | *output/model_comparison.csv* | *output/model_comparison.csv* | *output/model_comparison.csv* | — | — | — | — |

### 6.2 Hasil *Out-of-Fold (OOF) Threshold Tuning* & *Permutation Importance*

* Tabel lengkap evaluasi *threshold* (`0.30` hingga `0.80`) pada prediksi Out-of-Fold tersimpan di `output/threshold_tuning.csv`, sedangkan *threshold* rekomendasi aktif tersimpan di `model/model_info.json`.
* Peringkat *Permutation Importance* seluruh 9 fitur pada data *Test* tersimpan di `output/permutation_importance.csv`.

---

## 7. Pertanyaan Umum (FAQ) untuk Diskusi & Tanya Jawab Presentasi

### Q1: Mengapa korelasi linear (Pearson) semua fitur terhadap `Potability` sangat kecil (`< 0.035`)?

**Jawab**: Korelasi Pearson hanya menangkap hubungan **garis lurus (monotonik linear)**. Dalam standar kualitas air minum (seperti pedoman WHO / Permenkes), kelayakan ditentukan oleh **rentang ambang batas (*band-pass*)**—misalnya pH aman berada di kisaran `6.5–8.5`; air yang terlalu asam (`< 6.5`) maupun terlalu basa (`> 8.5`) sama-sama tidak layak minum. Pola non-linear semacam ini membuat korelasi linearnya mendekati nol dan menyebabkan model linear seperti Logistic Regression kurang efektif memisahkan kelas.

### Q2: Mengapa mengganti `GradientBoostingClassifier` dengan `CatBoost`?

**Jawab**: `CatBoost` menggunakan *symmetric/oblivious trees* dan mendukung penyeimbangan bobot kelas (`auto_class_weights="Balanced"`), sehingga menjadi perbandingan *gradient boosting* yang lebih modern bersama `SVM (RBF)`, `Random Forest`, `KNN`, dan `Logistic Regression`.

### Q3: Mengapa *Threshold Tuning* dilakukan pada prediksi **Out-of-Fold (OOF)** dan bukan langsung pada data Test?

**Jawab**: Jika ambang batas dipilih berdasarkan hasil pada data Test, maka data Test secara tidak langsung ikut memengaruhi parameter keputusan model (*threshold leakage*). Dengan mencarinya menggunakan prediksi **Out-of-Fold (OOF)** dari data *Train* (target penurunan `OOF_FP >= 40%`), data *Test* (656 sampel) benar-benar dijaga steril untuk validasi akhir yang jujur.

### Q4: Mengapa memilih `test_size = 0.2` (80/20) dan `random_state = 42`?

**Jawab**: Pembagian 80/20 dengan *stratified sampling* menjaga proporsi kelas (`60.99%` vs `39.01%`) tetap identik pada data latih (2.620 sampel) maupun uji (656 sampel). Nilai `random_state = 42` menjamin **reprodusibilitas** penuh sehingga pembagian data, lipatan 5-Fold CV, dan hasil evaluasi selalu konsisten setiap kali kode dijalankan ulang.

---

## 8. Keterbatasan & Pengembangan Selanjutnya (Slide N)

### 8.1 Keterbatasan Saat Ini

1. **Keterbatasan Dataset**:
* Dataset hanya mencakup **9 parameter fisikokimia** dan belum menyertakan indikator **mikrobiologis** (seperti bakteri *E. coli* atau total koliform) maupun **logam berat** (timbal, arsenik, merkuri) yang sangat menentukan kelayakan air minum secara medis.
* Tingginya persentase data hilang pada `Sulfate` (23.84%) dan `ph` (14.99%) membatasi informasi asli yang tersedia pada sampel tersebut.


2. **Keterbatasan Model**:
* Skor ROC-AUC berada pada rentang moderat sehingga aplikasi ini diposisikan sebagai **alat skrining awal (*early screening*)**, bukan pengganti uji laboratorium tersertifikasi.
* Menaikkan *threshold* untuk menekan *False Positive* menimbulkan *trade-off* berupa penurunan *Recall* kelas layak (lebih banyak air layak yang diklasifikasikan secara konservatif sebagai tidak layak).


3. **Keterbatasan Aplikasi**:
* Fitur *Live Monitoring* pada dashboard Streamlit saat ini masih menggunakan generator data acak (*simulated sensor stream*) dan belum terhubung langsung ke perangkat keras sensor fisik.



### 8.2 Rencana Pengembangan Selanjutnya

1. **Penambahan Fitur & Rekayasa Fitur Domain (*Domain Feature Engineering*)**: Menambahkan rasio interaksi kimia (seperti rasio `Sulfate/Hardness` atau deviasi `pH` dari titik netral) serta mengintegrasikan parameter sensor mikrobiologi/logam berat pada pengumpulan data berikutnya.
2. **Eksplorasi Imputasi Multivariat & Ensemble**: Menguji metode imputasi lanjutan di dalam pipeline validasi silang untuk membandingkan akurasi pengisian nilai hilang `Sulfate` dan `pH` dibanding imputasi median univariat.
3. **Integrasi Perangkat Keras IoT Nyata**: Menghubungkan endpoint inferensi model dengan mikrokontroler **ESP32** dan sensor fisik (sensor pH, TDS/Conductivity, dan Turbidity) via protokol MQTT/HTTP REST API agar halaman *Live Monitoring* memantau kualitas sumber air secara *real-time*.

```

```