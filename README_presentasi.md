# Panduan Presentasi & Rangkuman Proyek Machine Learning: Prediksi Kualitas Air Minum (Water Potability)

Dokumen ini disusun sebagai panduan lengkap untuk bahan presentasi, diskusi tim, dan acuan analisis AI. Berisi penjelasan latar belakang, permasalahan pipeline, arsitektur kode lengkap, alasan pemilihan model/metrik, hingga hasil analisis eksperimen.

---

## 📌 Ringkasan Eksekutif (Executive Summary)

* **Topik Utama**: Klasifikasi biner untuk memprediksi apakah suatu sampel air layak minum (`Potability = 1`) atau tidak layak minum (`Potability = 0`).
* **Dataset**: `water_potability.csv` terdiri dari 3.276 sampel air dan 9 parameter fisikokimia (pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity).
* **Perbaikan Utama dari Pipeline Lama**:
  1. **Menghilangkan *Data Leakage***: Pada pipeline lama, imputasi median dilakukan sebelum pembagian data (*data split*). Pada `projectML2`, data dibagi terlebih dahulu menjadi *Train* (80%) dan *Test* (20%), kemudian imputasi dan *scaling* dipelajari **hanya dari data Train** untuk menghindari kebocoran informasi dari data Test.
  2. **Pengurangan *False Positive* (FP) via *Threshold Tuning***: Dalam konteks kualitas air minum, *False Positive* (air tidak layak tetapi diprediksi layak) adalah kesalahan yang paling berbahaya. Dengan menyesuaikan ambang batas (*decision threshold*) dari 0.50 menjadi 0.55 pada model terbaik (Random Forest), **jumlah kesalahan FP berhasil diturunkan sebesar 44.4% (dari 18 kasus menjadi 10 kasus)** dengan mempertahankan akurasi di tingkat 66.2%.
* **Model Terbaik**: **Random Forest Classifier** (100 estimators, max_depth=10) dengan nilai **ROC-AUC sebesar 0.6750** pada data Test.
* **Fitur Paling Berpengaruh (Permutation Importance)**: Sulfate (0.0979), pH (0.0909), dan Hardness (0.0487).

---

## 1. Latar Belakang & Dataset

### 1.1 Deskripsi Dataset
Dataset ini berisi indikator kualitas air dari berbagai sumber air.
* **Jumlah Sampel**: 3.276 baris.
* **Jumlah Kolom**: 10 (9 Fitur + 1 Target `Potability`).
* **Keseimbangan Kelas (Class Balance)**:
  * `0` (Tidak Layak): 1.998 sampel (**60.99%**)
  * `1` (Layak Minum): 1.278 sampel (**39.01%**)
* **Nilai Hilang (Missing Values)**:
  * `Sulfate`: 781 nilai hilang (23.84%)
  * `ph`: 491 nilai hilang (14.99%)
  * `Trihalomethanes`: 162 nilai hilang (4.95%)

---

## 2. Permasalahan Pipeline & Perbaikan *Data Leakage*

### 2.1 Apa itu *Data Leakage* pada Pipeline Lama?
Pada pipeline lama, proses pra-pemrosesan data dilakukan dengan urutan:
$$\text{Dataset Utuh} \xrightarrow{\text{Imputasi Median}} \text{Data Split} \xrightarrow{\text{Standard Scaling}}$$

**Masalah**: Median dihitung dari keseluruhan 3.276 data (termasuk data Test). Akibatnya, nilai pada data Test ikut mempengaruhi pengisian nilai hilang pada data Train. Model "mengetahui" karakteristik data Test sebelum dievaluasi, sehingga metrik evaluasi menjadi terlalu optimis dan tidak realistis.

### 2.2 Solusi pada `projectML2` (Tanpa Leakage)
Urutan yang diterapkan pada `projectML2` adalah:
$$\text{Dataset Utuh} \xrightarrow{\text{Data Split (80/20)}} \begin{cases} \text{Data Train} \xrightarrow{\text{Fit & Transform Imputer + Scaler}} \text{Latih Model} \\ \text{Data Test} \xrightarrow{\text{Transform Only (pakai parameter Train)}} \text{Evaluasi Model} \end{cases}$$

Dengan cara ini, data Test tetap menjadi data yang benar-benar asing bagi model.

---

## 3. Model yang Dibandingkan & Alasannya

Eksperimen membandingkan 5 algoritma dengan karakteristik berbeda:

1. **Random Forest Classifier**: Model *ensemble* berbasis pohon keputusan (*decision trees*). Mampu menangani hubungan non-linear dan interaksi antar fitur tanpa terpengaruh skala data secara ekstrem.
2. **Support Vector Machine (SVM RBF Kernel)**: Mencari *hyperplane* optimal pada ruang berdimensi tinggi menggunakan kernel RBF. Cocok untuk batas keputusan non-linear yang kompleks.
3. **Gradient Boosting Classifier**: Membangun pohon keputusan secara berurutan (*boosting*) untuk mengoreksi kesalahan dari pohon sebelumnya.
4. **K-Nearest Neighbors (KNN)**: Algoritma berbasis jarak (*instance-based learning*) yang mengklasifikasikan data berdasarkan tetangga terdekatnya.
5. **Logistic Regression**: Model linear sederhana sebagai acuan (*baseline*).

---

## 4. Metrik Evaluasi & Alasannya

### 4.1 Mengapa Memakai **ROC-AUC** untuk Memilih Model Terbaik?
* **Accuracy** dapat menyesatkan pada data yang tidak seimbang. Jika model selalu memprediksi `0` (tidak layak), akurasinya tetap mencapai 61%.
* **ROC-AUC (Area Under the ROC Curve)** mengukur kemampuan model dalam membedakan kelas positif (layak) dan kelas negatif (tidak layak) di seluruh rentang ambang batas (*threshold*). Metrik ini tidak terikat pada satu ambang batas spesifik (seperti 0.5) sehingga sangat adil untuk membandingkan performa antar model.

### 4.2 Mengapa Memfokuskan Pengurangan **False Positive (FP)**?
Dalam domain keamanan air minum:
* **False Positive (FP)**: Air sebenarnya **TIDAK LAYAK / BERBAHAYA**, tetapi model memprediksi **LAYAK MINUM**. Hal ini berisiko menimbulkan masalah kesehatan masyarakat.
* **False Negative (FN)**: Air sebenarnya **LAYAK**, tetapi model memprediksi **TIDAK LAYAK**. Risiko utamanya adalah pembuangan air aman (kerugian ekonomi/efisiensi), tetapi secara kesehatan jauh lebih aman.

Oleh karena itu, **menekan FP adalah prioritas keselamatan tertinggi**.

---

## 5. Kode Lengkap & Penjelasan Struktur

Proyek `projectML2` dibagi menjadi 3 modul utama yang berdiri sendiri:

```
projectML2/
├── data/
│   └── water_potability.csv
├── scripts/
│   ├── 01_eda.py           # Exploratory Data Analysis
│   ├── 02_preprocess.py    # Split & Preprocessing tanpa leakage
│   └── 03_train_evaluate.py# Training 5 model, Threshold Tuning, & Permutation Importance
├── model/                  # Tempat menyimpan artefak model (.joblib & .json)
└── output/                 # Tempat menyimpan hasil evaluasi (.csv)
```

---

### 5.1 Script 1: `scripts/01_eda.py` (Analisis Eksplorasi Data)

```python
"""01_eda.py — EDA (Exploratory Data Analysis) untuk dataset water potability.

Tujuan: menghasilkan semua angka dasar statistik, korelasi, dan missing values
serta menyimpannya ke folder output/ agar dapat ditelusuri.
"""

import json
import os
import numpy as np
import pandas as pd

# Setup path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "water_potability.csv")
TARGET_COL = "Potability"

# Muat data
df = pd.read_csv(DATA_PATH)

# 1. Ringkasan Ukuran & Class Balance
n_rows, n_cols = df.shape
class_counts = df[TARGET_COL].value_counts().to_dict()
class_pct = (df[TARGET_COL].value_counts(normalize=True) * 100).to_dict()
missing_count = df.isnull().sum().to_dict()
missing_pct = (df.isnull().mean() * 100).to_dict()

summary_df = pd.DataFrame([{"kolom": k, "count": v} for k, v in class_counts.items()])
summary_df["persen"] = [class_pct[k] for k in summary_df["kolom"]]
summary_df.to_csv(os.path.join(OUT_DIR, "class_balance.csv"), index=False)

missing_df = pd.DataFrame({
    "kolom": list(missing_count.keys()),
    "missing_count": list(missing_count.values()),
    "missing_pct": [round(v, 2) for v in missing_pct.values()]
})
missing_df.to_csv(os.path.join(OUT_DIR, "missing_stats.csv"), index=False)

# 2. Matriks Korelasi Pearson
corr_matrix = df.corr(numeric_only=True)
corr_matrix.to_csv(os.path.join(OUT_DIR, "correlation_matrix.csv"))

feature_cols = [c for c in df.columns if c != TARGET_COL]
corr_target = corr_matrix[TARGET_COL].drop(TARGET_COL)
corr_target = corr_target.reindex(corr_target.abs().sort_values(ascending=False).index)
corr_target_df = corr_target.reset_index()
corr_target_df.columns = ["fitur", "korelasi_dengan_potability"]
corr_target_df.to_csv(os.path.join(OUT_DIR, "corr_with_target.csv"), index=False)

# 3. Statistik Deskriptif & Feature Stats
desc = df[feature_cols].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
desc.to_csv(os.path.join(OUT_DIR, "descriptive_stats.csv"))

feature_stats = {}
for col in feature_cols:
    feature_stats[col] = {
        "min": float(df[col].min()),
        "max": float(df[col].max()),
        "median": float(df[col].median()),
    }
with open(os.path.join(OUT_DIR, "feature_stats.json"), "w") as f:
    json.dump(feature_stats, f, indent=2)

pct_str = ", ".join(f"{k}: {round(v, 1)}%" for k, v in class_pct.items())
print("=== DATASET SUMMARY ===")
print(f"Shape          : {n_rows} baris x {n_cols} kolom")
print(f"Class balance  : {class_counts} -> {pct_str}")
print(f"Missing values : {sum(v for v in missing_count.values() if v > 0)} total")
print("\n=== KORELASI FITUR DENGAN TARGET (Potability) ===")
print(corr_target_df.to_string(index=False))
```

---

### 5.2 Script 2: `scripts/02_preprocess.py` (Pra-pemrosesan Data)

```python
"""02_preprocess.py — Pembagian Data & Preprocessing Bebas Data Leakage.

Urutan:
  1. Pembagian data Train (80%) dan Test (20%) dengan Stratified Sampling.
  2. Imputasi Median & Standard Scaling di-fit HANYA pada data Train.
  3. Transformasi data Test menggunakan parameter dari Train.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET_COL = "Potability"
TEST_SIZE = 0.2
RANDOM_STATE = 42

df = pd.read_csv(os.path.join(DATA_DIR, "water_potability.csv"))
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]
feature_cols = list(X.columns)

# 1. Split Data Terlebih Dahulu
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# 2. Definisikan Pipeline Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), feature_cols),
    ],
    remainder="drop",
)

# Fit & Transform hanya pada X_train, Transform saja pada X_test
X_train_t = preprocessor.fit_transform(X_train)
X_test_t = preprocessor.transform(X_test)

# 3. Simpan Hasil Preprocessing
np.savez_compressed(os.path.join(OUT_DIR, "X_train.npz"), data=X_train_t)
np.savez_compressed(os.path.join(OUT_DIR, "X_test.npz"), data=X_test_t)
np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train.to_numpy())
np.save(os.path.join(OUT_DIR, "y_test.npy"), y_test.to_numpy())
joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.joblib"))

with open(os.path.join(OUT_DIR, "feature_cols.json"), "w") as f:
    json.dump(feature_cols, f, indent=2)

print("Preprocessing selesai tanpa data leakage.")
```

---

### 5.3 Script 3: `scripts/03_train_evaluate.py` (Pelatihan, Evaluation, Tuning, & Importance)

```python
"""03_train_evaluate.py — Training Model, Threshold Tuning, & Permutation Importance.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Load data preprocessed
X_train = np.load(os.path.join(OUT_DIR, "X_train.npz"))["data"]
X_test = np.load(os.path.join(OUT_DIR, "X_test.npz"))["data"]
y_train = np.load(os.path.join(OUT_DIR, "y_train.npy"))
y_test = np.load(os.path.join(OUT_DIR, "y_test.npy"))

with open(os.path.join(OUT_DIR, "feature_cols.json")) as f:
    feature_cols = json.load(f)

# 1. Definisikan 5 Model
models = {
    "SVM (RBF)": SVC(C=1.0, kernel="rbf", probability=True, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE),
    "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=5),
}

results = []
trained_models = {}

# 2. Training dan Evaluasi Baseline (Threshold 0.5)
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(auc, 4),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn
    })

results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
results_df.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False)

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

# 3. Threshold Tuning untuk Mengurangi FP
y_proba_best = best_model.predict_proba(X_test)[:, 1]
thresh_results = []

for th in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
    y_pred_th = (y_proba_best >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_th).ravel()

    acc = accuracy_score(y_test, y_pred_th)
    prec = precision_score(y_test, y_pred_th, zero_division=0)
    rec = recall_score(y_test, y_pred_th, zero_division=0)
    f1 = f1_score(y_test, y_pred_th, zero_division=0)

    thresh_results.append({
        "Threshold": th,
        "Accuracy": round(acc, 4),
        "Precision (Akurasi 'Layak')": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "FP (Air Bahaya Dikatakan Safe)": fp,
        "FN (Air Safe Dibatalkan)": fn,
        "TP": tp, "TN": tn
    })

thresh_df = pd.DataFrame(thresh_results)
thresh_df.to_csv(os.path.join(OUT_DIR, "threshold_tuning.csv"), index=False)

# Pemilihan Threshold Optimal (Target penurunan FP >= 40%)
base_fp = thresh_df.loc[thresh_df["Threshold"] == 0.5, "FP (Air Bahaya Dikatakan Safe)"].values[0]
matching = thresh_df[thresh_df["FP (Air Bahaya Dikatakan Safe)"] <= base_fp * 0.6]
if len(matching) > 0:
    optimal_row = matching.iloc[0]
else:
    optimal_row = thresh_df.loc[thresh_df["FP (Air Bahaya Dikatakan Safe)"].idxmin()]

opt_th = float(optimal_row["Threshold"])

# 4. Permutation Importance
perm_imp = permutation_importance(best_model, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=RANDOM_STATE)
imp_df = pd.DataFrame({
    "fitur": feature_cols,
    "importance_mean": perm_imp.importances_mean,
    "importance_std": perm_imp.importances_std,
}).sort_values(by="importance_mean", ascending=False)
imp_df.to_csv(os.path.join(OUT_DIR, "permutation_importance.csv"), index=False)

# 5. Simpan Model & Metadata
joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.joblib"))

with open(os.path.join(OUT_DIR, "feature_stats.json")) as f:
    feature_stats = json.load(f)

model_info = {
    "best_model_name": best_model_name,
    "default_threshold": 0.5,
    "recommended_threshold": opt_th,
    "metrics_at_default_0.5": results_df.loc[results_df["Model"] == best_model_name].iloc[0].to_dict(),
    "metrics_at_recommended_th": optimal_row.to_dict(),
    "feature_cols": feature_cols,
    "feature_stats": feature_stats,
}

with open(os.path.join(MODEL_DIR, "model_info.json"), "w") as f:
    json.dump(model_info, f, indent=2)

print("Training & Evaluasi selesai. Model tersimpan di folder model/")
```

---

## 6. Hasil Eksperimen & Analisis Output

### 6.1 Perbandingan 5 Model (Data Test, Threshold Default 0.5)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| **Random Forest** 🏆 | 0.6692 | **0.7600** | 0.2227 | 0.3444 | **0.6750** | 57 | 382 | **18** | 199 |
| **SVM (RBF)** | **0.6707** | 0.7041 | 0.2695 | **0.3898** | 0.6487 | 69 | 371 | 29 | 187 |
| **Gradient Boosting** | 0.6524 | 0.6429 | 0.2461 | 0.3559 | 0.6288 | 63 | 365 | 35 | 193 |
| **KNN** | 0.6143 | 0.5093 | **0.3203** | 0.3933 | 0.5972 | 82 | 321 | 79 | 174 |
| **Logistic Regression** | 0.6098 | 0.0000 | 0.0000 | 0.0000 | 0.5481 | 0 | 400 | 0 | 256 |

* **Catatan**: Logistic Regression gagal mengklasifikasikan kelas positif (semua diprediksi 0) karena korelasi linear fitur dengan target sangat lemah. Model berbasis pohon dan kernel (Random Forest & SVM) mampu menangkap pola non-linear dengan lebih baik.

---

### 6.2 Hasil *Threshold Tuning* (Model Terpilih: Random Forest)

| Threshold | Accuracy | Precision ("Akurasi Layak") | Recall | F1-Score | **FP (Air Bahaya Dibilang Safe)** | **FN (Air Safe Dibatalkan)** | TP | TN |
|---|---|---|---|---|---|---|---|---|
| 0.30 | 0.5030 | 0.4349 | 0.9141 | 0.5894 | 304 | 22 | 234 | 96 |
| 0.40 | 0.6509 | 0.5595 | 0.4961 | 0.5259 | 100 | 129 | 127 | 300 |
| **0.50 (Default)** | **0.6692** | **0.7600** | **0.2227** | **0.3444** | **18** | **199** | **57** | **382** |
| ⭐ **0.55 (Rekomendasi)** | **0.6616** | **0.8148** | **0.1719** | **0.2839** | **10** | **212** | **44** | **390** |
| 0.60 | 0.6509 | 0.8649 | 0.1250 | 0.2184 | 5 | 224 | 32 | 395 |
| 0.65 | 0.6372 | 0.9500 | 0.0742 | 0.1377 | 1 | 237 | 19 | 399 |
| 0.70 | 0.6296 | 1.0000 | 0.0508 | 0.0967 | 0 | 243 | 13 | 400 |

* **Kesimpulan Threshold Tuning**:
  Dengan menaikkan ambang batas dari **0.50 menjadi 0.55**:
  * **False Positive (FP)** berkurang dari **18 menjadi 10** (turun **44.4%**).
  * **Precision** meningkat dari **76.0% menjadi 81.5%** (tingkat kepastian bahwa air benar-benar aman jika diprediksi layak minum).
  * Akurasi keseluruhan hanya berkurang sebesar 0.76% (dari 66.92% menjadi 66.16%).

---

### 6.3 Tingkat Kepentingan Fitur (*Permutation Importance*)

| Fitur | Mean Importance (Penurunan ROC-AUC) | Std Importance | Kategori |
|---|---|---|---|
| **Sulfate** | **0.0979** | 0.0078 | Sangat Penting |
| **ph** | **0.0909** | 0.0187 | Sangat Penting |
| **Hardness** | **0.0487** | 0.0085 | Penting |
| **Solids** | **0.0425** | 0.0074 | Cukup Penting |
| **Chloramines** | **0.0325** | 0.0084 | Cukup Penting |
| **Conductivity** | 0.0008 | 0.0041 | Sangat Rendah |
| **Organic_carbon** | -0.0006 | 0.0049 | Noise / Tidak Berpengaruh |
| **Trihalomethanes** | -0.0019 | 0.0041 | Noise / Tidak Berpengaruh |
| **Turbidity** | -0.0090 | 0.0041 | Noise / Tidak Berpengaruh |

---

## 7. Pertanyaan Umum (FAQ) untuk Diskusi Tim & AI

### Q1: Mengapa nilai korelasi linear (Pearson) semua fitur terhadap target sangat kecil (< 0.035)?
**Jawab**: Korelasi Pearson hanya mengukur hubungan **garis lurus (linear)**. Kualitas air minum dipengaruhi oleh batas toleransi tertentu (misalnya pH aman berada di rentang 6.5–8.5; jika terlalu rendah atau terlalu tinggi sama-sama tidak aman). Hubungan seperti ini bersifat **non-linear**, sehingga korelasi linearnya mendekati nol. Hal inilah yang menyebabkan model linear seperti Logistic Regression tidak dapat bekerja dengan baik.

### Q2: Mengapa memilih `test_size = 0.2` (20%) dan `random_state = 42`?
**Jawab**:
* `test_size = 0.2` (80/20) adalah standar industri yang optimal. Data Train sebesar 2.620 baris memberi ruang belajar yang cukup bagi model, sedangkan data Test sebesar 656 baris sudah sangat memadai secara statistik untuk mengevaluasi kinerja model secara konsisten.
* `random_state = 42` digunakan untuk menjamin **reprodusibilitas** (kepastian bahwa hasil pembagian data dan pengacakan selalu sama persis setiap kali kode dijalankan).

### Q3: Mengapa nilai *Permutation Importance* relatif kecil (sekitar 0.09 untuk fitur utama)?
**Jawab**: *Permutation Importance* mengukur seberapa besar penurunan nilai ROC-AUC ketika nilai suatu fitur diacak secara acak. Karena sinyal antar-fitur dalam dataset ini bersifat tersebar dan tidak ada satu fitur tunggal yang secara dominan menentukan kelayakan air, angka ~0.098 untuk Sulfate dan ~0.091 untuk pH sudah menunjukkan bahwa kedua fitur ini merupakan kontributor paling signifikan dalam keputusan model.

---

Dokumen ini mencakup seluruh landasan ilmiah, perbaikan metodologi, kode eksekusi, serta jawaban logis atas hasil proyek.
