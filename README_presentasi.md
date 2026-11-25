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
* **Tiga Perbaikan Metodologi Utama**:
  1. **Menghilangkan *Data Leakage* pada Preprocessing**: Dataset dibagi terlebih dahulu menjadi *Train* (80% = 2.620 sampel) dan *Test* (20% = 656 sampel) secara *stratified*. `SimpleImputer(strategy="median")` dan `StandardScaler()` dikemas dalam `preprocessor_blueprint` dan **di-fit ulang di dalam setiap *fold* pelatihan**, sehingga informasi distribusi data uji tidak pernah bocor ke proses pelatihan.
  2. **Pemilihan Model Berbasis 5-Fold Stratified CV (Mencegah *Overfitting*)**: Alih-alih memilih model hanya dari satu kali evaluasi *Test*, pemilihan model dilakukan menggunakan **5-Fold Stratified Cross-Validation**. Hasilnya, **Calibrated SVM (RBF Kernel)** terpilih sebagai model terbaik dengan **CV ROC-AUC tertinggi sebesar `0.7019 ± 0.0179`** dan generalisasi stabil, mengungguli Random Forest yang mengalami *overfitting* ekstrem (`Train ROC-AUC = 0.9760` vs `CV ROC-AUC = 0.6724`).
  3. **Pengurangan *False Positive* (FP) via *Out-of-Fold (OOF) Threshold Tuning***: Dalam konteks keselamatan air minum, *False Positive* (air berbahaya diprediksi layak minum) memiliki risiko kesehatan tertinggi. Pencarian ambang batas dilakukan secara adil menggunakan prediksi **Out-of-Fold (OOF)** data *Train* (tanpa menyentuh data *Test*), menghasilkan threshold rekomendasi **`0.60`** (naik dari `0.50`):
     * **Pada OOF (Train)**: Kesalahan FP turun **56.4%** (dari `188` menjadi `82` kasus) dengan *OOF Precision* meningkat dari `66.73%` menjadi `75.30%`.
     * **Pada Data Test (Evaluasi Akhir)**: Kesalahan FP turun **41.3%** (dari `46` menjadi `27` kasus), dengan **ROC-AUC = `0.6487`**, **Precision = `70.33%`**, **Specificity = `93.25%`**, dan **Accuracy = `66.62%`**.
* **Fitur Paling Berpengaruh (*Permutation Importance*)**: `Sulfate` (`0.0876`), `ph` (`0.0743`), dan `Hardness` (`0.0495`).

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
2. **Random Forest Classifier (`n_estimators=100, max_depth=10`)**: Model *ensemble bagging* berbasis pohon keputusan yang mampu menangkap interaksi non-linear antar fitur.
3. **Gradient Boosting Classifier (`n_estimators=100, max_depth=3`)**: Model *ensemble boosting* yang membangun pohon secara sekuensial untuk memperbaiki galat residual pohon sebelumnya.
4. **K-Nearest Neighbors (`KNN, n_neighbors=5`)**: Algoritma *instance-based* berbasis jarak Euclidean pada ruang fitur yang telah distandardisasi.
5. **Logistic Regression**: Model klasifikasi linear sebagai acuan dasar (*baseline*).

---

## 4. Metrik Evaluasi & Alasannya

### 4.1 Mengapa Menggunakan **5-Fold CV ROC-AUC** untuk Seleksi Model?
* **Accuracy** pada threshold default `0.50` rentan bias terhadap kelas mayoritas (`60.99%` tidak layak).
* **ROC-AUC** mengukur kemampuan diskriminasi model di seluruh kemungkinan *decision threshold* (`0.0` hingga `1.0`). Dengan menghitung rata-rata dan standar deviasi ROC-AUC pada **5-Fold Stratified Cross-Validation**, stabilitas model dan risiko *overfitting* (selisih `Train AUC` vs `CV AUC`) dapat terukur secara objektif.

### 4.2 Mengapa Prioritas *Threshold Tuning* Adalah Menekan **False Positive (FP)**?
* **False Positive (FP — Bahaya Kesehatan)**: Air sebenarnya **TIDAK LAYAK / TERKONTAMINASI (`0`)**, tetapi diprediksi **LAYAK MINUM (`1`)**. Kesalahan ini berisiko langsung membahayakan kesehatan konsumen.
* **False Negative (FN — Konservatif / Aman)**: Air sebenarnya **LAYAK (`1`)**, tetapi diprediksi **TIDAK LAYAK (`0`)**. Dampaknya hanya memerlukan penyaringan ulang atau pemanfaatan non-konsumsi, tanpa risiko keracunan.

---

## 5. Struktur Proyek & Kode Lengkap

```text
projectML2/
├── data/
│   └── water_potability.csv              # Dataset utama (3.276 sampel)
├── model/
│   ├── model_info.json                   # Metadata model, threshold rekomendasi, & statistik fitur
│   ├── model_pipeline.joblib             # Pipeline end-to-end (Imputer + Scaler + Calibrated SVM)
│   └── preprocessor_blueprint.joblib     # Blueprint preprocessing untuk validasi silang
├── notebooks/
│   └── 02+03_preprocess_train_eval.ipynb # Eksperimen interaktif CV, OOF tuning, & GridSearchCV
├── output/
│   ├── class_balance.csv                 # Distribusi kelas target
│   ├── corr_with_target.csv              # Korelasi Pearson fitur terhadap Potability
│   ├── correlation_matrix.csv            # Matriks korelasi antar fitur
│   ├── descriptive_stats.csv             # Statistik deskriptif dataset
│   ├── feature_cols.json                 # Urutan kolom fitur standar
│   ├── feature_stats.json                # Nilai min, max, dan median tiap fitur
│   ├── missing_stats.csv                 # Statistik nilai hilang (missing values)
│   ├── model_comparison.csv              # Perbandingan Train, CV, dan Test 5 algoritma
│   ├── permutation_importance.csv        # Skor Permutation Importance pada data Test
│   ├── threshold_tuning.csv              # Evaluasi trade-off OOF pada threshold 0.30–0.80
│   ├── train.csv                         # Data latih hasil split 80% (2.620 baris)
│   └── test.csv                          # Data uji hasil split 20% (656 baris)
├── scripts/
│   ├── 01_eda.py                         # Script Exploratory Data Analysis
│   ├── 02_preprocess.py                  # Script Stratified Split & Blueprint Preprocessing
│   ├── 03_train_evaluate.py              # Script 5-Fold CV, OOF Tuning, & Export Pipeline
│   └── cek_cepat.py                      # Script pemeriksaan skewness & missing values
├── app.py                                # Aplikasi Web Streamlit 5-halaman (termasuk SHAP)
├── predict_utils.py                      # Modul inferensi & validasi input aplikasi
├── requirements.txt                      # Daftar dependensi Python
├── README.md                             # Dokumentasi utama repositori
└── README_presentasi.md                  # Panduan lengkap presentasi & diskusi tim

```

---

### 5.1 Script 1: `scripts/01_eda.py` (Exploratory Data Analysis)

```python
"""01_eda.py — EDA (Exploratory Data Analysis) untuk dataset water potability."""

import json
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "water_potability.csv")
TARGET_COL = "Potability"

df = pd.read_csv(DATA_PATH)

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

corr_matrix = df.corr(numeric_only=True)
corr_matrix.to_csv(os.path.join(OUT_DIR, "correlation_matrix.csv"))

feature_cols = [c for c in df.columns if c != TARGET_COL]
corr_target = corr_matrix[TARGET_COL].drop(TARGET_COL)
corr_target = corr_target.reindex(corr_target.abs().sort_values(ascending=False).index)
corr_target_df = corr_target.reset_index()
corr_target_df.columns = ["fitur", "korelasi_dengan_potability"]
corr_target_df.to_csv(os.path.join(OUT_DIR, "corr_with_target.csv"), index=False)

desc = df[feature_cols].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
desc.to_csv(os.path.join(OUT_DIR, "descriptive_stats.csv"))

feature_stats = {
    col: {
        "min": float(df[col].min()),
        "max": float(df[col].max()),
        "median": float(df[col].median()),
    }
    for col in feature_cols
}
with open(os.path.join(OUT_DIR, "feature_stats.json"), "w") as f:
    json.dump(feature_stats, f, indent=2)

```

---

### 5.2 Script 2: `scripts/02_preprocess.py` (Stratified Split & Blueprint Preprocessing)

```python
"""02_preprocess.py — Pembagian Data (80/20) & Definisi Pipeline Preprocessing."""

import json
import os
import joblib
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

# 1. Stratified Train/Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

train_df = X_train.copy()
train_df[TARGET_COL] = y_train
test_df = X_test.copy()
test_df[TARGET_COL] = y_test

# 2. Definisi Blueprint Preprocessing (Median Imputation + Standard Scaling)
preprocessor_blueprint = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), feature_cols),
    ],
    remainder="drop",
)

# 3. Simpan hasil split & blueprint preprocessor
train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

with open(os.path.join(OUT_DIR, "feature_cols.json"), "w") as f:
    json.dump(feature_cols, f, indent=2)

joblib.dump(preprocessor_blueprint, os.path.join(MODEL_DIR, "preprocessor_blueprint.joblib"))

```

---

### 5.3 Script 3: `scripts/03_train_evaluate.py` (5-Fold CV, OOF Tuning, & Export Pipeline)

```python
"""03_train_evaluate.py — Training 5 Model (5-Fold CV), OOF Threshold Tuning, & Export Pipeline."""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

RANDOM_STATE = 42
TARGET_COL = "Potability"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "model")

train_df = pd.read_csv(os.path.join(OUT_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(OUT_DIR, "test.csv"))
preprocessor_blueprint = joblib.load(os.path.join(MODEL_DIR, "preprocessor_blueprint.joblib"))

with open(os.path.join(OUT_DIR, "feature_cols.json")) as f:
    feature_cols = json.load(f)

X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
X_test, y_test = test_df[feature_cols], test_df[TARGET_COL]

models = {
    "SVM (RBF)": CalibratedClassifierCV(
        SVC(C=1.0, kernel="rbf", random_state=RANDOM_STATE),
        cv=5, method="sigmoid", ensemble=False
    ),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_results = []
oof_proba = {}

for name, model in models.items():
    fold_aucs = []
    proba = np.zeros(len(y_train))

    for tr_idx, va_idx in cv.split(X_train, y_train):
        pipe = make_pipeline(clone(preprocessor_blueprint), clone(model))
        pipe.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        p_val = pipe.predict_proba(X_train.iloc[va_idx])[:, 1]
        proba[va_idx] = p_val
        fold_aucs.append(roc_auc_score(y_train.iloc[va_idx], p_val))

    oof_proba[name] = proba

    full_pipe = make_pipeline(clone(preprocessor_blueprint), clone(model))
    full_pipe.fit(X_train, y_train)
    p_train = full_pipe.predict_proba(X_train)[:, 1]
    p_test = full_pipe.predict_proba(X_test)[:, 1]
    pred_test_05 = (p_test >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_test_05).ravel()

    cv_results.append({
        "Model": name,
        "Train_ROC_AUC": round(roc_auc_score(y_train, p_train), 4),
        "CV_ROC_AUC_Mean": round(float(np.mean(fold_aucs)), 4),
        "CV_ROC_AUC_Std": round(float(np.std(fold_aucs)), 4),
        "Test_ROC_AUC": round(roc_auc_score(y_test, p_test), 4),
        "Test_Accuracy": round(accuracy_score(y_test, pred_test_05), 4),
        "Test_Precision": round(precision_score(y_test, pred_test_05, zero_division=0), 4),
        "Test_Recall": round(recall_score(y_test, pred_test_05, zero_division=0), 4),
        "Test_F1": round(f1_score(y_test, pred_test_05, zero_division=0), 4),
        "Test_TP": int(tp), "Test_TN": int(tn), "Test_FP": int(fp), "Test_FN": int(fn),
    })

comparison_df = pd.DataFrame(cv_results).sort_values(by="CV_ROC_AUC_Mean", ascending=False)
comparison_df.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False)

best_model_name = comparison_df.iloc[0]["Model"]

# Threshold Tuning via Out-of-Fold (OOF) Predictions (Zero Test Leakage)
y_proba_oof = oof_proba[best_model_name]
thresh_rows = []
for th in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
    y_pred_oof = (y_proba_oof >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred_oof).ravel()
    thresh_rows.append({
        "Threshold": th,
        "OOF_Accuracy": round(accuracy_score(y_train, y_pred_oof), 4),
        "OOF_Precision": round(precision_score(y_train, y_pred_oof, zero_division=0), 4),
        "OOF_Recall": round(recall_score(y_train, y_pred_oof, zero_division=0), 4),
        "OOF_F1": round(f1_score(y_train, y_pred_oof, zero_division=0), 4),
        "OOF_FP": int(fp), "OOF_FN": int(fn), "OOF_TP": int(tp), "OOF_TN": int(tn),
    })

thresh_df = pd.DataFrame(thresh_rows)
thresh_df.to_csv(os.path.join(OUT_DIR, "threshold_tuning.csv"), index=False)

base_fp = thresh_df.loc[thresh_df["Threshold"] == 0.5, "OOF_FP"].values[0]
matching = thresh_df[thresh_df["OOF_FP"] <= base_fp * 0.6]
optimal_row = matching.iloc[0] if len(matching) > 0 else thresh_df.loc[thresh_df["OOF_FP"].idxmin()]
opt_th = float(optimal_row["Threshold"])

# Final Fit pada Full X_train & Evaluasi Akhir pada X_test
final_pipeline = make_pipeline(clone(preprocessor_blueprint), clone(models[best_model_name]))
final_pipeline.fit(X_train, y_train)

proba_test_final = final_pipeline.predict_proba(X_test)[:, 1]
y_pred_test_final = (proba_test_final >= opt_th).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test_final).ravel()

perm_imp = permutation_importance(
    final_pipeline, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=RANDOM_STATE
)
imp_df = pd.DataFrame({
    "fitur": feature_cols,
    "importance_mean": perm_imp.importances_mean,
    "importance_std": perm_imp.importances_std,
}).sort_values(by="importance_mean", ascending=False)
imp_df.to_csv(os.path.join(OUT_DIR, "permutation_importance.csv"), index=False)

joblib.dump(final_pipeline, os.path.join(MODEL_DIR, "model_pipeline.joblib"))

```

---

### 5.4 Implementasi Aplikasi Web (`predict_utils.py` & `app.py`)

Aplikasi web dibangun menggunakan **Streamlit** yang memuat artefak `model/model_pipeline.joblib` (gabungan *Imputer + Scaler + Calibrated SVM*) sehingga preprocessing saat inferensi dijamin identik 100% dengan saat pelatihan:

1. **Home**: Ringkasan dashboard, informasi ambang batas aktif (`0.60`), serta tabel perbandingan hasil evaluasi 5 model.
2. **Prediction**: Form input 9 parameter fisikokimia (nilai default otomatis terisi median data latih), *slider* pengatur *threshold* dinamis (`0.10–0.90`), validasi *out-of-range* terhadap rentang data latih (`feature_stats.json`), serta visualisasi probabilitas kelas.
3. **Model Explanation (SHAP)**: Penjelasan lokal (*instance-level*) berbasis **SHAP Waterfall Plot** dan **Feature Contribution Bar** untuk memvisualisasikan parameter mana yang mendorong air menuju status *Potable* atau *Not Potable*.
4. **Comparison Mode (Batch CSV)**: Pengunggahan file CSV berisi banyak sampel air sekaligus, prediksi massal, grafik ringkasan, inspeksi detail per baris, serta fitur unduh hasil prediksi (`water_potability_predictions.csv`).
5. **Live Monitoring**: Simulasi aliran data sensor air secara berkelanjutan (*non-blocking `@st.fragment*`, interval 2 detik) yang menampilkan metrik status real-time dan grafik tren 20 pembacaan terakhir.

---

## 6. Hasil Eksperimen & Analisis Output

### 6.1 Perbandingan 5 Model (5-Fold Stratified CV & Evaluasi Test pada Threshold `0.50`)

| Model | Train ROC-AUC | **CV ROC-AUC (Mean ± Std)** | Test ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1 | TP | TN | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **SVM (RBF)** 🏆 | 0.8184 | **`0.7019 ± 0.0179`** | 0.6487 | 0.6692 | 0.6489 | 0.3320 | **0.4393** | 85 | 354 | 46 | 171 |
| **Random Forest** | **0.9760** | `0.6724 ± 0.0129` | **0.6750** | 0.6692 | **0.7600** | 0.2227 | 0.3444 | 57 | 382 | 18 | 199 |
| **Gradient Boosting** | 0.8381 | `0.6343 ± 0.0221` | 0.6285 | 0.6524 | 0.6429 | 0.2461 | 0.3559 | 63 | 365 | 35 | 193 |
| **KNN** | 0.8325 | `0.6299 ± 0.0205` | 0.5978 | 0.6143 | 0.5093 | 0.3203 | 0.3933 | 82 | 321 | 79 | 174 |
| **Logistic Regression** | 0.5133 | `0.4774 ± 0.0177` | 0.5481 | 0.6098 | 0.0000 | 0.0000 | 0.0000 | 0 | 400 | 0 | 256 |

* **Analisis Perbandingan Model**:
* **SVM (RBF)** unggul signifikan pada validasi silang 5-Fold (`CV ROC-AUC = 0.7019`) dengan selisih *Train–CV* yang moderat (`0.8184` vs `0.7019`), menandakan kemampuan generalisasi terbaik.
* **Random Forest** menunjukkan indikasi *overfitting* berat (`Train ROC-AUC = 0.9760` turun drastis ke `CV ROC-AUC = 0.6724`), sehingga meskipun skor `Test ROC-AUC` kebetulan lebih tinggi pada satu kali split, model ini kurang andal dibanding SVM secara validasi silang.
* **Logistic Regression** gagal memprediksi kelas positif (`Recall = 0.0000`) karena batas keputusan kualitas air bersifat non-linear.



---

### 6.2 Hasil *Out-of-Fold (OOF) Threshold Tuning* (Model Terpilih: Calibrated SVM RBF)

| Threshold | OOF Accuracy | OOF Precision | OOF Recall | OOF F1-Score | **OOF FP (Air Bahaya Dibilang Safe)** | **OOF FN (Air Safe Dibatalkan)** | OOF TP | OOF TN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.30 | 0.5782 | 0.4754 | 0.7838 | 0.5918 | 884 | 221 | 801 | 714 |
| 0.40 | 0.6706 | 0.5817 | 0.5538 | 0.5674 | 407 | 456 | 566 | 1191 |
| **0.50 (Default)** | **0.6821** | **0.6673** | **0.3689** | **0.4751** | **188** | **645** | **377** | **1410** |
| 0.55 | 0.6790 | 0.7119 | 0.2975 | 0.4196 | 123 | 718 | 304 | 1475 |
| ⭐ **0.60 (Rekomendasi)** | **0.6740** | **0.7530** | **0.2446** | **0.3693** | **82** | **772** | **250** | **1516** |
| 0.65 | 0.6622 | 0.7729 | 0.1898 | 0.3048 | 57 | 828 | 194 | 1541 |
| 0.70 | 0.6542 | 0.8053 | 0.1497 | 0.2525 | 37 | 869 | 153 | 1561 |
| 0.75 | 0.6408 | 0.8293 | 0.0998 | 0.1782 | 21 | 920 | 102 | 1577 |
| 0.80 | 0.6298 | 0.8171 | 0.0656 | 0.1214 | 15 | 955 | 67 | 1583 |

* **Perbandingan Evaluasi Akhir pada Data Test (656 Sampel) — Sebelum vs Sesudah Threshold Tuning**:

| Metrik Evaluasi (Data Test) | Threshold Default (`0.50`) | Threshold Rekomendasi (`0.60`) | Perubahan / Dampak |
| --- | --- | --- | --- |
| **ROC-AUC** | `0.6487` | `0.6487` | Tetap (metrik ranking bebas threshold) |
| **False Positive (FP)** | `46` | **`27`** | **Turun 41.3% (-19 kasus berbahaya lolos)** |
| **Precision (Akurasi Layak)** | `64.89%` | **`70.33%`** | **Naik +5.44%** |
| **Specificity (Deteksi Air Bahaya)** | `88.50%` | **`93.25%`** | **Naik +4.75%** |
| **Accuracy** | `66.92%` | `66.62%` | Stabil (-0.30%) |
| **Recall / F1-Score** | `33.20%` / `0.4393` | `25.00%` / `0.3689` | *Trade-off* wajar demi keselamatan FP |
| **Confusion Matrix (`TP, TN, FP, FN`)** | `85, 354, 46, 171` | **`64, 373, 27, 192`** | Lebih konservatif dan aman |

---

### 6.3 Tingkat Kepentingan Fitur (*Permutation Importance* pada Pipeline Terbaik)

| Fitur | Mean Importance (Penurunan ROC-AUC) | Std Importance | Kategori Pengaruh |
| --- | --- | --- | --- |
| **Sulfate** | **`0.0876`** | `0.0119` | Sangat Penting |
| **ph** | **`0.0743`** | `0.0106` | Sangat Penting |
| **Hardness** | **`0.0495`** | `0.0071` | Penting |
| **Solids** | **`0.0429`** | `0.0088` | Cukup Penting |
| **Chloramines** | **`0.0274`** | `0.0087` | Cukup Penting |
| **Organic_carbon** | `-0.0004` | `0.0088` | Sangat Rendah / Noise |
| **Conductivity** | `-0.0073` | `0.0058` | Tidak Berpengaruh |
| **Trihalomethanes** | `-0.0085` | `0.0047` | Tidak Berpengaruh |
| **Turbidity** | `-0.0160` | `0.0063` | Tidak Berpengaruh |

---

## 7. Pertanyaan Umum (FAQ) untuk Diskusi & Tanya Jawab Presentasi

### Q1: Mengapa korelasi linear (Pearson) semua fitur terhadap `Potability` sangat kecil (`< 0.035`)?

**Jawab**: Korelasi Pearson hanya menangkap hubungan **garis lurus (monotonik linear)**. Dalam standar kualitas air minum (seperti pedoman WHO), kelayakan ditentukan oleh **rentang ambang batas (*band-pass*)**—misalnya pH aman berada di kisaran `6.5–8.5`; air yang terlalu asam (`< 6.5`) maupun terlalu basa (`> 8.5`) sama-sama tidak layak minum. Pola non-linear semacam ini membuat korelasi linearnya mendekati nol dan menyebabkan model linear seperti Logistic Regression gagal memisahkan kelas.

### Q2: Mengapa memilih **Calibrated SVM (RBF)** dibanding Random Forest padahal `Test ROC-AUC` Random Forest (`0.6750`) lebih tinggi dari SVM (`0.6487`)?

**Jawab**: Pemilihan model terbaik tidak boleh didasarkan pada satu kali evaluasi *Test* karena rentan terhadap fluktuasi sampel tunggal. Pada evaluasi **5-Fold Stratified Cross-Validation**, Random Forest mengalami *overfitting* berat (`Train ROC-AUC = 0.9760` turun ke `CV ROC-AUC = 0.6724`), sedangkan **SVM (RBF)** mencapai **`CV ROC-AUC = 0.7019 ± 0.0179`** (tertinggi di antara kelima algoritma). Selain itu, pembungkusan dengan `CalibratedClassifierCV` memastikan keluaran probabilitas SVM terkalibrasi dengan baik untuk penentuan ambang batas keselamatan.

### Q3: Mengapa *Threshold Tuning* dilakukan pada prediksi **Out-of-Fold (OOF)** dan bukan langsung pada data Test?

**Jawab**: Jika ambang batas (`0.60`) dipilih berdasarkan hasil pada data Test, maka data Test secara tidak langsung ikut memengaruhi parameter keputusan model (*threshold leakage*). Dengan mencarinya menggunakan prediksi **Out-of-Fold (OOF)** dari 2.620 sampel latih (target penurunan `OOF_FP >= 40%`), data Test (656 sampel) benar-benar dijaga steril untuk validasi akhir yang jujur.

### Q4: Mengapa memilih `test_size = 0.2` (80/20) dan `random_state = 42`?

**Jawab**: Pembagian 80/20 dengan *stratified sampling* menjaga proporsi kelas (`60.99%` vs `39.01%`) tetap identik pada data latih (2.620 sampel) maupun uji (656 sampel). Nilai `random_state = 42` menjamin **reprodusibilitas** penuh sehingga pembagian data, lipatan 5-Fold CV, dan hasil evaluasi selalu konsisten setiap kali kode dijalankan ulang.

### Q5: Mengapa nilai *Permutation Importance* untuk `Turbidity` atau `Trihalomethanes` bernilai sedikit negatif (`-0.0160` dan `-0.0085`)?

**Jawab**: *Permutation Importance* mengukur penurunan skor ROC-AUC ketika nilai suatu fitur diacak secara acak pada data Test. Nilai mendekati nol atau sedikit negatif menandakan fitur tersebut tidak memberikan kontribusi sinyal prediktif tambahan pada batas keputusan SVM RBF (pengacakannya secara kebetulan mengurangi *noise* minor pada sampel uji), sedangkan lima parameter kimia utama (`Sulfate`, `ph`, `Hardness`, `Solids`, `Chloramines`) menjadi penentu dominan kelayakan air.

```

```