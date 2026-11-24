"""01_eda.py — EDA (Exploratory Data Analysis) untuk dataset water potability.

Tujuan: menghasilkan semua angka yang dibutuhkan pipeline, DAN menyimpannya
ke file CSV/JSON supaya setiap angka bisa ditelusuri ke sumbernya.

Output (ditulis ke folder ../output/):
  - dataset_summary.csv    : shape, class balance, missing value count & persen
  - correlation_matrix.csv : matriks korelasi semua fitur (incl. target)
  - corr_with_target.csv   : korelasi tiap fitur dengan Potability (diurutkan)
  - missing_stats.csv      : count & persen missing per kolom
  - descriptive_stats.csv  : describe() tiap fitur (mean/std/min/25%/50%/75%/max)
  - feature_stats.json     : min/max/median tiap fitur (untuk UI & threshold)

Cara pakai:
  python scripts/01_eda.py
"""

import json
import os

import numpy as np
import pandas as pd

# --- 1. Tentukan path -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "water_potability.csv")
TARGET_COL = "Potability"

# --- 2. Muat dataset --------------------------------------------------------
df = pd.read_csv(DATA_PATH)

# --- 3. Ringkasan dataset ---------------------------------------------------
n_rows, n_cols = df.shape
class_counts = df[TARGET_COL].value_counts().to_dict()          # {0: n, 1: n}
class_pct = (df[TARGET_COL].value_counts(normalize=True) * 100).to_dict()
missing_count = df.isnull().sum().to_dict()
missing_pct = (df.isnull().mean() * 100).to_dict()

summary = {
    "n_rows": n_rows,
    "n_cols": n_cols,
    "class_counts": class_counts,
    "class_pct": class_pct,
    "missing_count": missing_count,
    "missing_pct": missing_pct,
}
summary_df = pd.DataFrame(
    [{"kolom": k, "count": v} for k, v in class_counts.items()]
)
summary_df["persen"] = [class_pct[k] for k in summary_df["kolom"]]
summary_df.to_csv(os.path.join(OUT_DIR, "class_balance.csv"), index=False)

missing_df = pd.DataFrame(
    {"kolom": list(missing_count.keys()),
     "missing_count": list(missing_count.values()),
     "missing_pct": [round(v, 2) for v in missing_pct.values()]}
)
missing_df.to_csv(os.path.join(OUT_DIR, "missing_stats.csv"), index=False)

# --- 4. Korelasi ------------------------------------------------------------
# Korelasi Pearson antar semua kolom numerik (termasuk target).
corr_matrix = df.corr(numeric_only=True)
corr_matrix.to_csv(os.path.join(OUT_DIR, "correlation_matrix.csv"))

# Ekstrak saja kolom fitur vs target, diurutkan by |value| descending.
feature_cols = [c for c in df.columns if c != TARGET_COL]
corr_target = corr_matrix[TARGET_COL].drop(TARGET_COL)
corr_target = corr_target.reindex(corr_target.abs().sort_values(ascending=False).index)
corr_target_df = corr_target.reset_index()
corr_target_df.columns = ["fitur", "korelasi_dengan_potability"]
corr_target_df.to_csv(os.path.join(OUT_DIR, "corr_with_target.csv"), index=False)

# --- 5. Statistik deskriptif -------------------------------------------------
desc = df[feature_cols].describe().T
desc = desc[["mean", "std", "min", "25%", "50%", "75%", "max"]]
desc.to_csv(os.path.join(OUT_DIR, "descriptive_stats.csv"))

# --- 6. Feature stats untuk UI (min/max/median) ------------------------------
feature_stats = {}
for col in feature_cols:
    feature_stats[col] = {
        "min": float(df[col].min()),
        "max": float(df[col].max()),
        "median": float(df[col].median()),
    }
with open(os.path.join(OUT_DIR, "feature_stats.json"), "w") as f:
    json.dump(feature_stats, f, indent=2)

# --- 7. Cetak ringkasan ke terminal ------------------------------------------
pct_str = ", ".join(f"{k}: {round(v, 1)}%" for k, v in class_pct.items())
print("=== DATASET SUMMARY ===")
print(f"Shape          : {n_rows} baris x {n_cols} kolom")
print(f"Class balance  : {class_counts}  -> {pct_str}")
print(f"Missing values : {sum(v for v in missing_count.values() if v > 0)} total (lihat missing_stats.csv)")
print("\n=== KORELASI FITUR DENGAN TARGET (Potability) ===")
print(corr_target_df.to_string(index=False))
print("\n=== FITUR Paling BERPENGARUH (|korelasi| tertinggi) ===")
top = corr_target_df.head(3)
print(top.to_string(index=False))
print("\nOutput ditulis ke:", OUT_DIR)