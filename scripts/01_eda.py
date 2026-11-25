"""01_eda.py — EDA (Exploratory Data Analysis) untuk dataset water potability.

Menghasilkan statistik deskriptif, pemeriksaan duplikat, analisis outlier (IQR),
korelasi, serta 5 visualisasi utama ke folder output/ untuk bahan laporan & PPT.

Cara pakai:
  python scripts/01_eda.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- 1. Tentukan path -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "water_potability.csv")
TARGET_COL = "Potability"

# --- 2. Muat dataset --------------------------------------------------------
df = pd.read_csv(DATA_PATH)
feature_cols = [c for c in df.columns if c != TARGET_COL]

# --- 3. Ringkasan dataset, duplikat & class balance -------------------------
n_rows, n_cols = df.shape
n_duplicates = int(df.duplicated().sum())
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
    "missing_pct": [round(v, 2) for v in missing_pct.values()],
})
missing_df.to_csv(os.path.join(OUT_DIR, "missing_stats.csv"), index=False)

# --- 4. Pemeriksaan Outlier (Metode IQR 1.5x) -------------------------------
outlier_rows = []
for col in feature_cols:
    s = df[col].dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n_out = int(((s < lower) | (s > upper)).sum())
    outlier_rows.append({
        "fitur": col,
        "Q1": round(q1, 4),
        "Q3": round(q3, 4),
        "IQR": round(iqr, 4),
        "batas_bawah": round(lower, 4),
        "batas_atas": round(upper, 4),
        "outlier_count": n_out,
        "outlier_pct": round(n_out / len(s) * 100, 2),
    })

outlier_df = pd.DataFrame(outlier_rows).sort_values(by="outlier_count", ascending=False)
outlier_df.to_csv(os.path.join(OUT_DIR, "outlier_stats.csv"), index=False)

# --- 5. Korelasi Pearson ----------------------------------------------------
corr_matrix = df.corr(numeric_only=True)
corr_matrix.to_csv(os.path.join(OUT_DIR, "correlation_matrix.csv"))

corr_target = corr_matrix[TARGET_COL].drop(TARGET_COL)
corr_target = corr_target.reindex(corr_target.abs().sort_values(ascending=False).index)
corr_target_df = corr_target.reset_index()
corr_target_df.columns = ["fitur", "korelasi_dengan_potability"]
corr_target_df.to_csv(os.path.join(OUT_DIR, "corr_with_target.csv"), index=False)

# --- 6. Statistik deskriptif & feature_stats.json ---------------------------
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

# --- 7. Visualisasi EDA (Disimpan ke output/) -------------------------------
sns.set_theme(style="whitegrid", font_scale=1.0)

# Chart 1: Distribusi Kelas Target
fig, ax = plt.subplots(figsize=(6.5, 4.2))
counts = df[TARGET_COL].value_counts().sort_index()
labels = ["0 - Tidak Layak\n(Non-Potable)", "1 - Layak Minum\n(Potable)"]
bars = ax.bar(labels, counts.values, color=["#e74c3c", "#2ecc71"], width=0.52, edgecolor="black", linewidth=0.7)
ax.set_title("Distribusi Kelas Target (Potability)", fontsize=12, fontweight="bold", pad=12)
ax.set_ylabel("Jumlah Sampel")
ax.set_ylim(0, max(counts.values) * 1.18)
for bar, c in zip(bars, counts.values):
    pct = c / len(df) * 100
    ax.text(bar.get_x() + bar.get_width() / 2, c + 35, f"{c:,}\n({pct:.2f}%)",
            ha="center", va="bottom", fontweight="bold", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_01_class_balance.png"), dpi=200)
plt.close(fig)

# Chart 2: Missing Values
missing_s = df.isnull().sum()
missing_s = missing_s[missing_s > 0].sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(7, 3.8))
bars = ax.barh(missing_s.index, missing_s.values, color="#f39c12", height=0.5, edgecolor="black", linewidth=0.7)
ax.set_title("Jumlah & Persentase Missing Values per Fitur", fontsize=12, fontweight="bold", pad=12)
ax.set_xlabel("Jumlah Nilai Hilang (dari 3.276 sampel)")
ax.set_xlim(0, max(missing_s.values) * 1.28)
for bar, c in zip(bars, missing_s.values):
    pct = c / len(df) * 100
    ax.text(c + 15, bar.get_y() + bar.get_height() / 2, f"{c} ({pct:.2f}%)",
            va="center", fontweight="bold", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_02_missing_values.png"), dpi=200)
plt.close(fig)

# Chart 3: Correlation Heatmap
fig, ax = plt.subplots(figsize=(9.5, 7.5))
sns.heatmap(
    corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
    vmin=-0.3, vmax=0.3, square=True, linewidths=0.5,
    cbar_kws={"shrink": 0.8}, ax=ax
)
ax.set_title("Matriks Korelasi Pearson Antarparameter & Target (Potability)", fontsize=12, fontweight="bold", pad=12)
plt.xticks(rotation=35, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_03_correlation_heatmap.png"), dpi=200)
plt.close(fig)

# Chart 4: Boxplot 9 Fitur Berdasarkan Kelas Target
fig, axes = plt.subplots(3, 3, figsize=(13, 10.5))
axes = axes.flatten()
for idx, col in enumerate(feature_cols):
    data_0 = df.loc[df[TARGET_COL] == 0, col].dropna()
    data_1 = df.loc[df[TARGET_COL] == 1, col].dropna()
    bp = axes[idx].boxplot(
        [data_0, data_1],
        labels=["0 (Tidak Layak)", "1 (Layak)"],
        patch_artist=True,
        widths=0.45,
        flierprops=dict(marker="o", markersize=2.5, alpha=0.5),
        medianprops=dict(color="black", linewidth=1.5),
    )
    bp["boxes"][0].set_facecolor("#e74c3c")
    bp["boxes"][0].set_alpha(0.8)
    bp["boxes"][1].set_facecolor("#2ecc71")
    bp["boxes"][1].set_alpha(0.8)
    axes[idx].set_title(col, fontsize=11, fontweight="bold")
    axes[idx].grid(axis="y", linestyle="--", alpha=0.5)

fig.suptitle("Distribusi & Outlier 9 Parameter Fisikokimia Berdasarkan Kelas Potability", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_04_feature_boxplots.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# Chart 5: Histogram + KDE 3 Fitur Paling Berpengaruh
top_features = ["Sulfate", "ph", "Hardness"]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for idx, col in enumerate(top_features):
    sns.histplot(
        data=df, x=col, hue=TARGET_COL, kde=True, stat="density", common_norm=False,
        palette={0: "#e74c3c", 1: "#2ecc71"}, alpha=0.45, ax=axes[idx]
    )
    axes[idx].set_title(f"Distribusi {col}", fontsize=11, fontweight="bold")
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel("Density")
    leg = axes[idx].get_legend()
    if leg:
        leg.set_title("Potability")
        for t, new_label in zip(leg.texts, ["0 (Tidak Layak)", "1 (Layak)"]):
            t.set_text(new_label)

fig.suptitle("Perbandingan Sebaran 3 Fitur Paling Berpengaruh (Sulfate, pH, Hardness)", fontsize=12.5, fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_05_top_features_hist.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# --- 8. Cetak ringkasan ke terminal ------------------------------------------
pct_str = ", ".join(f"{k}: {round(v, 1)}%" for k, v in class_pct.items())
print("=== DATASET SUMMARY ===")
print(f"Shape          : {n_rows} baris x {n_cols} kolom")
print(f"Duplicate rows : {n_duplicates} baris duplikat")
print(f"Class balance  : {class_counts}  -> {pct_str}")
print(f"Missing values : {sum(v for v in missing_count.values() if v > 0)} total (lihat missing_stats.csv)")
print("\n=== RINGKASAN OUTLIER (IQR 1.5x) ===")
print(outlier_df[["fitur", "outlier_count", "outlier_pct"]].to_string(index=False))
print("\n=== KORELASI FITUR DENGAN TARGET (Potability) ===")
print(corr_target_df.to_string(index=False))
print("\nOutput CSV, JSON, dan 5 grafik PNG berhasil disimpan ke:", OUT_DIR)