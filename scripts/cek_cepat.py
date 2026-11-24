import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE_DIR, "data", "water_potability.csv"))

print("=== 1. Ukuran data kalau baris kosong dihapus ===")
print("Sebelum:", df.shape)
print("Sesudah:", df.dropna().shape)

print("\n=== 2. Kemiringan (skewness) tiap fitur ===")
print(df.skew(numeric_only=True))

print("\n=== 3. Proporsi nilai kosong per kelas ===")
print(df.isnull().groupby(df["Potability"]).mean())