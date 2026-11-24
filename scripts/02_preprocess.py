"""02_preprocess.py — Pembagian Data (80/20) & Definisi Pipeline Preprocessing.

Output:
  - output/train.csv & output/test.csv (data hasil stratified split 80/20)
  - output/feature_cols.json (urutan kolom fitur standar)
  - model/preprocessor_blueprint.joblib (pipeline imputer + scaler untuk CV tanpa leakage)
"""

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

# 1. Muat dataset & pisahkan fitur vs target
df = pd.read_csv(os.path.join(DATA_DIR, "water_potability.csv"))
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]
feature_cols = list(X.columns)

# 2. Stratified Train/Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

train_df = X_train.copy()
train_df[TARGET_COL] = y_train
test_df = X_test.copy()
test_df[TARGET_COL] = y_test

# 3. Definisi Blueprint Preprocessing (Median Imputation + Standard Scaling)
preprocessor_blueprint = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), feature_cols),
    ],
    remainder="drop",
)

# 4. Simpan hasil split & blueprint preprocessor
train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

with open(os.path.join(OUT_DIR, "feature_cols.json"), "w") as f:
    json.dump(feature_cols, f, indent=2)

joblib.dump(preprocessor_blueprint, os.path.join(MODEL_DIR, "preprocessor_blueprint.joblib"))

print(f"Split selesai: Train = {len(train_df)} baris | Test = {len(test_df)} baris")
print("Blueprint preprocessor tersimpan di model/preprocessor_blueprint.joblib")