"""predict_utils.py — Helper inferensi & validasi input untuk aplikasi Water Potability."""

import json
import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
OUT_DIR = os.path.join(BASE_DIR, "output")


def load_artifacts():
    pipeline = joblib.load(os.path.join(MODEL_DIR, "model_pipeline.joblib"))
    with open(os.path.join(MODEL_DIR, "model_info.json")) as f:
        info = json.load(f)
    return pipeline, info


def load_eval_tables():
    tables = {}
    for key, filename in [
        ("comparison", "model_comparison.csv"),
        ("threshold", "threshold_tuning.csv"),
        ("importance", "permutation_importance.csv"),
    ]:
        path = os.path.join(OUT_DIR, filename)
        tables[key] = pd.read_csv(path) if os.path.exists(path) else None
    return tables


def predict_single(raw_input: dict, pipeline, info: dict, threshold: float = None) -> dict:
    feature_cols = info["feature_cols"]
    stats = info["feature_stats"]
    th = float(threshold if threshold is not None else info["recommended_threshold"])

    clean_row = {}
    imputed_fields = []
    out_of_range_fields = []

    for col in feature_cols:
        val = raw_input.get(col, None)
        if val is None or (isinstance(val, float) and np.isnan(val)) or str(val).strip() == "":
            clean_row[col] = np.nan
            imputed_fields.append(col)
        else:
            num_val = float(val)
            clean_row[col] = num_val
            if num_val < stats[col]["min"] or num_val > stats[col]["max"]:
                out_of_range_fields.append(col)

    df_input = pd.DataFrame([clean_row])[feature_cols]
    prob_potable = float(pipeline.predict_proba(df_input)[0, 1])
    pred_class = int(prob_potable >= th)

    return {
        "potability": pred_class,
        "label": "Layak Minum (Potable)" if pred_class == 1 else "Tidak Layak Minum (Non-Potable)",
        "probability_potable": round(prob_potable, 4),
        "probability_non_potable": round(1.0 - prob_potable, 4),
        "threshold_used": th,
        "imputed_fields": imputed_fields,
        "out_of_range_fields": out_of_range_fields,
    }


def predict_batch(df_raw: pd.DataFrame, pipeline, info: dict, threshold: float = None) -> pd.DataFrame:
    feature_cols = info["feature_cols"]
    th = float(threshold if threshold is not None else info["recommended_threshold"])

    missing_cols = [c for c in feature_cols if c not in df_raw.columns]
    if missing_cols:
        raise ValueError(f"Kolom wajib tidak ditemukan di CSV: {', '.join(missing_cols)}")

    X_batch = df_raw[feature_cols].copy()
    probs = pipeline.predict_proba(X_batch)[:, 1]
    preds = (probs >= th).astype(int)

    result_df = df_raw.copy()
    result_df["Probabilitas_Layak"] = np.round(probs, 4)
    result_df["Prediksi_Potability"] = preds
    result_df["Status_Air"] = np.where(preds == 1, "Layak Minum", "Tidak Layak Minum")
    return result_df