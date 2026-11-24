"""03_train_evaluate.py — Training 5 Model (5-Fold CV), OOF Threshold Tuning, & Export Pipeline.

Membaca hasil split & blueprint dari 02_preprocess.py.
"""

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

# 1. Muat data split & blueprint preprocessor dari 02_preprocess.py
train_df = pd.read_csv(os.path.join(OUT_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(OUT_DIR, "test.csv"))
preprocessor_blueprint = joblib.load(os.path.join(MODEL_DIR, "preprocessor_blueprint.joblib"))

with open(os.path.join(OUT_DIR, "feature_cols.json")) as f:
    feature_cols = json.load(f)

X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
X_test, y_test = test_df[feature_cols], test_df[TARGET_COL]

# 2. Definisi 5 Algoritma Klasifikasi
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

# 3. Evaluasi 5-Fold Stratified CV pada Data Train (Bebas Leakage)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_results = []
oof_proba = {}

for name, model in models.items():
    fold_aucs = []
    proba = np.zeros(len(y_train))

    for tr_idx, va_idx in cv.split(X_train, y_train):
        # Preprocessor di-fit ulang di dalam tiap fold
        pipe = make_pipeline(clone(preprocessor_blueprint), clone(model))
        pipe.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        p_val = pipe.predict_proba(X_train.iloc[va_idx])[:, 1]
        proba[va_idx] = p_val
        fold_aucs.append(roc_auc_score(y_train.iloc[va_idx], p_val))

    oof_proba[name] = proba

    # Evaluasi Train vs Test (threshold 0.5) untuk analisis overfitting/underfitting
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

# 4. Threshold Tuning via Out-of-Fold (OOF) Predictions (Zero Test Leakage)
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

# 5. Final Fit pada Full X_train & Evaluasi Akhir pada X_test
final_pipeline = make_pipeline(clone(preprocessor_blueprint), clone(models[best_model_name]))
final_pipeline.fit(X_train, y_train)

proba_test_final = final_pipeline.predict_proba(X_test)[:, 1]
y_pred_test_final = (proba_test_final >= opt_th).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test_final).ravel()

final_test_metrics = {
    "Threshold": opt_th,
    "ROC_AUC": round(roc_auc_score(y_test, proba_test_final), 4),
    "Accuracy": round(accuracy_score(y_test, y_pred_test_final), 4),
    "Precision": round(precision_score(y_test, y_pred_test_final, zero_division=0), 4),
    "Recall": round(recall_score(y_test, y_pred_test_final, zero_division=0), 4),
    "Specificity": round(recall_score(y_test, y_pred_test_final, pos_label=0, zero_division=0), 4),
    "F1_Score": round(f1_score(y_test, y_pred_test_final, zero_division=0), 4),
    "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
}

# 6. Permutation Importance pada Data Test
perm_imp = permutation_importance(
    final_pipeline, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=RANDOM_STATE
)
imp_df = pd.DataFrame({
    "fitur": feature_cols,
    "importance_mean": perm_imp.importances_mean,
    "importance_std": perm_imp.importances_std,
}).sort_values(by="importance_mean", ascending=False)
imp_df.to_csv(os.path.join(OUT_DIR, "permutation_importance.csv"), index=False)

# 7. Simpan Pipeline Akhir & Metadata untuk Aplikasi
joblib.dump(final_pipeline, os.path.join(MODEL_DIR, "model_pipeline.joblib"))

with open(os.path.join(OUT_DIR, "feature_stats.json")) as f:
    feature_stats = json.load(f)

model_info = {
    "best_model_name": best_model_name,
    "default_threshold": 0.5,
    "recommended_threshold": opt_th,
    "selection_criterion": "5-Fold Stratified CV ROC-AUC + OOF FP Reduction >= 40%",
    "final_test_metrics_at_recommended_th": final_test_metrics,
    "feature_cols": feature_cols,
    "feature_stats": feature_stats,
}
with open(os.path.join(MODEL_DIR, "model_info.json"), "w") as f:
    json.dump(model_info, f, indent=2)

print(f"Model Terbaik (CV): {best_model_name} | Threshold Rekomendasi (OOF): {opt_th}")
print("Evaluasi Akhir Data Test:", final_test_metrics)