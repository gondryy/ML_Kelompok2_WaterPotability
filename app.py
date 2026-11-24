"""app.py — Aplikasi Web Prediksi Kelayakan Air Minum (Water Potability)."""

import numpy as np
import pandas as pd
import streamlit as st
from predict_utils import load_artifacts, load_eval_tables, predict_batch, predict_single

st.set_page_config(
    page_title="Prediksi Kelayakan Air Minum (ML)",
    page_icon="💧",
    layout="wide",
)

FEATURE_META = {
    "ph": {"label": "pH", "unit": "0–14", "who": "6.5 – 8.5", "optional": True},
    "Hardness": {"label": "Hardness (Kesadahan)", "unit": "mg/L", "who": "< 300 mg/L", "optional": False},
    "Solids": {"label": "Total Dissolved Solids (TDS)", "unit": "ppm", "who": "< 1000 ppm", "optional": False},
    "Chloramines": {"label": "Chloramines", "unit": "ppm", "who": "< 4 ppm", "optional": False},
    "Sulfate": {"label": "Sulfate (Sulfat)", "unit": "mg/L", "who": "< 250 mg/L", "optional": True},
    "Conductivity": {"label": "Conductivity (Konduktivitas)", "unit": "μS/cm", "who": "< 400 μS/cm", "optional": False},
    "Organic_carbon": {"label": "Organic Carbon (TOC)", "unit": "ppm", "who": "< 10 ppm", "optional": False},
    "Trihalomethanes": {"label": "Trihalomethanes (THM)", "unit": "μg/L", "who": "< 80 μg/L", "optional": True},
    "Turbidity": {"label": "Turbidity (Kekeruhan)", "unit": "NTU", "who": "< 5.0 NTU", "optional": False},
}

PRESETS = {
    "Nilai Median Dataset": None,
    "Contoh Layak Minum (Verified Test Sample)": {
        "ph": 9.04, "Hardness": 227.36, "Solids": 34353.84, "Chloramines": 5.57,
        "Sulfate": 231.05, "Conductivity": 421.66, "Organic_carbon": 16.33,
        "Trihalomethanes": 92.64, "Turbidity": 3.71,
    },
    "Contoh Tidak Layak + Missing pH": {
        "ph": None, "Hardness": 176.39, "Solids": 26003.16, "Chloramines": 7.81,
        "Sulfate": 358.43, "Conductivity": 336.14, "Organic_carbon": 14.45,
        "Trihalomethanes": 90.22, "Turbidity": 4.41,
    },
}


@st.cache_resource
def get_model():
    return load_artifacts()


pipeline, model_info = get_model()
eval_tables = load_eval_tables()
feature_cols = model_info["feature_cols"]
feature_stats = model_info["feature_stats"]
rec_th = float(model_info["recommended_threshold"])

# --- SIDEBAR ---
with st.sidebar:
    st.header("Konfigurasi Model")
    st.markdown(f"**Algoritma Terpilih:** `{model_info['best_model_name']}`")
    st.markdown(f"**Threshold Rekomendasi (OOF):** `{rec_th:.2f}`")

    active_threshold = st.slider(
        "Ambang Batas Keputusan (Threshold)",
        min_value=0.30,
        max_value=0.80,
        value=rec_th,
        step=0.05,
        help="Threshold lebih tinggi (mis. 0.60) memperketat syarat 'Layak Minum' untuk menekan False Positive (air berbahaya diprediksi aman).",
    )

    if active_threshold >= rec_th:
        st.success(f"Mode Keamanan Tinggi aktif (Threshold ≥ {rec_th:.2f}).")
    else:
        st.warning(f"Threshold di bawah rekomendasi ({rec_th:.2f}): risiko False Positive meningkat.")

    st.divider()
    st.subheader("Preset Uji Cepat")
    selected_preset = st.selectbox("Pilih contoh data sampel:", list(PRESETS.keys()))

# --- HEADER ---
st.title("💧 Sistem Prediksi Kelayakan Air Minum (Water Potability)")
st.caption(
    "Implementasi Machine Learning End-to-End (TEK1302) — Pipeline bebas data leakage dengan optimasi threshold berbasis keselamatan."
)

tab1, tab2, tab3 = st.tabs([
    "1. Prediksi Sampel Tunggal",
    "2. Prediksi Batch (Upload CSV)",
    "3. Evaluasi & Analisis Model",
])

# === TAB 1: SINGLE SAMPLE PREDICTION ===
with tab1:
    st.subheader("Input Parameter Fisikokimia Air")
    st.write("Masukkan 9 parameter kualitas air. Parameter dengan tanda *(opsional)* dapat dikosongkan untuk diimputasi otomatis menggunakan median data latih.")

    preset_vals = PRESETS[selected_preset]
    user_inputs = {}

    cols_ui = st.columns(3)
    for idx, col in enumerate(feature_cols):
        meta = FEATURE_META[col]
        stats = feature_stats[col]
        default_val = stats["median"] if preset_vals is None else preset_vals[col]

        with cols_ui[idx % 3]:
            st.markdown(f"**{meta['label']}** ({meta['unit']})")
            if meta["optional"]:
                is_missing = st.checkbox(
                    f"Kosongkan nilai {col} (Missing)",
                    value=(default_val is None),
                    key=f"miss_{col}_{selected_preset}",
                )
            else:
                is_missing = False

            if is_missing:
                user_inputs[col] = None
                st.caption(f"Akan diisi median latih: `{stats['median']:.2f}`")
            else:
                init_num = float(stats["median"] if default_val is None else default_val)
                val = st.number_input(
                    label=f"Input {col}",
                    value=round(init_num, 4),
                    step=0.1,
                    format="%.4f",
                    label_visibility="collapsed",
                    key=f"num_{col}_{selected_preset}",
                )
                user_inputs[col] = val
                st.caption(f"Rentang latih: `{stats['min']:.1f}` – `{stats['max']:.1f}` | Acuan WHO: {meta['who']}")

    if st.button("🔍 Prediksi Kelayakan Air", type="primary", use_container_width=True):
        res = predict_single(user_inputs, pipeline, model_info, threshold=active_threshold)

        st.divider()
        res_col1, res_col2 = st.columns([1.3, 1])

        with res_col1:
            if res["potability"] == 1:
                st.success(f"### ✅ HASIL: {res['label']}")
                st.write(
                    f"Probabilitas layak minum (**{res['probability_potable']*100:.2f}%**) memenuhi ambang batas keamanan (**{res['threshold_used']*100:.0f}%**)."
                )
            else:
                st.error(f"### ⚠️ HASIL: {res['label']}")
                st.write(
                    f"Probabilitas layak minum (**{res['probability_potable']*100:.2f}%**) berada di bawah ambang batas keamanan (**{res['threshold_used']*100:.0f}%**)."
                )

            if res["imputed_fields"]:
                st.info(
                    f"ℹ️ **Imputasi Otomatis:** Fitur `{', '.join(res['imputed_fields'])}` kosong dan diisi otomatis menggunakan median data latih oleh `SimpleImputer`."
                )
            if res["out_of_range_fields"]:
                st.warning(
                    f"⚠️ **Peringatan Ekstrapolasi:** Fitur `{', '.join(res['out_of_range_fields'])}` berada di luar rentang minimum–maksimum data latih."
                )

        with res_col2:
            st.metric("Probabilitas Layak Minum (Kelas 1)", f"{res['probability_potable']*100:.2f}%")
            st.progress(min(max(res["probability_potable"], 0.0), 1.0))
            st.metric("Ambang Batas Aktif (Threshold)", f"{res['threshold_used']*100:.0f}%")

# === TAB 2: BATCH CSV PREDICTION ===
with tab2:
    st.subheader("Prediksi Massal dari File CSV")
    st.write(f"File CSV harus memiliki 9 kolom fitur berikut: `{', '.join(feature_cols)}` (kolom tambahan seperti `Potability` akan dipertahankan).")

    template_df = pd.DataFrame([{c: round(feature_stats[c]["median"], 4) for c in feature_cols}])
    st.download_button(
        label="⬇️ Unduh Template CSV Contoh",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="template_sampel_air.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Unggah file CSV sampel air", type=["csv"])
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            pred_df = predict_batch(raw_df, pipeline, model_info, threshold=active_threshold)

            total_rows = len(pred_df)
            potable_count = int((pred_df["Prediksi_Potability"] == 1).sum())
            non_potable_count = total_rows - potable_count

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Sampel Diproses", total_rows)
            m2.metric("Diprediksi Layak Minum", f"{potable_count} ({potable_count/total_rows*100:.1f}%)")
            m3.metric("Diprediksi Tidak Layak", f"{non_potable_count} ({non_potable_count/total_rows*100:.1f}%)")

            st.dataframe(pred_df, use_container_width=True)

            st.download_button(
                label="⬇️ Unduh Hasil Prediksi Lengkap (.csv)",
                data=pred_df.to_csv(index=False).encode("utf-8"),
                file_name="hasil_prediksi_water_potability.csv",
                mime="text/csv",
                type="primary",
            )
        except Exception as e:
            st.error(f"Gagal memproses file CSV: {e}")

# === TAB 3: MODEL EVALUATION & ANALYSIS ===
with tab3:
    st.subheader("Ringkasan Evaluasi Model & Analisis Fitur")

    test_metrics = model_info.get("final_test_metrics_at_recommended_th", {})
    if test_metrics:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Model Terbaik", model_info["best_model_name"])
        c2.metric("Test ROC-AUC", f"{test_metrics.get('ROC_AUC', 0):.4f}")
        c3.metric("Test Precision", f"{test_metrics.get('Precision', 0):.4f}")
        c4.metric("Test Specificity", f"{test_metrics.get('Specificity', 0):.4f}")
        c5.metric("False Positives (Test)", test_metrics.get("FP", "-"))

    if eval_tables["comparison"] is not None:
        st.markdown("#### 1. Perbandingan 5 Algoritma (5-Fold Stratified CV & Test Set)")
        st.dataframe(eval_tables["comparison"], use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        if eval_tables["threshold"] is not None:
            st.markdown("#### 2. Trade-off Threshold Tuning (Out-of-Fold)")
            th_df = eval_tables["threshold"].set_index("Threshold")
            st.line_chart(th_df[["OOF_Precision", "OOF_Recall", "OOF_Accuracy"]])
            st.dataframe(eval_tables["threshold"], use_container_width=True)

    with col_right:
        if eval_tables["importance"] is not None:
            st.markdown("#### 3. Permutation Feature Importance (Test Set)")
            imp_df = eval_tables["importance"].set_index("fitur")
            st.bar_chart(imp_df["importance_mean"])
            st.dataframe(eval_tables["importance"], use_container_width=True)