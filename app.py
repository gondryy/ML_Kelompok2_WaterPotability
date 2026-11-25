import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ============================================
# Page Config & CSS
# ============================================
st.set_page_config(
    page_title="Water Potability Classification",
    page_icon="💧",
    layout="centered",
)

def load_css(file_name: str):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css("style.css")

# ============================================
# Constants (Match Your Training Schema)
# ============================================
DATA_FILE = "data/water_potability.csv"
MODEL_FILE = "model/water_potability_model.joblib"

FEATURES = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]

CLASS_NAMES = ["Not Potable", "Potable"]  # 0 = Not Potable, 1 = Potable
COLOR_MAP = {"Potable": "#2ecc71", "Not Potable": "#e74c3c"}

# ============================================
# Cached Loaders
# ============================================
@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

df = load_data()
model = load_model()

# ============================================
# SHAP Helper (Handles Both Pipelines & Raw Tree Models)
# ============================================
def compute_binary_shap(trained_model, X_row: pd.DataFrame, target_class_idx: int):
    if hasattr(trained_model, "named_steps"):
        estimator = trained_model[-1]
        X_transformed = trained_model[:-1].transform(X_row)
    else:
        estimator = trained_model
        X_transformed = X_row.values

    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer(X_transformed)

    # 3D output: (samples, features, classes) e.g., RandomForest
    if shap_values.values.ndim == 3:
        vals = shap_values.values[0, :, target_class_idx]
        base = shap_values.base_values[0, target_class_idx]
    # 2D output: (samples, features) for class 1 e.g., XGBoost / LightGBM
    else:
        sign = 1 if target_class_idx == 1 else -1
        vals = sign * shap_values.values[0]
        base = shap_values.base_values[0] if target_class_idx == 1 else -shap_values.base_values[0]

    return shap.Explanation(
        values=vals,
        base_values=base,
        data=X_row.values[0],
        feature_names=FEATURES,
    )

# ============================================
# Sidebar Navigation
# ============================================
st.sidebar.title("💧 Water Potability")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Prediction", "Model Explanation", "Comparison Mode", "Live Monitoring"],
)

# ============================================
# 1. Home
# ============================================
if page == "Home":
    with st.container(border=True):
        st.title("💧 Water Potability Prediction Dashboard")
        st.info(
            """
            Evaluate whether water is safe for human consumption (**Potable** vs. **Not Potable**) using 9 physicochemical parameters:

            - 🔹 **Single-Sample Prediction** with class probability breakdown
            - 🔹 **SHAP Explainability** showing how each parameter pushes the prediction
            - 🔹 **Batch Comparison Mode** for evaluating uploaded CSV water samples
            - 🔹 **Simulated Live Sensor Stream** for continuous threshold monitoring
            """
        )

# ============================================
# 2. Prediction
# ============================================
elif page == "Prediction":
    with st.container(border=True):
        st.title("Water Potability Prediction")
        st.caption("Default values are initialized to dataset medians.")

        col1, col2, col3 = st.columns(3)
        with col1:
            ph = st.number_input("pH (0–14)", min_value=0.0, max_value=14.0, value=float(df["ph"].median()))
            chloramines = st.number_input("Chloramines (ppm)", min_value=0.0, value=float(df["Chloramines"].median()))
            organic_carbon = st.number_input("Organic Carbon (ppm)", min_value=0.0, value=float(df["Organic_carbon"].median()))
        with col2:
            hardness = st.number_input("Hardness (mg/L)", min_value=0.0, value=float(df["Hardness"].median()))
            sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0, value=float(df["Sulfate"].median()))
            trihalomethanes = st.number_input("Trihalomethanes (µg/L)", min_value=0.0, value=float(df["Trihalomethanes"].median()))
        with col3:
            solids = st.number_input("Solids / TDS (ppm)", min_value=0.0, value=float(df["Solids"].median()))
            conductivity = st.number_input("Conductivity (µS/cm)", min_value=0.0, value=float(df["Conductivity"].median()))
            turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, value=float(df["Turbidity"].median()))

        with st.expander("Parameter Reference Guide"):
            st.markdown(
                """
                - **pH**: Acid-base balance of water (WHO guideline: 6.5–8.5)
                - **Hardness**: Capacity of water to precipitate soap, mainly calcium & magnesium (mg/L)
                - **Solids (TDS)**: Total dissolved minerals and salts (ppm)
                - **Chloramines**: Disinfectant levels used in public water systems (ppm)
                - **Sulfate**: Naturally occurring minerals in groundwater (mg/L)
                - **Conductivity**: Electrical conductivity based on ionic concentration (µS/cm)
                - **Organic Carbon**: Total organic carbon from decaying natural matter (ppm)
                - **Trihalomethanes**: Disinfection byproducts formed with chlorine (µg/L)
                - **Turbidity**: Light-emitting property measuring suspended matter (NTU)
                """
            )

        if st.button("Predict Potability", type="primary", use_container_width=True):
            X_row = pd.DataFrame(
                [[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]],
                columns=FEATURES,
            )

            pred_idx = int(model.predict(X_row)[0])
            probas = model.predict_proba(X_row)[0]
            pred_label = CLASS_NAMES[pred_idx]

            st.session_state["last_input"] = X_row
            st.session_state["last_prediction"] = pred_label

            if pred_idx == 1:
                st.success(f"Prediction: **{pred_label}** ({probas[1]*100:.2f}% confidence)")
            else:
                st.error(f"Prediction: **{pred_label}** ({probas[0]*100:.2f}% confidence)")

            proba_df = pd.DataFrame(
                {"Class": CLASS_NAMES, "Confidence (%)": probas * 100}
            ).sort_values(by="Confidence (%)", ascending=False)

            st.subheader("Prediction Confidence")
            st.dataframe(proba_df.style.format({"Confidence (%)": "{:.2f}"}), use_container_width=True)

            colors = [COLOR_MAP[c] for c in proba_df["Class"]]
            fig, ax = plt.subplots(figsize=(6, 2.5))
            bars = ax.barh(proba_df["Class"], proba_df["Confidence (%)"], color=colors)
            ax.set_xlim(0, 112)
            ax.set_xlabel("Confidence (%)")
            ax.set_title("Class Probability Breakdown")
            for bar in bars:
                width = bar.get_width()
                ax.annotate(
                    f"{width:.2f}%",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                )
            st.pyplot(fig)
            plt.close(fig)

# ============================================
# 3. Model Explanation (SHAP)
# ============================================
elif page == "Model Explanation":
    with st.container(border=True):
        st.title("📖 Model Explanation (SHAP)")

        if "last_input" in st.session_state:
            X_row = st.session_state["last_input"]
            default_class = st.session_state.get("last_prediction", "Potable")
        else:
            X_row = pd.DataFrame([[df[c].median() for c in FEATURES]], columns=FEATURES)
            default_class = "Potable"
            st.info("Showing explanation for median baseline sample. Run a prediction first to inspect your custom sample.")

        selected_class = st.selectbox(
            "Target class to explain",
            CLASS_NAMES,
            index=CLASS_NAMES.index(default_class),
        )
        class_idx = CLASS_NAMES.index(selected_class)

        shap_single = compute_binary_shap(model, X_row, class_idx)

        st.subheader(f"1. Waterfall Plot ({selected_class})")
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_single, show=False)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader(f"2. Feature Contribution Bar ({selected_class})")
        fig, ax = plt.subplots()
        shap.plots.bar(shap_single, show=False)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown(
            f"""
            **How to read these plots for _{selected_class}_:**
            - **Red / positive bars** push the prediction **toward** `{selected_class}`.
            - **Blue / negative bars** push the prediction **away** from `{selected_class}`.
            """
        )

# ============================================
# 4. Comparison Mode (Batch CSV)
# ============================================
elif page == "Comparison Mode":
    with st.container(border=True):
        st.title("📊 Batch Sample Comparison")
        st.write(f"Upload a CSV containing columns: **{', '.join(FEATURES)}**")

        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            compare_df = pd.read_csv(uploaded_file)
            missing = [c for c in FEATURES if c not in compare_df.columns]

            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                preds = model.predict(compare_df[FEATURES])
                probas = model.predict_proba(compare_df[FEATURES])

                result_df = compare_df.copy()
                result_df["Prediction"] = [CLASS_NAMES[int(p)] for p in preds]
                result_df["Potable Prob (%)"] = (probas[:, 1] * 100).round(2)
                result_df["Top Confidence (%)"] = (probas.max(axis=1) * 100).round(2)

                st.subheader("Batch Predictions")
                st.dataframe(result_df, use_container_width=True)

                st.subheader("Confidence Breakdown (First 5 Samples)")
                for i in range(min(5, len(result_df))):
                    fig, ax = plt.subplots(figsize=(6, 2.2))
                    p = probas[i] * 100
                    colors = [COLOR_MAP[c] for c in CLASS_NAMES]
                    bars = ax.barh(CLASS_NAMES, p, color=colors)
                    ax.set_xlim(0, 112)
                    ax.set_xlabel("Confidence (%)")
                    ax.set_title(f"Sample {i + 1} — {result_df['Prediction'].iloc[i]}")
                    for bar in bars:
                        w = bar.get_width()
                        ax.text(w + 1.5, bar.get_y() + bar.get_height() / 2, f"{w:.2f}%", va="center")
                    st.pyplot(fig)
                    plt.close(fig)

# ============================================
# 5. Live Monitoring (Non-Blocking Fragment)
# ============================================
elif page == "Live Monitoring":
    with st.container(border=True):
        st.title("📡 Live Sensor Monitoring (Simulated)")
        streaming = st.toggle("Enable live stream (2s interval)", value=True)

        if "live_data" not in st.session_state:
            st.session_state["live_data"] = pd.DataFrame(columns=FEATURES + ["Prediction"])

        @st.fragment(run_every=2 if streaming else None)
        def render_live_stream():
            if streaming:
                sample = {
                    col: float(np.random.uniform(df[col].quantile(0.05), df[col].quantile(0.95)))
                    for col in FEATURES
                }
                X_live = pd.DataFrame([sample], columns=FEATURES)
                pred_idx = int(model.predict(X_live)[0])
                sample["Prediction"] = CLASS_NAMES[pred_idx]

                st.session_state["live_data"] = pd.concat(
                    [st.session_state["live_data"], pd.DataFrame([sample])],
                    ignore_index=True,
                ).tail(20)

            live_df = st.session_state["live_data"]
            if not live_df.empty:
                latest = live_df.iloc[-1]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Status", latest["Prediction"])
                m2.metric("pH", f"{latest['ph']:.2f}")
                m3.metric("Solids (ppm)", f"{latest['Solids']:.0f}")
                m4.metric("Turbidity (NTU)", f"{latest['Turbidity']:.2f}")

                st.subheader("Normalized Sensor Trends (Last 20 Readings)")
                numeric_df = live_df[FEATURES]
                norm_df = (numeric_df - numeric_df.min()) / (numeric_df.max() - numeric_df.min() + 1e-9)
                st.line_chart(norm_df)

                st.subheader("Recent Classification Counts")
                st.bar_chart(live_df["Prediction"].value_counts())

        render_live_stream()