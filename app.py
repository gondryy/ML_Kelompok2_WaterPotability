import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from predict_utils import load_artifacts, load_eval_tables, predict_single, predict_batch

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

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
# Load Model Artifacts & Evaluation Tables
# ============================================
@st.cache_resource
def get_artifacts():
    return load_artifacts()

@st.cache_data
def get_eval_tables():
    return load_eval_tables()

pipeline, info = get_artifacts()
eval_tables = get_eval_tables()

FEATURES = info["feature_cols"]
STATS = info["feature_stats"]
DEFAULT_THRESHOLD = float(info["recommended_threshold"])
CLASS_NAMES = ["Not Potable", "Potable"]
COLOR_MAP = {"Potable": "#2ecc71", "Not Potable": "#e74c3c"}

# ============================================
# SHAP Helper (Handles Pipelines & Estimator Types)
# ============================================
def compute_binary_shap(trained_pipeline, X_row: pd.DataFrame, target_class_idx: int):
    if hasattr(trained_pipeline, "named_steps"):
        estimator = trained_pipeline[-1]
        X_transformed = trained_pipeline[:-1].transform(X_row)
    else:
        estimator = trained_pipeline
        X_transformed = X_row.values

    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer(X_transformed)
    except Exception:
        explainer = shap.Explainer(estimator, X_transformed)
        shap_values = explainer(X_transformed)

    if shap_values.values.ndim == 3:
        vals = shap_values.values[0, :, target_class_idx]
        base = shap_values.base_values[0, target_class_idx]
    else:
        sign = 1 if target_class_idx == 1 else -1
        vals = sign * shap_values.values[0]
        base = (
            shap_values.base_values[0]
            if target_class_idx == 1
            else -shap_values.base_values[0]
        )

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
            f"""
            Evaluate whether water is safe for consumption (**Potable** vs. **Not Potable**) using 9 physicochemical parameters (decision threshold: **{DEFAULT_THRESHOLD:.2f}**):

            - 🔹 **Single-Sample Prediction** with tuned threshold & range validation
            - 🔹 **SHAP Explainability** showing how each parameter pushes the prediction
            - 🔹 **Batch Comparison Mode** for evaluating uploaded CSV water samples
            - 🔹 **Simulated Live Sensor Stream** for continuous threshold monitoring
            """
        )

        if eval_tables.get("comparison") is not None:
            st.subheader("Model Evaluation Summary")
            st.dataframe(eval_tables["comparison"], width="stretch")

# ============================================
# 2. Prediction
# ============================================
elif page == "Prediction":
    with st.container(border=True):
        st.title("Water Potability Prediction")
        st.caption("Default values are initialized to training set medians.")

        def default_val(col):
            return float(STATS[col].get("median", (STATS[col]["min"] + STATS[col]["max"]) / 2))

        col1, col2, col3 = st.columns(3)
        with col1:
            ph = st.number_input("pH (0–14)", min_value=0.0, max_value=14.0, value=default_val("ph"))
            chloramines = st.number_input("Chloramines (ppm)", min_value=0.0, value=default_val("Chloramines"))
            organic_carbon = st.number_input("Organic Carbon (ppm)", min_value=0.0, value=default_val("Organic_carbon"))
        with col2:
            hardness = st.number_input("Hardness (mg/L)", min_value=0.0, value=default_val("Hardness"))
            sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0, value=default_val("Sulfate"))
            trihalomethanes = st.number_input("Trihalomethanes (µg/L)", min_value=0.0, value=default_val("Trihalomethanes"))
        with col3:
            solids = st.number_input("Solids / TDS (ppm)", min_value=0.0, value=default_val("Solids"))
            conductivity = st.number_input("Conductivity (µS/cm)", min_value=0.0, value=default_val("Conductivity"))
            turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, value=default_val("Turbidity"))

        threshold = st.slider(
            "Classification Threshold (Potable cutoff)",
            min_value=0.10,
            max_value=0.90,
            value=DEFAULT_THRESHOLD,
            step=0.01,
        )

        with st.expander("Parameter Reference Guide"):
            st.markdown(
                """
                - **pH**: Acid-base balance of water (WHO guideline: 6.5–8.5)
                - **Hardness**: Calcium & magnesium concentration (mg/L)
                - **Solids (TDS)**: Total dissolved minerals and salts (ppm)
                - **Chloramines**: Disinfectant levels in public water systems (ppm)
                - **Sulfate**: Naturally occurring minerals in groundwater (mg/L)
                - **Conductivity**: Electrical conductivity based on ionic concentration (µS/cm)
                - **Organic Carbon**: Total organic carbon from decaying organic matter (ppm)
                - **Trihalomethanes**: Disinfection byproducts formed with chlorine (µg/L)
                - **Turbidity**: Suspended particulate matter (NTU)
                """
            )

        if st.button("Predict Potability", type="primary", width="stretch"):
            raw_input = {
                "ph": ph,
                "Hardness": hardness,
                "Solids": solids,
                "Chloramines": chloramines,
                "Sulfate": sulfate,
                "Conductivity": conductivity,
                "Organic_carbon": organic_carbon,
                "Trihalomethanes": trihalomethanes,
                "Turbidity": turbidity,
            }

            result = predict_single(raw_input, pipeline, info, threshold=threshold)
            X_row = pd.DataFrame([raw_input])[FEATURES]

            pred_idx = result["potability"]
            pred_label = CLASS_NAMES[pred_idx]
            prob_pot = result["probability_potable"] * 100
            prob_non = result["probability_non_potable"] * 100

            st.session_state["last_input"] = X_row
            st.session_state["last_prediction"] = pred_label

            if result["out_of_range_fields"]:
                st.warning(
                    f"Out-of-training-range values detected for: **{', '.join(result['out_of_range_fields'])}**"
                )

            if pred_idx == 1:
                st.success(f"Prediction: **{result['label']}** (Potable prob: {prob_pot:.2f}% | Threshold: {threshold:.2f})")
            else:
                st.error(f"Prediction: **{result['label']}** (Potable prob: {prob_pot:.2f}% | Threshold: {threshold:.2f})")

            proba_df = pd.DataFrame(
                {"Class": ["Potable", "Not Potable"], "Probability (%)": [prob_pot, prob_non]}
            ).sort_values(by="Probability (%)", ascending=False)

            st.subheader("Prediction Probabilities")
            st.dataframe(proba_df.style.format({"Probability (%)": "{:.2f}"}), width="stretch")

            colors = [COLOR_MAP[c] for c in proba_df["Class"]]
            fig, ax = plt.subplots(figsize=(6, 2.5))
            bars = ax.barh(proba_df["Class"], proba_df["Probability (%)"], color=colors)
            ax.set_xlim(0, 112)
            ax.set_xlabel("Probability (%)")
            ax.set_title("Class Probability Breakdown")
            for bar in bars:
                w = bar.get_width()
                ax.annotate(
                    f"{w:.2f}%",
                    xy=(w, bar.get_y() + bar.get_height() / 2),
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

        if not SHAP_AVAILABLE:
            st.warning("`shap` is not installed in the current environment. Showing saved permutation importance instead.")
            if eval_tables.get("importance") is not None:
                st.dataframe(eval_tables["importance"], width="stretch")
        else:
            if "last_input" in st.session_state:
                X_row = st.session_state["last_input"]
                default_class = st.session_state.get("last_prediction", "Potable")
            else:
                X_row = pd.DataFrame(
                    [[STATS[c].get("median", (STATS[c]["min"] + STATS[c]["max"]) / 2) for c in FEATURES]],
                    columns=FEATURES,
                )
                default_class = "Potable"
                st.info("Showing explanation for baseline median sample. Run a prediction first to inspect your custom sample.")

            selected_class = st.selectbox(
                "Target class to explain",
                CLASS_NAMES,
                index=CLASS_NAMES.index(default_class),
            )
            class_idx = CLASS_NAMES.index(selected_class)

            try:
                shap_single = compute_binary_shap(pipeline, X_row, class_idx)

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
            except Exception as e:
                st.warning(f"Local SHAP plot unavailable for this estimator ({e}).")
                if eval_tables.get("importance") is not None:
                    st.subheader("Global Permutation Importance")
                    st.dataframe(eval_tables["importance"], width="stretch")

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
            try:
                result_df = predict_batch(compare_df, pipeline, info, threshold=DEFAULT_THRESHOLD)
                st.subheader("Batch Predictions")
                st.dataframe(result_df, width="stretch")

                st.subheader("Potable Probability (First 5 Samples)")
                for i in range(min(5, len(result_df))):
                    p_pot = result_df["Probabilitas_Layak"].iloc[i] * 100
                    p_non = 100.0 - p_pot
                    fig, ax = plt.subplots(figsize=(6, 2.2))
                    bars = ax.barh(
                        ["Potable", "Not Potable"],
                        [p_pot, p_non],
                        color=[COLOR_MAP["Potable"], COLOR_MAP["Not Potable"]],
                    )
                    ax.set_xlim(0, 112)
                    ax.set_xlabel("Probability (%)")
                    ax.set_title(f"Sample {i + 1} — {result_df['Status_Air'].iloc[i]}")
                    for bar in bars:
                        w = bar.get_width()
                        ax.text(w + 1.5, bar.get_y() + bar.get_height() / 2, f"{w:.2f}%", va="center")
                    st.pyplot(fig)
                    plt.close(fig)
            except ValueError as err:
                st.error(str(err))

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
                    col: float(np.random.uniform(STATS[col]["min"], STATS[col]["max"]))
                    for col in FEATURES
                }
                res = predict_single(sample, pipeline, info, threshold=DEFAULT_THRESHOLD)
                sample["Prediction"] = CLASS_NAMES[res["potability"]]

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