import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ----------------------------
# PAGE SETUP
# ----------------------------
st.set_page_config(page_title="Wilson’s Disease Predictor", layout="wide")
st.title("🧬 Wilson’s Disease Prediction with Explainability")

st.markdown("""
This web app uses a trained *Random Forest model* to predict *Wilson’s Disease*
based on clinical features and provides *SHAP explainability* — showing why the model made this prediction.
""")
st.divider()

# ----------------------------
# LOAD TRAINED MODEL
# ----------------------------
try:
    model = joblib.load("model.pkl")
    st.success("✅ Model loaded successfully.")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ----------------------------
# USER INPUT SECTION
# ----------------------------
st.subheader("🧾 Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age (years)", value=0.0)
    Sex = st.selectbox("Sex", ["Male", "Female"])
    Ceruloplasmin = st.number_input("Ceruloplasmin Level (mg/dL)", value=0.0)
    Copper_Blood = st.number_input("Copper in Blood Serum (µg/dL)", value=0.0)
    Free_Copper = st.number_input("Free Copper in Blood Serum (µg/dL)", value=0.0)
    Copper_Urine = st.number_input("Copper in Urine (µg/day)", value=0.0)
    ALT = st.number_input("ALT (U/L)", value=0.0)
    AST = st.number_input("AST (U/L)", value=0.0)
    Total_Bilirubin = st.number_input("Total Bilirubin (mg/dL)", value=0.0)

with col2:
    Albumin = st.number_input("Albumin (g/dL)", value=0.0)
    ALP = st.number_input("Alkaline Phosphatase (U/L)", value=0.0)
    INR = st.number_input("Prothrombin Time / INR", value=0.0)
    GGT = st.number_input("Gamma-Glutamyl Transferase (GGT) (U/L)", value=0.0)
    KF_Rings = st.selectbox("Kayser–Fleischer Rings", ["Present", "Absent"])
    Neuro_Score = st.number_input("Neurological Syndrome Score", value=0.0)
    Psychiatric_Symptoms = st.selectbox("Psychiatric Symptoms", ["Yes", "No"])
    Family_History = st.selectbox("Family History of Wilson’s Disease", ["Yes", "No"])
    ATP7B = st.selectbox("ATP7B Gene Mutation", ["Yes", "No"])

# ----------------------------
# Convert Inputs into Model Format
# ----------------------------
input_df = pd.DataFrame({
    "Age": [Age],
    "Sex": [1 if Sex == "Male" else 0],
    "Ceruloplasmin Level": [Ceruloplasmin],
    "Copper in Blood Serum": [Copper_Blood],
    "Free Copper in Blood Serum": [Free_Copper],
    "Copper in Urine": [Copper_Urine],
    "ALT": [ALT],
    "AST": [AST],
    "Total Bilirubin": [Total_Bilirubin],
    "Albumin": [Albumin],
    "Alkaline Phosphatase (ALP)": [ALP],
    "Prothrombin Time / INR": [INR],
    "Gamma-Glutamyl Transferase (GGT)": [GGT],
    "Kaiser-Fleischer Rings": [1 if KF_Rings == "Present" else 0],
    "Neurological Syndrome Score": [Neuro_Score],
    "Psychiatric Symptoms": [1 if Psychiatric_Symptoms == "Yes" else 0],
    "Family History": [1 if Family_History == "Yes" else 0],
    "ATP7B Gene Mutation": [1 if ATP7B == "Yes" else 0]
})

# ----------------------------
# PREDICTION SECTION
# ----------------------------
if st.button("🔍 Predict Wilson’s Disease"):

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"🧬 *Prediction:* Wilson’s Disease likely detected (Confidence ≈ {probability:.2f})")
        else:
            st.success(f"✅ *Prediction:* No Wilson’s Disease detected (Confidence ≈ {probability:.2f})")
# ----------------------------
# SHAP Explanation (robust)
# ----------------------------
st.subheader("🔍 Explanation of Prediction (SHAP)")

try:
    # prefer the new API
    explainer = None
    try:
        explainer = shap.Explainer(model)           # generic explainer (works for many models)
        shap_exp = explainer(input_df)              # shap Explanation object
    except Exception:
        # fallback to TreeExplainer for tree models
        explainer = shap.TreeExplainer(model)
        shap_vals_raw = explainer.shap_values(input_df)  # could be list or array
        # convert raw output to a shap.Explanation-like structure below

        # Create a shap_explanation-like dict
        if isinstance(shap_vals_raw, list):
            # binary classification often returns [neg, pos] arrays shape (n_samples, n_features)
            if len(shap_vals_raw) == 2:
                vals = shap_vals_raw[1]   # use class=1 contributions
            else:
                vals = shap_vals_raw[0]
        else:
            vals = shap_vals_raw

        # now make a simple object with .values and .base_values if needed
        class SimpleExp:
            pass
        shap_exp = SimpleExp()
        shap_exp.values = vals              # shape (n_samples, n_features) or (n_samples, n_features, classes)
        # base_values may be returned by explainer in different ways
        try:
            shap_exp.base_values = explainer.expected_value
        except Exception:
            shap_exp.base_values = None

    # At this point shap_exp should exist.
    # Extract numeric contributions for the single sample:
    vals = shap_exp.values
    # If returned shape is (1, n_features, n_classes) -> pick class 1 if available
    if hasattr(vals, "ndim") and vals.ndim == 3:
        # shape = (1, n_features, n_classes)
        # choose the class index that matches the positive class (1) if possible
        class_idx = 1 if vals.shape[2] > 1 else 0
        contributions = vals[0, :, class_idx]
    elif hasattr(vals, "ndim") and vals.ndim == 2:
        # shape = (1, n_features)
        contributions = vals[0, :]
    else:
        # fallback: try to convert to array
        contributions = np.array(vals).reshape(-1)

    # Build a DataFrame for plotting and top-features
    feat_names = input_df.columns.tolist()
    contrib_df = pd.DataFrame({
        "feature": feat_names,
        "shap_value": contributions
    }).set_index("feature")

    # Plot horizontal bars (manual matplotlib) so it works for a single sample
    fig, ax = plt.subplots(figsize=(8, 5))
    vals_plot = contrib_df["shap_value"]
    colors = ["red" if v > 0 else "blue" for v in vals_plot]
    vals_plot.sort_values().plot(kind="barh", color=[ "red" if v>0 else "blue" for v in vals_plot.sort_values() ], ax=ax)
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_ylabel("Feature")
    ax.grid(False)
    st.pyplot(fig)

    # Top 3 features (by absolute contribution)
    top3 = contrib_df["shap_value"].abs().sort_values(ascending=False).head(3)
    st.write("*Top factors influencing this prediction:*")
    for feat in top3.index:
        v = contrib_df.loc[feat, "shap_value"]
        direction = "increased" if v > 0 else "decreased"
        st.write(f"- *{feat}* {direction} the chance of disease (SHAP = {v:.3f})")

except Exception as e:
    st.error(f"⚠ Could not generate SHAP explanation: {e}")

# ----------------------------
# FOOTER
# ----------------------------
st.divider()
st.markdown("🧠 This app is for educational and demonstration purposes only.")