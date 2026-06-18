import streamlit as st
import numpy as np
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Parkinson's Disease Prediction", layout="wide")


model = pickle.load(open("models/parkinsons_model.pkl", "rb"))
scaler = pickle.load(open("models/parkinsons_scaler.pkl", "rb"))
feature_names = pickle.load(open("models/parkinsons_features.pkl", "rb"))
explainer = pickle.load(open("models/parkinsons_shap_explainer.pkl", "rb"))

st.title("🧠 Parkinson's Disease Prediction")
st.write("Enter patient voice measurement values to predict Parkinson's disease.")


input_data = {}
for feature in feature_names:
    default_value = 0.0
    input_data[feature] = st.number_input(feature, value=float(default_value))


input_df = pd.DataFrame([input_data])


if st.button("Predict", key="predict_parkinsons"):
   
    input_scaled = scaler.transform(input_df)

    
    st.session_state["input_scaled"] = input_scaled
    st.session_state["input_df"] = input_df

    # Prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1] * 100

    if prediction[0] == 1:
        st.error(f"⚠️ The model predicts **Parkinson's Disease** (Probability: {probability:.2f}%)")
    else:
        st.success(f"✅ The model predicts **No Parkinson's Disease** (Confidence: {100-probability:.2f}%)")

# -----------------------------
# SHAP EXPLAINABILITY BUTTON
# -----------------------------
if st.button("Explain Prediction (SHAP)", key="explain_shap"):
    if "input_scaled" not in st.session_state:
        st.error("⚠️ Please click Predict first!")
    else:
        st.subheader("📘 SHAP Explainable AI")

        input_scaled = st.session_state["input_scaled"]

        # Get SHAP values
        shap_values_all = explainer.shap_values(input_scaled)

        # Handle binary vs multi-class
        if isinstance(shap_values_all, list):
            # Multi-class
            prediction_class = model.predict(input_scaled)[0]
            shap_single_values = shap_values_all[prediction_class][0]
            base_value = explainer.expected_value[prediction_class]
        else:
            # Binary classification (tree-based)
            shap_single_values = shap_values_all[0]  # first row
            base_value = explainer.expected_value

        # Create SHAP Explanation object (pass 2D array as data)
        shap_single = shap.Explanation(
            values=shap_single_values,
            base_values=base_value,
            data=input_scaled,          # pass full 2D array
            feature_names=feature_names
        )

        # ------------ BAR PLOT -------------
        st.write("### 🔍 Feature Importance (Bar Plot)")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        shap.plots.bar(shap_single[0], show=False)   # plot first row
        st.pyplot(fig1)

        # # ------------ WATERFALL PLOT ----------
        # st.write("### 🌊 Waterfall Plot")
        # fig2, ax2 = plt.subplots(figsize=(10, 6))
        # shap.plots.waterfall(shap_single[0], show=False)  # plot first row
        # st.pyplot(fig2)
