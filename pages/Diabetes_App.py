import streamlit as st
import pandas as pd
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt

# ---------------------------------------
# PAGE CONFIG
# ---------------------------------------
st.set_page_config(page_title="Diabetes Prediction", layout="wide")

# ---------------------------------------
# LOAD MODEL, SCALER, FEATURE NAMES
# ---------------------------------------
model = pickle.load(open("models/diabetes_rf_model.pkl", "rb"))
scaler = pickle.load(open("models/diabetes_scaler.pkl", "rb"))
feature_names = pickle.load(open("models/diabetes_feature_names.pkl", "rb"))

# ---------------------------------------
# TITLE
# ---------------------------------------
st.title("🩺 Diabetes Prediction App")
st.write("Enter the details below to check your Diabetes risk:")

# ---------------------------------------
# USER INPUT FUNCTION
# ---------------------------------------
def user_input_features():
    Age = st.number_input("Age", 1, 120, 30)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Gender = 1 if Gender == "Male" else 0

    Polyuria = 1 if st.selectbox("Polyuria (Frequent urination)", ["No", "Yes"]) == "Yes" else 0
    Polydipsia = 1 if st.selectbox("Polydipsia (Excessive thirst)", ["No", "Yes"]) == "Yes" else 0
    sudden_weight_loss = 1 if st.selectbox("Sudden Weight Loss", ["No", "Yes"]) == "Yes" else 0
    weakness = 1 if st.selectbox("Weakness", ["No", "Yes"]) == "Yes" else 0
    Polyphagia = 1 if st.selectbox("Polyphagia (Excessive hunger)", ["No", "Yes"]) == "Yes" else 0
    Genital_thrush = 1 if st.selectbox("Genital Thrush", ["No", "Yes"]) == "Yes" else 0
    visual_blurring = 1 if st.selectbox("Visual Blurring", ["No", "Yes"]) == "Yes" else 0
    Itching = 1 if st.selectbox("Itching", ["No", "Yes"]) == "Yes" else 0
    Irritability = 1 if st.selectbox("Irritability", ["No", "Yes"]) == "Yes" else 0
    delayed_healing = 1 if st.selectbox("Delayed Healing", ["No", "Yes"]) == "Yes" else 0
    partial_paresis = 1 if st.selectbox("Partial Paresis", ["No", "Yes"]) == "Yes" else 0
    muscle_stiffness = 1 if st.selectbox("Muscle Stiffness", ["No", "Yes"]) == "Yes" else 0
    Alopecia = 1 if st.selectbox("Alopecia (Hair loss)", ["No", "Yes"]) == "Yes" else 0
    Obesity = 1 if st.selectbox("Obesity", ["No", "Yes"]) == "Yes" else 0

    data = {
        "Age": Age,
        "Gender": Gender,
        "Polyuria": Polyuria,
        "Polydipsia": Polydipsia,
        "sudden weight loss": sudden_weight_loss,
        "weakness": weakness,
        "Polyphagia": Polyphagia,
        "Genital thrush": Genital_thrush,
        "visual blurring": visual_blurring,
        "Itching": Itching,
        "Irritability": Irritability,
        "delayed healing": delayed_healing,
        "partial paresis": partial_paresis,
        "muscle stiffness": muscle_stiffness,
        "Alopecia": Alopecia,
        "Obesity": Obesity
    }

    return pd.DataFrame([data])


# GET USER INPUT
input_df = user_input_features()

# Prediction and SHAP Explaination
if st.button("Predict"):
    # Scale input
    scaled = scaler.transform(input_df)

    # Prediction and probability
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    # Show accuracy
    MODEL_ACCURACY = 0.95
    st.info(f"Model Accuracy: **{MODEL_ACCURACY*100:.2f}%**")

    # Show prediction
    if pred == 1:
        st.error("⚠️ The model predicts you **may have Diabetes**.")
        st.warning(f"Risk Score: **{prob*100:.2f}%**")
    else:
        st.success("✅ The model predicts **No Diabetes**.")
        st.info(f"Risk Score: **{prob*100:.2f}%**")

    
    # SHAP EXPLAINABLE AI
    st.subheader("🔍 Explainable AI – Feature Importance (SHAP)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(scaled)

    # Binary classifier: pick class 1 if SHAP returns a list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Ensure 1D array for single-row input
    shap_values = np.array(shap_values)
    if shap_values.ndim > 1:
        shap_values = shap_values.flatten()

    # Double-check lengths match
    if len(shap_values) != len(feature_names):
        shap_values = shap_values[:len(feature_names)]

    # Create DataFrame
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": np.abs(shap_values)
    }).sort_values(by="SHAP Value", ascending=True)

    # Plot horizontal bar
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(shap_df["Feature"], shap_df["SHAP Value"], color='skyblue')
    ax.set_xlabel("Impact on Prediction")
    ax.set_title("Feature Importance using SHAP")
    plt.tight_layout()
    st.pyplot(fig)
