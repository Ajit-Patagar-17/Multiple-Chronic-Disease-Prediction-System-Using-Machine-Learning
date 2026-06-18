import streamlit as st
import numpy as np
import pickle
import pandas as pd
import shap
import matplotlib.pyplot as plt


# PAGE SETTINGS
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")


# LOAD MODEL, SCALER, SHAP
model = pickle.load(open("models/heart_model.pkl", "rb"))
scaler = pickle.load(open("models/heart_scaler.pkl", "rb"))

# Create SHAP explainer (works for ANY model)
explainer = shap.Explainer(model)

st.title("❤️ Heart Disease Prediction with Explainable AI")

st.write("Enter patient details to predict the risk of heart disease.")


# INPUT FIELDS
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=50, max_value=250, value=120)
chol = st.number_input("Cholesterol (chol)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", ["No", "Yes"])
restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved (thalach)", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina (exang)", ["No", "Yes"])
oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, format="%.1f")
slope = st.selectbox("Slope of peak exercise ST segment (slope)", [0, 1, 2])
ca = st.number_input("Number of major vessels colored by fluoroscopy (ca)", min_value=0, max_value=3, value=0)
thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

# Convert categorical to numeric
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

# Create DF
feature_df = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]],
                          columns=["age", "sex", "cp", "trestbps", "chol", "fbs",
                                   "restecg", "thalach", "exang", "oldpeak", "slope",
                                   "ca", "thal"])

features_scaled = scaler.transform(feature_df)


# PREDICTION BUTTON
if st.button("Predict", key="predict_heart"):
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)[0][1] * 100

    st.subheader("🔍 Prediction Result")
    if prediction[0] == 1:
        st.error(f"⚠️ The model predicts **Heart Disease** (Risk: {probability:.2f}%)")
    else:
        st.success(f"✅ The model predicts **No Heart Disease** (Confidence: {100-probability:.2f}%)")

    st.session_state["input_scaled"] = features_scaled  # store for SHAP use


# SHAP EXPLAINABLE AI
if st.button("Explain Prediction (SHAP)", key="explain_heart"):

    if "input_scaled" not in st.session_state:
        st.warning("⚠️ Please click Predict first.")
    else:
        st.subheader("📘 SHAP Explainable AI")

        #shap.initjs()

        # Use ORIGINAL, NOT SCALED DATA
        X_input = feature_df.copy()

        # Background data must have same columns
        background = pd.DataFrame(np.random.rand(50, len(X_input.columns)), 
                                  columns=X_input.columns)

        # Universal Explainer (most stable)
        explainer = shap.Explainer(model, background, feature_names=X_input.columns)

        # Compute SHAP values
        shap_values = explainer(X_input)

        # Extract proper Explanation object
        shap_single = shap_values[0]

        

        # -------------- BAR PLOT --------------
        st.write("### 🔎 Feature Impact (Bar Plot)")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        shap.plots.bar(shap_single, show=False)
        st.pyplot(fig1)

        # # -------------- WATERFALL --------------
        # st.write("### 🌊 Waterfall Plot")
        # fig2, ax2 = plt.subplots(figsize=(10, 6))
        # shap.plots.waterfall(shap_single, show=False)
        # st.pyplot(fig2)
