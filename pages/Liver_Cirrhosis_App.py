import streamlit as st
import pandas as pd
import pickle
import os
import shap
import matplotlib.pyplot as plt

# 🧠 Load model, scaler, feature names, and label encoder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

model = pickle.load(open(os.path.join(MODEL_DIR, "Cirrhosis_rf_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(MODEL_DIR, "Cirrhosis_scaler.pkl"), "rb"))
feature_names = pickle.load(open(os.path.join(MODEL_DIR, "Cirrhosis_feature_names.pkl"), "rb"))
label_encoder = pickle.load(open(os.path.join(MODEL_DIR, "Cirrhosis_label_encoder.pkl"), "rb"))

st.title("🩺 Liver Cirrhosis Disease Prediction")
st.write("Enter patient details to predict cirrhosis outcome:")

# 🧾 Collect inputs
n_days = st.number_input("Number of Days", min_value=0, value=100)
drug = st.selectbox("Drug (0 = None, 1 = D-penicillamine)", [0, 1])
age = st.number_input("Age (in days)", min_value=0, value=20000)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
ascites = st.selectbox("Ascites (0 = No, 1 = Yes)", [0, 1])
hepatomegaly = st.selectbox("Hepatomegaly (0 = No, 1 = Yes)", [0, 1])
spiders = st.selectbox("Spiders (0 = No, 1 = Yes)", [0, 1])
edema = st.selectbox("Edema (0 = No, 1 = Yes)", [0, 1])
bilirubin = st.number_input("Bilirubin", min_value=0.0, value=1.0, format="%.2f")
cholesterol = st.number_input("Cholesterol", min_value=0.0, value=200.0, format="%.2f")
albumin = st.number_input("Albumin", min_value=0.0, value=3.0, format="%.2f")
copper = st.number_input("Copper", min_value=0.0, value=50.0, format="%.2f")
alk_phos = st.number_input("Alkaline Phosphatase", min_value=0.0, value=1000.0, format="%.2f")
sgot = st.number_input("SGOT", min_value=0.0, value=100.0, format="%.2f")
triglycerides = st.number_input("Triglycerides", min_value=0.0, value=150.0, format="%.2f")
platelets = st.number_input("Platelets", min_value=0, value=200, step=1)
prothrombin = st.number_input("Prothrombin", min_value=0.0, value=10.0, format="%.2f")
stage = st.selectbox("Stage", [1, 2, 3, 4])
age_years = st.number_input("Age (in years)", min_value=0.0, value=50.0, format="%.1f")

# Prepare DataFrame
input_data = pd.DataFrame([[n_days, drug, age, sex, ascites, hepatomegaly,
    spiders, edema, bilirubin, cholesterol, albumin, copper, alk_phos,
    sgot, triglycerides, platelets, prothrombin, stage, age_years]],
    columns=feature_names)

# Predict button
if st.button("Predict"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    prediction_label = label_encoder.inverse_transform([prediction])[0]
    confidence = probabilities[prediction] * 100

    # Save for explanation
    st.session_state["input_scaled"] = input_scaled
    st.session_state["prediction"] = prediction
    st.session_state["input_data"] = input_data

    st.subheader("🔎 Prediction Result")
    if prediction_label == 1:
        st.error(f"⚠️ The model predicts a **Risk of Cirrhosis Disease** (Risk: {confidence:.2f}%)")
    else:
        st.success(f"✅ The model predicts **No Cirrhosis Disease**")

    
    # Show Accuracy Only for Predicted Class
    try:
        report_path = os.path.join(MODEL_DIR, "Cirrhosis_class_report.pkl")
        class_report = pickle.load(open(report_path, "rb"))

        st.subheader("📊 Model Accuracy for This Prediction")

        if prediction_label == 1:  # Disease
            disease_acc = class_report["1"]["precision"] * 100
            st.info(f"**Accuracy for predicting Disease (Class 1):** {disease_acc:.2f}%")
        else:  # No Disease
            no_disease_acc = class_report["0"]["precision"] * 100
            st.info(f"**Accuracy for predicting No Disease (Class 0):** {no_disease_acc:.2f}%")

    except Exception as e:
        st.warning("⚠️ Class-wise accuracy file not found. Save 'Cirrhosis_class_report.pkl' during training.")


# Explain Prediction button
if "input_scaled" in st.session_state and st.button("🔍 Explain Prediction"):
    st.subheader("🧠 Why this prediction?")
    input_scaled = st.session_state["input_scaled"]
    prediction = st.session_state["prediction"]
    input_data = st.session_state["input_data"]

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_scaled)

        # Handle binary SHAP format
        if isinstance(shap_values, list):
            shap_values_for_class = shap_values[prediction].reshape(-1)
        else:
            shap_values_for_class = shap_values.reshape(-1)

        # Ensure lengths match
        feature_len = len(feature_names)
        shap_len = len(shap_values_for_class)

        if shap_len != feature_len:
            min_len = min(feature_len, shap_len)
            shap_values_for_class = shap_values_for_class[:min_len]
            feature_names = feature_names[:min_len]

        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "Impact": shap_values_for_class
        }).sort_values(by="Impact", key=abs, ascending=False)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        shap_df.head(10).set_index("Feature")["Impact"].plot(kind="barh", ax=ax)
        ax.set_title("Top 10 Factors Affecting Prediction")
        ax.set_xlabel("Feature Impact (Positive → Higher Risk, Negative → Lower Risk)")
        st.pyplot(fig)

        st.write("### 🩸 Top 5 reasons influencing this prediction:")
        for i, row in shap_df.head(5).iterrows():
            direction = "⬆️ Increases risk" if row["Impact"] > 0 else "⬇️ Decreases risk"
            st.write(f"**{row['Feature']}** — {direction}")

        st.info("💡 Higher SHAP values (positive impact) contribute to higher disease risk.")

    except Exception as e:
        st.error(f"⚠️ Could not generate explanation: {str(e)}")
