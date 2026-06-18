import streamlit as st

st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Multiple Chronic Disease Prediction System using Machine Learning")

st.markdown("""
## Welcome, 

This system predicts multiple chronic diseases using trained Machine Learning models with better accuracy.

### Available Diseases to Predict

- ❤️ Heart Disease
- 🩸 Diabetes
- 🧠 Parkinson's Disease
- 🫁 COPD
- 🏥 Liver Cirrhosis

### How to Use

1. Select a disease from the sidebar.
2. Enter patient details.
3. Click Predict.
4. View prediction result along with Explainable AI.

Use the navigation menu on the left to begin.
""")