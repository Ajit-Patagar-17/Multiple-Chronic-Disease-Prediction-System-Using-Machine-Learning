# copd_app.py
import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap

st.set_page_config(page_title="COPD Severity Prediction (XAI)", layout="wide")

# -----------------------
# USER CONFIG: adjust labels if you want different wording
# -----------------------
CLASS_LABELS = {
    0: "No COPD",
    1: "Mild COPD",
    2: "Moderate COPD",
    3: "Severe COPD"
}

# -----------------------
# locate models folder (relative to this file)
# -----------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
candidates = [
    os.path.join(current_dir, "models"),
    os.path.join(current_dir, "..", "models"),
    os.path.join(current_dir, "..", "..", "models"),
]
models_dir = next((c for c in candidates if os.path.isdir(c)), None)
if models_dir is None:
    st.error("Could not find a 'models' folder. Put COPD_model.pkl, COPD_scaler.pkl and COPD_features.pkl in a folder named 'models' near this app.")
    st.stop()

model_path = os.path.join(models_dir, "COPD_model.pkl")
scaler_path = os.path.join(models_dir, "COPD_scaler.pkl")
features_path = os.path.join(models_dir, "COPD_features.pkl")

for p in (model_path, scaler_path, features_path):
    if not os.path.exists(p):
        st.error(f"Required file not found: {p}")
        st.stop()


# load model, scaler, features
with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

with open(features_path, "rb") as f:
    feature_names = pickle.load(f)

# sanitize feature names and remove possible target column
feature_names = [str(x) for x in feature_names]
feature_names = [f for f in feature_names if f.upper() != "COPDSEVERITY" and f != "COPDSEVERITY"]


# Title and instructions
st.title("🫁 COPD Severity Prediction — Explainable AI (SHAP)")
st.write("Provide patient data. Make sure features correspond to training features and order.")


# INPUT FIELDS (vertical layout one-per-row)
st.write("### Patient features (enter values same as used for training)")
inputs = {}
binary_cols = {"gender", "Diabetes", "muscular", "hypertension", "AtrialFib", "IHD"}
# some columns in your dataset were 'copd' and 'smoking' with specific categorical values
for feat in feature_names:
    label = feat
    if feat == "smoking":
        inputs[feat] = st.selectbox(label + " (1=No, 2=Yes)", options=[1, 2], index=0)
    elif feat == "copd":
        inputs[feat] = st.selectbox(label + " (1..4)", options=[1, 2, 3, 4], index=0)
    elif feat in binary_cols:
        inputs[feat] = st.selectbox(label + " (0=No,1=Yes)", options=[0, 1], index=0)
    else:
        # show float number input with a reasonable default 0.0
        inputs[feat] = st.number_input(label, value=0.0, format="%.5f")

# build DataFrame in exact order
input_df = pd.DataFrame([[inputs[f] for f in feature_names]], columns=feature_names)


# SCALE & PREDICT (with error handling)
try:
    input_scaled = scaler.transform(input_df)
except Exception as e:
    st.error("Error applying scaler to input — ensure scaler was fitted with the same feature names/order.")
    st.exception(e)
    st.stop()

if st.button("Predict"):
    try:
        pred = int(model.predict(input_scaled)[0])
        proba = model.predict_proba(input_scaled)[0]
        pred_prob = float(proba[pred])  # probability of predicted class
    except Exception as e:
        st.error("Model prediction failed. Ensure model and scaler were created from the same features/order.")
        st.exception(e)
        st.stop()

    # Display result (simple, no full probability table)
    if pred in CLASS_LABELS:
        label_text = CLASS_LABELS[pred]
    else:
        label_text = f"Class {pred}"

    if pred == 0:
        st.success(f"✅ The model predicts **{label_text}**  — Confidence: {pred_prob*100:.2f}%")
    else:
        st.error(f"⚠️ The model predicts **{label_text}**  — Confidence: {pred_prob*100:.2f}%")

    # store for SHAP explain button
    st.session_state["copd_input_scaled"] = input_scaled
    st.session_state["copd_input_df"] = input_df
    st.session_state["copd_prediction"] = pred
    st.session_state["copd_pred_prob"] = pred_prob


# Explain (SHAP)
if st.button("Explain Prediction (SHAP)"):
    if "copd_input_scaled" not in st.session_state:
        st.warning("⚠️ Please click Predict first.")
    else:
        st.subheader("📘 SHAP Explanation")

        input_scaled = st.session_state["copd_input_scaled"]
        input_df_saved = st.session_state["copd_input_df"]
        pred_class = int(st.session_state.get("copd_prediction", 0))

        # Create explainer robustly
        try:
            # prefer newer universal Explainer (works for many models)
            explainer = shap.Explainer(model, feature_names=feature_names)
        except Exception:
            try:
                explainer = shap.TreeExplainer(model)
            except Exception:
                explainer = None

        if explainer is None:
            st.error("Could not create a SHAP explainer for this model (SHAP may not support the model type).")
            st.stop()

        # compute shap values (try modern API)
        try:
            shap_exp = explainer(input_scaled)   # returns Explanation object
            # shap_exp should be indexable; shap_exp[0] is the single-instance Explanation
            if hasattr(shap_exp, "values"):
                # this is an Explanation object
                shap_vals_for_inst = shap_exp[0]   # Explanation for instance 0
            else:
                # fallback - try list/array style
                shap_vals_for_inst = shap_exp
        except Exception as e:
            st.warning("SHAP calculation failed (trying alternate calls).")
            st.exception(e)
            try:
                explainer_alt = shap.TreeExplainer(model)
                shap_vals_alt = explainer_alt.shap_values(input_scaled)
                # normalize to array for plotting
                if isinstance(shap_vals_alt, list) and len(shap_vals_alt) > pred_class:
                    arr = np.array(shap_vals_alt[pred_class])
                    shap_arr = arr[0]
                elif isinstance(shap_vals_alt, np.ndarray) and shap_vals_alt.ndim >= 2:
                    shap_arr = shap_vals_alt[0]
                else:
                    shap_arr = np.array(shap_vals_alt).flatten()
                # create simple Explanation-like object for further plotting
                class SimpleExp:
                    def __init__(self, values, data, feature_names):
                        self.values = np.array([values])
                        self.data = np.array([data])
                        self.feature_names = feature_names
                        self.base_values = None
                shap_vals_for_inst = SimpleExp(shap_arr, input_df_saved.iloc[0].values, feature_names)
            except Exception as e2:
                st.error("Failed to compute SHAP values with fallback method.")
                st.exception(e2)
                st.stop()

       
        # Some SHAP versions produce nested arrays; normalize to plain numpy 1d array
        try:
            if hasattr(shap_vals_for_inst, "values"):
                vals = np.array(shap_vals_for_inst.values)
                # vals shape can be (1, n_features) or (1, n_features, outputs)
                if vals.ndim == 3:
                    # multi-output: pick pred_class if possible
                    try:
                        arr = vals[0, :, pred_class]
                    except Exception:
                        arr = vals[0, :, 0]
                elif vals.ndim == 2:
                    arr = vals[0, :]
                else:
                    arr = vals.flatten()
            else:
                arr = np.array(shap_vals_for_inst).flatten()
        except Exception as e:
            st.error("Failed to normalize SHAP values.")
            st.exception(e)
            st.stop()

        # align lengths if mismatch
        fnames = list(feature_names)
        if len(arr) != len(fnames):
            st.warning(f"Feature count ({len(fnames)}) and SHAP values count ({len(arr)}) differ. Aligning to the minimum length.")
            min_len = min(len(fnames), len(arr))
            fnames = fnames[:min_len]
            arr = arr[:min_len]

        # Build DataFrame sorted by absolute impact
        shap_df = pd.DataFrame({"Feature": fnames, "SHAP": arr})
        shap_df["absSHAP"] = np.abs(shap_df["SHAP"])
        shap_df = shap_df.sort_values("absSHAP", ascending=False).reset_index(drop=True)

        # ---------- BAR (matplotlib) ----------
        st.write("### 🔍 Feature importance (SHAP values) — bar plot")
        fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(shap_df))))
        ax.barh(shap_df["Feature"], shap_df["SHAP"])
        ax.set_xlabel("SHAP value (impact on model output)")
        ax.set_title("Feature contributions (sorted by absolute SHAP value)")
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)

        # ---------- WATERFALL (try modern API then legacy fallback) ----------
        # st.write("### 🌊 SHAP Waterfall (single prediction)")
        # plotted = False
        # try:
        #     # Try using modern SHAP waterfall if we have an Explanation object
        #     # We reconstruct a minimal Explanation if necessary
        #     try:
        #         # If we still have shap_exp Explanation object from above, use it
        #         if hasattr(shap_exp, "__len__") and hasattr(shap_exp[0], "values"):
        #             shap_target_exp = shap_exp[0]
        #             shap.plots.waterfall(shap_target_exp, max_display=15, show=False)
        #             plt.gcf().set_size_inches((8, 6))
        #             st.pyplot(plt.gcf())
        #             plotted = True
        #         else:
        #             raise Exception("modern waterfall not available")
        #     except Exception:
        #         # Legacy fallback: call waterfall_legacy with scalar expected_value if possible
        #         ev = None
        #         if hasattr(shap_exp, "base_values"):
        #             ev = shap_exp.base_values
        #         elif hasattr(explainer, "expected_value"):
        #             ev = explainer.expected_value
        #         # pick scalar expected value for shown class if needed
        #         if isinstance(ev, (list, np.ndarray)) and len(ev) > 1:
        #             try:
        #                 ev_plot = float(ev[pred_class])
        #             except Exception:
        #                 ev_plot = float(ev[0])
        #         elif ev is None:
        #             ev_plot = None
        #         else:
        #             ev_plot = float(ev)

        #         # try to call legacy waterfall using shap.plots._waterfall.waterfall_legacy
        #         try:
        #             # shap_values for a single instance (1d arr) and features as Series
        #             from shap.plots._waterfall import waterfall_legacy
        #             waterfall_legacy(
        #                 expected_val_for_plot=ev_plot,
        #                 shap_values=arr,
        #                 features=pd.Series({k: input_df.iloc[0][k] for k in fnames}),
        #                 feature_names=fnames,
        #                 show=False
        #             )
        #             plt.gcf().set_size_inches((8, 6))
        #             st.pyplot(plt.gcf())
        #             plotted = True
        #         except Exception as e_legacy:
        #             # last resort: show textual top features and warn
        #             st.warning("Could not render a SHAP waterfall using modern or legacy API. Showing textual reasons instead.")
        #             st.exception(e_legacy)
        # except Exception as e:
        #     st.warning("Waterfall plotting failed.")
        #     st.exception(e)

        # textual explanation
        st.write("### 📝 Top feature reasons (textual explanation)")
        topn = min(8, len(shap_df))
        for _, row in shap_df.head(topn).iterrows():
            feat = row["Feature"]
            val = row["SHAP"]
            direction = "increased" if val > 0 else "decreased"
            st.write(f"- **{feat}** (SHAP={val:.4f}) → {direction} the model's prediction for this patient.")

        st.success("SHAP explanation generated.")
