import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Thyroid Cancer Risk - Demo App", layout="centered")

st.title("Thyroid Cancer Risk — Streamlit App")
st.markdown(
    """
This app accepts human-friendly dropdowns (e.g., 'Male'/'Female') and internally encodes them
the same way your model expects. Upload your `thyroid_cancer_risk_model.pkl`, `label_encoders.pkl`, and
`scaler.pkl` for real predictions. Optionally upload `feature_order.pkl` (exact column order used during training).
"""
)

# === Feature definitions (match notebook) ===
numerical_cols = ['Age', 'TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']
categorical_cols = [
    'Gender', 'Family_History', 'Radiation_Exposure', 'Iodine_Deficiency',
    'Smoking', 'Obesity', 'Diabetes', 'Thyroid_Cancer_Risk'
]
target_col = 'Diagnosis'  # name used when saving encoders for target (if applicable)

# === Friendly options and fallback encodings ===
fallback_maps = {
    'Gender': ['Female', 'Male'],
    'Family_History': ['No', 'Yes'],
    'Radiation_Exposure': ['No', 'Yes'],
    'Iodine_Deficiency': ['No', 'Yes'],
    'Smoking': ['No', 'Yes'],
    'Obesity': ['No', 'Yes'],
    'Diabetes': ['No', 'Yes'],
    'Thyroid_Cancer_Risk': ['Low', 'Moderate', 'High'],
}

fallback_encoding = {
    'Gender': {'Female': 0, 'Male': 1},
    'Family_History': {'No': 0, 'Yes': 1},
    'Radiation_Exposure': {'No': 0, 'Yes': 1},
    'Iodine_Deficiency': {'No': 0, 'Yes': 1},
    'Smoking': {'No': 0, 'Yes': 1},
    'Obesity': {'No': 0, 'Yes': 1},
    'Diabetes': {'No': 0, 'Yes': 1},
    'Thyroid_Cancer_Risk': {'Low': 0, 'Moderate': 1, 'High': 2},
}

# === Sidebar: upload model artifacts (optional) ===
st.sidebar.header("Upload model files (optional)")
model_file = st.sidebar.file_uploader("Upload thyroid_cancer_risk_model.pkl", type=["pkl", "joblib"])
encoders_file = st.sidebar.file_uploader("Upload label_encoders.pkl", type=["pkl", "joblib"])
scaler_file = st.sidebar.file_uploader("Upload scaler.pkl", type=["pkl", "joblib"])
feature_order_file = st.sidebar.file_uploader("Optional: upload feature_order.pkl", type=["pkl", "joblib"])

loaded_model = None
loaded_encoders = None
loaded_scaler = None
saved_feature_order = None

# Try to load uploaded files
if model_file:
    try:
        loaded_model = joblib.load(model_file)
        st.sidebar.success("Model loaded from upload.")
    except Exception as e:
        st.sidebar.error(f"Failed to load uploaded model: {e}")

if encoders_file:
    try:
        loaded_encoders = joblib.load(encoders_file)
        st.sidebar.success("Encoders loaded from upload.")
    except Exception as e:
        st.sidebar.error(f"Failed to load uploaded encoders: {e}")

if scaler_file:
    try:
        loaded_scaler = joblib.load(scaler_file)
        st.sidebar.success("Scaler loaded from upload.")
    except Exception as e:
        st.sidebar.error(f"Failed to load uploaded scaler: {e}")

if feature_order_file:
    try:
        saved_feature_order = joblib.load(feature_order_file)
        st.sidebar.success("Feature order loaded from upload.")
    except Exception as e:
        st.sidebar.error(f"Failed to load uploaded feature order: {e}")

# If not uploaded, try files on disk
if loaded_model is None:
    if os.path.exists("thyroid_cancer_risk_model.pkl"):
        try:
            loaded_model = joblib.load("thyroid_cancer_risk_model.pkl")
            st.sidebar.success("Model loaded from disk.")
        except Exception as e:
            st.sidebar.error(f"Failed to load model from disk: {e}")

if loaded_encoders is None:
    if os.path.exists("label_encoders.pkl"):
        try:
            loaded_encoders = joblib.load("label_encoders.pkl")
            st.sidebar.success("Encoders loaded from disk.")
        except Exception as e:
            st.sidebar.error(f"Failed to load encoders from disk: {e}")

if loaded_scaler is None:
    if os.path.exists("scaler.pkl"):
        try:
            loaded_scaler = joblib.load("scaler.pkl")
            st.sidebar.success("Scaler loaded from disk.")
        except Exception as e:
            st.sidebar.error(f"Failed to load scaler from disk: {e}")

if saved_feature_order is None and os.path.exists("feature_order.pkl"):
    try:
        saved_feature_order = joblib.load("feature_order.pkl")
        st.sidebar.success("Feature order loaded from disk.")
    except Exception:
        saved_feature_order = None

# === Input UI ===
st.header("Patient History & Demographics")

col1, col2 = st.columns(2)
with col1:
    Age = st.number_input("Age", min_value=0, max_value=120, value=40, step=1)
    TSH_Level = st.number_input("TSH Level", value=2.5, format="%.3f")
    T3_Level = st.number_input("T3 Level", value=1.2, format="%.3f")
    T4_Level = st.number_input("T4 Level", value=8.0, format="%.3f")
    Nodule_Size = st.number_input("Nodule Size (cm)", value=1.0, format="%.2f")

with col2:
    widgets = {}
    for col in categorical_cols:
        opts = fallback_maps.get(col, ['No', 'Yes'])
        widgets[col] = st.selectbox(col.replace('_', ' '), opts, index=0 if len(opts) > 0 else 0)

# Build DataFrame of human-readable inputs
input_dict = {
    'Age': Age, 'TSH_Level': TSH_Level, 'T3_Level': T3_Level, 'T4_Level': T4_Level, 'Nodule_Size': Nodule_Size
}
for c in categorical_cols:
    input_dict[c] = widgets.get(c)

df_input = pd.DataFrame([input_dict])

st.subheader("Patient Inputs (human-readable)")
st.table(df_input.transpose())

# === Encoding function (silent) ===
def encode_with_loaded(encoders, col, value):
    le = None
    if encoders:
        # encoders may be dict-like: col -> LabelEncoder
        le = encoders.get(col) if isinstance(encoders, dict) else None
    if le is None:
        return fallback_encoding.get(col, {}).get(value, value)
    try:
        return int(le.transform([value])[0])
    except Exception:
        try:
            return int(list(le.classes_).index(value))
        except Exception:
            return fallback_encoding.get(col, {}).get(value, value)

# Build encoded row silently (not displayed)
encoded_row = {}
for n in numerical_cols:
    encoded_row[n] = df_input.loc[0, n]

for c in categorical_cols:
    encoded_row[c] = encode_with_loaded(loaded_encoders, c, df_input.loc[0, c])

encoded_df = pd.DataFrame([encoded_row])

# === Prediction UI: Predict button ===
predict = st.button("Predict")

if predict:
    # If no model loaded, warn and stop
    if loaded_model is None:
        st.warning("No model loaded. Please upload or place `thyroid_cancer_risk_model.pkl` in the app folder to enable prediction.")
    else:
        features_for_model = encoded_df.copy()

        # Apply scaler if available
        try:
            if loaded_scaler is not None:
                missing_nums = [c for c in numerical_cols if c not in features_for_model.columns]
                if missing_nums:
                    st.error(f"Missing numerical columns before scaling: {missing_nums}")
                else:
                    scaled_nums = loaded_scaler.transform(features_for_model[numerical_cols])
                    for i, col in enumerate(numerical_cols):
                        features_for_model[col] = scaled_nums[:, i]
        except Exception as e:
            st.error(f"Error applying scaler: {e}")
            features_for_model = encoded_df.copy()  # revert to unscaled for debugging

        # Determine desired order
        desired_order = None
        if saved_feature_order is not None:
            desired_order = list(saved_feature_order)
        elif hasattr(loaded_model, "feature_names_in_"):
            try:
                desired_order = list(loaded_model.feature_names_in_)
            except Exception:
                desired_order = None

        if desired_order is None:
            desired_order = numerical_cols + categorical_cols

        # Normalize helper for small name differences
        def normalize(name):
            return name.replace(" ", "_").lower()

        present_cols = list(features_for_model.columns)
        norm_present = {normalize(c): c for c in present_cols}

        mapped_order = []
        missing = []
        for want in desired_order:
            if want in features_for_model.columns:
                mapped_order.append(want)
            else:
                n = normalize(want)
                if n in norm_present:
                    mapped_order.append(norm_present[n])
                else:
                    missing.append(want)

        if missing:
            st.error(
                "Model expects these features but they are missing or named differently:\n"
                + ", ".join(missing)
                + "\n\nPresent columns in the app: "
                + ", ".join(present_cols)
            )
            st.info("Tip: save & upload `feature_order.pkl` from your training script (joblib.dump(list(X_train.columns), 'feature_order.pkl'))")
        else:
            try:
                features_for_model = features_for_model[mapped_order]
            except Exception as e:
                st.error(f"Failed to reorder features for the model: {e}")

            # Run prediction
            try:
                pred_encoded = loaded_model.predict(features_for_model)[0]
                if loaded_encoders and target_col in loaded_encoders:
                    try:
                        pred_label = loaded_encoders[target_col].inverse_transform([pred_encoded])[0]
                    except Exception:
                        pred_label = str(pred_encoded)
                else:
                    pred_label = str(pred_encoded)

                # Nice result display
                st.success(f"🧠 Predicted diagnosis: **{pred_label}**")
                st.write(f"Encoded prediction value: `{pred_encoded}`")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.write("Desired feature order (expected by model):")
                st.write(desired_order)
                st.write("Mapped (actual) order used for prediction:")
                st.write(mapped_order)

# Footer
st.markdown("---")
st.caption("If you want me to adapt the app to your exact `label_encoders.pkl` format, upload that file and I can further tighten mappings.")
