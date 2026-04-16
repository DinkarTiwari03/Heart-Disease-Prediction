import streamlit as st
import pandas as pd
import joblib

# Load artifacts
model = joblib.load('KNN_heart.pkl')
scaler = joblib.load('heart_scaler.pkl')
expected_columns = joblib.load('heart_columns.pkl')

if callable(expected_columns):
    expected_columns = expected_columns()

# UI
st.title("❤️ Heart Disease Prediction by Dinkar")
st.markdown("### Provide the following details:")

# Inputs
age = st.slider("Age", 18, 120, 30)
sex = st.selectbox("Sex", ["Male", "Female"])

# ✅ Chest Pain Mapping
chest_pain_map = {
    "Typical Angina": "TA",
    "Atypical Angina": "ATA",
    "Non-Anginal Pain": "NAP",
    "Asymptomatic": "ASY"
}
chest_pain_display = st.selectbox("Chest Pain Type", list(chest_pain_map.keys()))
chest_pain = chest_pain_map[chest_pain_display]

resting_bp = st.number_input("Resting Blood Pressure (mmHg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)

# ✅ Fasting BS (Yes/No → 1/0)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
fasting_bs_value = 1 if fasting_bs == "Yes" else 0

# ✅ ECG Mapping
resting_ecg_map = {
    "Normal": "Normal",
    "ST-T Wave Abnormality": "ST",
    "Left Ventricular Hypertrophy": "LVH"
}
resting_ecg_display = st.selectbox("Resting ECG", list(resting_ecg_map.keys()))
resting_ecg = resting_ecg_map[resting_ecg_display]

max_hr = st.slider("Max Heart Rate", 60, 220, 150)

exercise_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
exercise_angina_value = 1 if exercise_angina == "Yes" else 0

oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)

# ✅ Slope Mapping (already same but keeping structure clean)
slope_map = {
    "Upsloping": "Up",
    "Flat": "Flat",
    "Downsloping": "Down"
}
slope_display = st.selectbox("ST Slope", list(slope_map.keys()))
slope = slope_map[slope_display]

# Prediction
if st.button("Predict"):
    raw_data = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs_value,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_Y': exercise_angina_value,
        'ST_Slope_' + slope: 1
    }

    input_df = pd.DataFrame([raw_data])

    # Add missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure correct order
    input_df = input_df[expected_columns]

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)[0]

    # Output
    st.markdown("### Result:")
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")