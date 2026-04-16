import streamlit as st
import pandas as pd
import joblib

# Load artifacts
model = joblib.load('KNN_heart.pkl')
scaler = joblib.load('heart_scaler.pkl')
expected_columns = joblib.load('heart_columns.pkl')

# Fix if mistakenly saved as method
if callable(expected_columns):
    expected_columns = expected_columns()

# UI
st.title("❤️ Heart Disease Prediction by Dinkar")
st.markdown("### Provide the following details:")

# Inputs
age = st.slider("Age", 18, 120, 30)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
resting_bp = st.number_input("Resting Blood Pressure (mmHg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    ["Yes", "No"]
)

# Convert to 1/0
fasting_bs_value = 1 if fasting_bs == "Yes" else 0

resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Prediction
if st.button("Predict"):
    # Create input dictionary
    raw_data = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': int(fasting_bs),
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_Y': 1 if exercise_angina == "Y" else 0,
        'ST_Slope_' + slope: 1
    }

    # Convert to DataFrame
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