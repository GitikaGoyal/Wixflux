import streamlit as st
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load('svm_model.pkl')       # or 'rf_model.pkl'
scaler = joblib.load('scaler.pkl')

st.title("🧬 Breast Cancer Prediction App")
st.write("Enter 30 numerical tumor features below:")

# Input fields for 30 features
features = []
for i in range(30):
    val = st.number_input(f"Feature {i+1}", format="%.5f")
    features.append(val)

# Predict on button click
if st.button("Predict"):
    input_data = np.array(features).reshape(1, -1)
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]

    result = "Benign ✅" if prediction == 1 else "Malignant ❌"
    st.success(f"Prediction: {result}")
