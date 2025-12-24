import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "finalized_model.sav")
scaler_path = os.path.join(BASE_DIR, "scaler_model.sav")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)


# -------------------------------
# Feature names (order must match training)
# -------------------------------
feature_names = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
    'density', 'pH', 'sulphates', 'alcohol'
]

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🍷 Wine Quality Prediction")
st.write("Enter wine chemical properties to predict wine quality.")

# Inputs
fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, max_value=20.0, value=7.0, step=0.1)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=1.5, value=0.3, step=0.01)
residual_sugar = st.number_input("Residual Sugar", min_value=0.0, max_value=20.0, value=2.5, step=0.1)
chlorides = st.number_input("Chlorides", min_value=0.0, max_value=0.5, value=0.05, step=0.001)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, max_value=100.0, value=15.0, step=1.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, max_value=300.0, value=46.0, step=1.0)
density = st.number_input("Density", min_value=0.990, max_value=1.005, value=0.995, step=0.0001)
pH = st.number_input("pH", min_value=2.5, max_value=4.5, value=3.3, step=0.01)
sulphates = st.number_input("Sulphates", min_value=0.0, max_value=2.0, value=0.6, step=0.01)
alcohol = st.number_input("Alcohol", min_value=5.0, max_value=15.0, value=10.0, step=0.1)

# -------------------------------
# Prepare input array
# -------------------------------
input_data = np.array([[
    fixed_acidity,
    volatile_acidity,
    citric_acid,
    residual_sugar,
    chlorides,
    free_sulfur_dioxide,
    total_sulfur_dioxide,
    density,
    pH,
    sulphates,
    alcohol
]])

# Apply log transform (same as training)
epsilon = 1e-6
log_idx = [3, 4, 5, 6, 9]  # residual sugar, chlorides, free/total sulfur, sulphates
for idx in log_idx:
    input_data[0][idx] = np.log(input_data[0][idx] + epsilon)

# Convert to DataFrame
input_df = pd.DataFrame(input_data, columns=feature_names)

# Scale input
scaled_input = scaler.transform(input_df)

# -------------------------------
# Prediction + Balloons 🎈
# -------------------------------
if st.button("Predict Wine Quality"):
    prediction = model.predict(scaled_input)[0]

    # 🎈 Balloons blast
    st.balloons()

    # Map prediction to category
    if prediction <= 5:
        quality_label = "Bad 🍷"
        st.error(f"Predicted Wine Quality: {prediction} ({quality_label})")

    elif prediction == 6:
        quality_label = "Average 🍷"
        st.warning(f"Predicted Wine Quality: {prediction} ({quality_label})")

    else:
        quality_label = "Good 🍷"
        st.success(f"Predicted Wine Quality: {prediction} ({quality_label})")
