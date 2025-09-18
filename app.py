import streamlit as st
import pandas as pd
import numpy as np
import pickle

# st.markdown(
#     """
#     <style>
#         body {
#             background-color: #e6ffe6;
#         }
#         .stApp {
#             background-color: #e6ffe6;
#         }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

st.markdown(
    """
    <style>
        .stApp {
            background-color: #ccffcc;
        }
    </style>
    """,
    unsafe_allow_html=True
)
# Load trained model
model_filename = 'crop_yield_model1.pkl'
with open(model_filename, 'rb') as f:
    model = pickle.load(f)

# Define input fields
st.title("🌾 Crop Yield Predictor")
st.markdown("Enter farm details to estimate expected crop yield.")

farm_area = st.number_input("Farm Area (acres)", min_value=0.0, value=100.0)
fertilizer = st.number_input("Fertilizer Used (tons)", min_value=0.0, value=5.0)
pesticide = st.number_input("Pesticide Used (kg)", min_value=0.0, value=1.0)
water_usage = st.number_input("Water Usage (cubic meters)", min_value=0.0, value=50000.0)

crop_type = st.selectbox("Crop Type", ['Cotton', 'Wheat', 'Rice', 'Maize'])  # Add your actual categories
irrigation_type = st.selectbox("Irrigation Type", ['Sprinkler', 'Drip', 'Flood'])
soil_type = st.selectbox("Soil Type", ['Loamy', 'Clay', 'Sandy'])
season = st.selectbox("Season", ['Kharif', 'Rabi', 'Summer'])

categorical_cols = ['Crop_Type', 'Irrigation_Type', 'Soil_Type', 'Season']

# Prepare input for prediction
input_data = pd.DataFrame([{
    'Farm_Area(acres)': farm_area,
    'Fertilizer_Used(tons)': fertilizer,
    'Pesticide_Used(kg)': pesticide,
    'Water_Usage(cubic meters)': water_usage,
    'Crop_Type': crop_type,
    'Irrigation_Type': irrigation_type,
    'Soil_Type': soil_type,
    'Season': season
}])

# Match training columns
# You must save X_encoded.columns during training and load it here
with open('model_columns4.pkl', 'rb') as f:
    model_columns = pickle.load(f)

input_encoded = pd.get_dummies(input_data , columns=categorical_cols, drop_first=True)
input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

# Predict
if st.button("Predict Yield"):
    prediction = model.predict(input_encoded)[0]
    st.success(f"🌟 Estimated Crop Yield: {prediction:.2f} tons")