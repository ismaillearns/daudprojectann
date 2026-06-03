import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")

st.title("🩺 Diabetes Prediction App")

# Create a simple model on the fly if no file exists
@st.cache_resource
def get_model():
    try:
        scaler = pickle.load(open("scaler.pkl", "rb"))
        model = pickle.load(open("model.pkl", "rb"))
        return model, scaler
    except:
        # Create dummy model if files don't exist
        return None, None

model, scaler = get_model()

# Input form
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", value=1)
    glucose = st.number_input("Glucose", value=120)
    blood_pressure = st.number_input("Blood Pressure", value=70)
    skin_thickness = st.number_input("Skin Thickness", value=20)

with col2:
    insulin = st.number_input("Insulin", value=80)
    bmi = st.number_input("BMI", value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", value=0.5)
    age = st.number_input("Age", value=25)

if st.button("Predict"):
    if model is None:
        st.error("⚠️ Model files missing. Please run training first.")
    else:
        input_data = np.array([[pregnancies, glucose, blood_pressure, 
                                skin_thickness, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]
        
        if pred == 1:
            st.error("⚠️ DIABETIC")
        else:
            st.success("✅ NOT DIABETIC")
