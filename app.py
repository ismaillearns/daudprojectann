import streamlit as st
import numpy as np
import pandas as pd
import random

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #f0f4f8; }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        font-size: 16px;
        padding: 10px 30px;
        border-radius: 8px;
        width: 100%;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        margin-top: 20px;
    }
    .diabetic { background-color: #fee2e2; color: #dc2626; border: 2px solid #dc2626; }
    .healthy { background-color: #dcfce7; color: #16a34a; border: 2px solid #16a34a; }
</style>
""", unsafe_allow_html=True)

st.title("🩺 Diabetes Prediction App")
st.write("Fill in the patient's biomedical data below and click **Predict**.")

# Input form
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin (μU/mL)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age = st.number_input("Age", min_value=1, max_value=120, value=25)

st.markdown("---")

# Simple prediction logic (dummy model for demo)
def predict_diabetes(pregnancies, glucose, bp, skin, insulin, bmi, dpf, age):
    # Simple risk scoring based on medical guidelines
    risk_score = 0
    
    if glucose > 140:
        risk_score += 2
    elif glucose > 120:
        risk_score += 1
    
    if bmi > 30:
        risk_score += 2
    elif bmi > 25:
        risk_score += 1
    
    if age > 50:
        risk_score += 1
    elif age > 35:
        risk_score += 0.5
    
    if pregnancies > 5:
        risk_score += 1
    
    if bp > 80:
        risk_score += 1
    
    if dpf > 0.8:
        risk_score += 1
    
    # Return probability based on risk score
    probability = min(0.95, risk_score / 8)
    return probability

if st.button("🔍 Predict"):
    # Calculate prediction
    probability = predict_diabetes(pregnancies, glucose, blood_pressure, 
                                   skin_thickness, insulin, bmi, dpf, age)
    
    # Show result
    if probability > 0.5:
        st.markdown(
            f'<div class="result-box diabetic">'
            f'⚠️ Diabetic &nbsp;|&nbsp; Risk: {probability:.0%}'
            f'</div>', unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-box healthy">'
            f'✅ Not Diabetic &nbsp;|&nbsp; Risk: {probability:.0%}'
            f'</div>', unsafe_allow_html=True
        )
    
    # Show input summary
    st.markdown("#### Input Summary")
    summary = pd.DataFrame({
        "Feature": ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness",
                    "Insulin", "BMI", "Diabetes Pedigree", "Age"],
        "Value": [pregnancies, glucose, blood_pressure, skin_thickness,
                  insulin, bmi, dpf, age]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Built with Streamlit · Diabetes Risk Assessment Tool")
