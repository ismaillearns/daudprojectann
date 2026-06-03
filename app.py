import streamlit as st
import numpy as np
import os
import pickle

# Page config
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

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
        border: none;
        width: 100%;
    }
    .stButton>button:hover { background-color: #1d4ed8; }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        margin-top: 20px;
    }
    .diabetic   { background-color: #fee2e2; color: #dc2626; border: 2px solid #dc2626; }
    .healthy    { background-color: #dcfce7; color: #16a34a; border: 2px solid #16a34a; }
</style>
""", unsafe_allow_html=True)

# ── Load model & scaler ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        import tensorflow as tf
        model  = tf.keras.models.load_model("diabetes_model.h5")
        scaler = pickle.load(open("scaler.pkl", "rb"))
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)

model, scaler, load_error = load_artifacts()

# ── UI ───────────────────────────────────────────────────────────────────────
st.title("🩺 Diabetes Prediction App")
st.write("Fill in the patient's biomedical data below and click **Predict**.")

if load_error:
    st.warning(
        "⚠️ Model files not found. Please follow the **Setup Instructions** below "
        "to generate `diabetes_model.h5` and `scaler.pkl` from your Colab notebook, "
        "then place them in the same folder as this app."
    )
    with st.expander("📋 Setup Instructions"):
        st.markdown("""
**In your Google Colab notebook, add these lines after training:**

```python
import pickle

# Save the trained model
model.save('diabetes_model.h5')

# Save the scaler
pickle.dump(scaler, open('scaler.pkl', 'wb'))
```

Then download both files from Colab:
- `Files` panel (left sidebar) → right-click each file → **Download**

Place them in the same folder as `app.py`, then restart the app.
        """)

st.markdown("---")

# Input form
col1, col2 = st.columns(2)

with col1:
    pregnancies    = st.number_input("Pregnancies",              min_value=0,   max_value=20,  value=1)
    glucose        = st.number_input("Glucose Level (mg/dL)",   min_value=0,   max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)",  min_value=0,   max_value=150, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)",     min_value=0,   max_value=100, value=20)

with col2:
    insulin        = st.number_input("Insulin (μU/mL)",          min_value=0,   max_value=900, value=80)
    bmi            = st.number_input("BMI",                      min_value=0.0, max_value=70.0,value=25.0, step=0.1)
    dpf            = st.number_input("Diabetes Pedigree Function",min_value=0.0,max_value=3.0, value=0.5, step=0.01)
    age            = st.number_input("Age",                      min_value=1,   max_value=120, value=25)

st.markdown("---")

if st.button("🔍 Predict"):
    if model is None or scaler is None:
        st.error("Model not loaded. Please follow the setup instructions above.")
    else:
        input_data   = np.array([[pregnancies, glucose, blood_pressure,
                                   skin_thickness, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)
        prediction   = model.predict(input_scaled)[0][0]

        if prediction > 0.5:
            st.markdown(
                f'<div class="result-box diabetic">'
                f'⚠️ Diabetic &nbsp;|&nbsp; Confidence: {prediction:.0%}'
                f'</div>', unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-box healthy">'
                f'✅ Not Diabetic &nbsp;|&nbsp; Confidence: {1 - prediction:.0%}'
                f'</div>', unsafe_allow_html=True
            )

        st.markdown("#### Input Summary")
        import pandas as pd
        summary = pd.DataFrame({
            "Feature": ["Pregnancies","Glucose","Blood Pressure","Skin Thickness",
                        "Insulin","BMI","Diabetes Pedigree","Age"],
            "Value":   [pregnancies, glucose, blood_pressure, skin_thickness,
                        insulin, bmi, dpf, age]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Built with Streamlit · ANN trained on PIMA Indians Diabetes Dataset")
