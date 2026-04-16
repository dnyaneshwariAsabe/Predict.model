import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(page_title="Diabetes Prediction Pro", layout="centered")

# Custom CSS for animations and styling
st.markdown("""
    <style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .main {
        animation: fadeIn 0.8s ease-out;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        background-color: #4CAF50;
        color: white;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

st.title("🩺 Health Indicator Predictor")
st.write("Enter the patient details below to assess the health risk.")

# Create two columns for a better layout
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=100)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
    age = st.number_input("Age", min_value=0, max_value=120, value=30)

# Prediction Logic
if st.button("Analyze Results"):
    # Prepare the input data
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                            insulin, bmi, dpf, age]])
    
    prediction = model.predict(input_data)
    
    st.divider()
    
    if prediction[0] == 1:
        st.error("### Risk Detected: Positive")
        st.write("The model suggests a high probability of health risk. Please consult a professional.")
    else:
        st.success("### Risk Detected: Negative")
        st.write("The model suggests health indicators are within a standard range.")

st.info("Disclaimer: This tool is for educational purposes and is not a substitute for medical advice.")
