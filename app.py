import streamlit as st
import pandas as pd
import joblib

# Load trained models
parkinson_model = joblib.load("parkinson_model.pkl")
diabetes_model = joblib.load("diabetes_model.pkl")
heart_model = joblib.load("heart_model.pkl")

st.title("🩺 Multiple Disease Prediction App")
st.write("Predict Parkinson’s, Diabetes, or Heart Disease from user inputs.")

# Sidebar for disease selection
disease_choice = st.sidebar.selectbox(
    "Choose Disease to Predict",
    ["Parkinson’s", "Diabetes", "Heart Disease"]
)

# Input forms depending on disease
if disease_choice == "Parkinson’s":
    st.header("Parkinson’s Prediction")
    # Example inputs (replace with your dataset features)
    mdvp_fo = st.number_input("MDVP:Fo(Hz)", 80.0, 300.0, 120.0)
    jitter = st.number_input("Jitter(%)", 0.0, 1.0, 0.1)
    shimmer = st.number_input("Shimmer", 0.0, 1.0, 0.1)

    input_data = pd.DataFrame({
        "MDVP:Fo(Hz)": [mdvp_fo],
        "Jitter(%)": [jitter],
        "Shimmer": [shimmer]
    })

    if st.button("Predict Parkinson’s"):
        prediction = parkinson_model.predict(input_data)[0]
        st.success(f"Prediction: {'Parkinson’s Detected' if prediction==1 else 'Healthy'}")

elif disease_choice == "Diabetes":
    st.header("Diabetes Prediction")
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose", 0, 200, 100)
    bmi = st.number_input("BMI", 0.0, 60.0, 25.0)

    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BMI": [bmi]
    })

    if st.button("Predict Diabetes"):
        prediction = diabetes_model.predict(input_data)[0]
        st.success(f"Prediction: {'Diabetic' if prediction==1 else 'Non‑Diabetic'}")

elif disease_choice == "Heart Disease":
    st.header("Heart Disease Prediction")
    age = st.slider("Age", 20, 100, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [0,1,2,3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 400, 200)

    input_data = pd.DataFrame({
        "age": [age],
        "sex": [1 if sex=="Male" else 0],
        "cp": [cp],
        "trestbps": [trestbps],
        "chol": [chol]
    })

    if st.button("Predict Heart Disease"):
        prediction = heart_model.predict(input_data)[0]
        st.success(f"Prediction: {'Heart Disease Detected' if prediction==1 else 'No Heart Disease'}")
