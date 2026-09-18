import streamlit as st
import pandas as pd
import joblib

# Load trained models
parkinson_model = joblib.load("parkinsons_model.sav")
diabetes_model = joblib.load("diabetes_model.sav")
heart_model = joblib.load("heartdisease_model.sav")

st.title("🩺 Multiple Disease Prediction App")
st.write("Predict Parkinson’s, Diabetes, or Heart Disease using ML models.")

# Sidebar for disease selection
disease_choice = st.sidebar.selectbox(
    "Choose Disease to Predict",
    ["Parkinson’s", "Diabetes", "Heart Disease"]
)

# Parkinson’s dataset features
if disease_choice == "Parkinson’s":
    st.header("Parkinson’s Prediction")
    fo = st.number_input("MDVP:Fo(Hz)", 80.0, 300.0, 120.0)
    fhi = st.number_input("MDVP:Fhi(Hz)", 100.0, 600.0, 200.0)
    flo = st.number_input("MDVP:Flo(Hz)", 50.0, 300.0, 100.0)
    jitter = st.number_input("MDVP:Jitter(%)", 0.0, 1.0, 0.1)
    shimmer = st.number_input("MDVP:Shimmer", 0.0, 1.0, 0.1)
    hnr = st.number_input("HNR", 0.0, 50.0, 20.0)
    rpde = st.number_input("RPDE", 0.0, 1.0, 0.5)
    dfa = st.number_input("DFA", 0.0, 1.0, 0.5)
    spread1 = st.number_input("Spread1", -10.0, 10.0, 0.0)
    spread2 = st.number_input("Spread2", -10.0, 10.0, 0.0)
    d2 = st.number_input("D2", 0.0, 3.0, 1.0)

    input_data = pd.DataFrame({
        "MDVP:Fo(Hz)": [fo],
        "MDVP:Fhi(Hz)": [fhi],
        "MDVP:Flo(Hz)": [flo],
        "MDVP:Jitter(%)": [jitter],
        "MDVP:Shimmer": [shimmer],
        "HNR": [hnr],
        "RPDE": [rpde],
        "DFA": [dfa],
        "spread1": [spread1],
        "spread2": [spread2],
        "D2": [d2]
    })

    if st.button("Predict Parkinson’s"):
        prediction = parkinsons_model.predict(input_data)[0]
        st.success("Parkinson’s Detected" if prediction == 1 else "Healthy")

# Diabetes dataset features
elif disease_choice == "Diabetes":
    st.header("Diabetes Prediction")
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose", 0, 200, 100)
    bp = st.number_input("BloodPressure", 0, 150, 70)
    skin = st.number_input("SkinThickness", 0, 100, 20)
    insulin = st.number_input("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("DiabetesPedigreeFunction", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 1, 120, 30)

    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [bp],
        "SkinThickness": [skin],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [dpf],
        "Age": [age]
    })

    if st.button("Predict Diabetes"):
        prediction = diabetes_model.predict(input_data)[0]
        st.success("Diabetic" if prediction == 1 else "Non‑Diabetic")

# Heart disease dataset features
elif disease_choice == "Heart Disease":
    st.header("Heart Disease Prediction")
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 400, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1])
    restecg = st.selectbox("Resting ECG (0-2)", [0,1,2])
    thalach = st.number_input("Max Heart Rate Achieved", 70, 210, 150)
    exang = st.selectbox("Exercise Induced Angina", [0,1])
    oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope (0-2)", [0,1,2])
    ca = st.selectbox("Number of Major Vessels (0-3)", [0,1,2,3])
    thal = st.selectbox("Thal (0=normal,1=fixed defect,2=reversible defect)", [0,1,2])

    input_data = pd.DataFrame({
        "age": [age],
        "sex": [1 if sex=="Male" else 0],
        "cp": [cp],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [fbs],
        "restecg": [restecg],
        "thalach": [thalach],
        "exang": [exang],
        "oldpeak": [oldpeak],
        "slope": [slope],
        "ca": [ca],
        "thal": [thal]
    })

    if st.button("Predict Heart Disease"):
        prediction = heartdisease_model.predict(input_data)[0]
        st.success("Heart Disease Detected" if prediction == 1 else "No Heart Disease")

