# 🩺 Multiple Disease Prediction App

A Streamlit web application that predicts **Parkinson’s Disease, Diabetes, and Heart Disease** using machine learning models trained on Kaggle datasets.  
This project combines three individual disease prediction models into one unified app, making it easier to test multiple health conditions in a single interface.

Live Demo ---- https://multiple-disease-prediction-phegkcpafxc7qu2xfoujqm.streamlit.app/
---

## 📌 Project Overview
- Built with **Python, Streamlit, and scikit-learn**
- Uses three trained ML models:
  - Parkinson’s Disease Prediction
  - Diabetes Prediction
  - Heart Disease Prediction
- Interactive web UI for entering patient details
- Instant predictions with clear results

---

## 🚀 Features
- 🎛 **Sidebar selection** → choose which disease to predict  
- 📝 **User input forms** → enter patient details for each condition  
- ⚡ **Real-time prediction** → results displayed instantly  
- 📂 **Model integration** → loads pre-trained `.pkl` files for each disease  

---

## 🛠 Tech Stack
- **Python 3.9+**
- **Streamlit** → web app framework  
- **Pandas & NumPy** → data handling  
- **Scikit-learn** → ML models  
- **Joblib** → model persistence  

---

## 📦 Installation & Setup
Clone the repository and install dependencies:

'''bash
git clone https://github.com/your-username/multiple-disease-prediction.git
cd multiple-disease-prediction
pip install -r requirements.txt
'''


Repository Structure
├── app.py                # Streamlit app code
├── requirements.txt      # Dependencies
├── models/
│   ├── parkinson_model.sav
│   ├── diabetes_model.sav
│   └── heart_model.sav
├── notebooks/            # (Optional) Jupyter notebooks for training
└── README.md             # Project documentation



📊 Datasets
Parkinson’s dataset → Kaggle

Diabetes dataset → Kaggle

Heart disease dataset → Kaggle

