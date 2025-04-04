import streamlit as st
import sqlite3
import hashlib
import pickle
import os
import pandas as pd
import numpy as np
import sys

# Ensure feature_metadata module can be found
sys.path.append('/mount/src/healthcareprediction-1.app/')

try:
    from feature_metadata import feature_metadata  # Corrected module name
except ModuleNotFoundError:
    st.error("Error: feature_metadata module not found! Ensure it's in the correct directory.")

def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    if user and user[0] == hash_password(password):
        return True
    return False

st.set_page_config(page_title="Disease Prediction", page_icon="⚕️", layout="wide", initial_sidebar_state="collapsed")

st.title("Healthcare Prediction Using ML")

if 'menu' not in st.session_state:
    st.session_state['menu'] = "Login"

# Paths for dataset and trained models
feature_files = {
    "Diabetes": r"C:\Users\Dell\OneDrive\Desktop\Project\Project\datasets\diabetes_data.csv",
    "Heart Disease": r"C:\Users\Dell\OneDrive\Desktop\Project\Project\datasets\heart_disease_data.csv",
    "Parkinson's": r"C:\Users\Dell\OneDrive\Desktop\Project\Project\datasets\parkinson_data.csv",
    "Lung Cancer": r"C:\Users\Dell\OneDrive\Desktop\Project\Project\datasets\prepocessed_lungs_data.csv",
    "Thyroid": r"C:\Users\Dell\OneDrive\Desktop\Project\Project\datasets\hypothyroid.csv"
}

model_files = {
    "Diabetes": r"C:\\Project\\models minor\\-diabetes.sav",
    "Heart Disease": r"C:\\Project\\models minor\\heart_disease_prediction_analysis.sav",
    "Parkinson's": r"C:\\Project\\models minor\\par.sav",
    "Lung Cancer": r"C:\\Project\\models minor\\lung.sav",
    "Thyroid": r"C:\\Project\\models minor\\thyroid_disease_prediction_analysis.sav"
}   

def get_features(disease):
    file_path = feature_files.get(disease, None)
    if file_path and os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df.columns.tolist()[:-1]  # Exclude target column
    return []

def load_model(disease):
    model_path = model_files.get(disease, None)
    if model_path and os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    return None

if st.session_state['menu'] == "Sign Up":
    st.subheader("Create New Account")
    new_user = st.text_input("Username", key="signup_username")
    new_pass = st.text_input("Password", type="password", key="signup_password")
    if st.button("Sign Up"):
        if register_user(new_user, new_pass):
            st.success("Account created successfully! Go to Login.")
            st.session_state['menu'] = "Login"
            st.rerun()
        else:
            st.error("Username already exists. Choose a different one.")
    if st.button("Back to Login"):
        st.session_state['menu'] = "Login"
        st.rerun()

elif st.session_state['menu'] == "Login":
    st.subheader("Login to Your Account")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login"):
        if login_user(username, password):
            st.success(f"Welcome {username}!")
            st.session_state['menu'] = "Disease Prediction Page"
            st.rerun()
        else:
            st.error("Invalid username or password")
    if st.button("Create an Account"):
        st.session_state["menu"] = "Sign Up"
        st.rerun()

elif st.session_state['menu'] == "Disease Prediction Page":
    st.subheader("Select Disease for Prediction")
    disease = st.selectbox("Choose a Disease", list(feature_files.keys()))
    
    if disease:
        st.subheader(f"Enter Patient Details for {disease} Prediction")
        features = get_features(disease)
        user_inputs = {}

        for feature in features:
            if feature == "sex":
                selected_gender = st.selectbox("Sex", options=["Male", "Female"])
                user_inputs['sex'] = 1 if selected_gender == "Male" else 0
            else:
                min_val, max_val, default_val = 0, 100, 0
                if disease in feature_metadata and feature in feature_metadata[disease]:
                    min_val, max_val, default_val = feature_metadata[disease][feature]

                if isinstance(default_val, int):
                    user_inputs[feature] = st.number_input(
                        feature, min_value=min_val, max_value=max_val, value=default_val, step=1
                    ) 
                else:
                    user_inputs[feature] = st.number_input(
                        feature, min_value=min_val, max_value=max_val, value=default_val, step=0.1, format="%.2f"
                    )   

        if st.button("Predict Disease"):
            model = load_model(disease)
            if model:
                input_data = np.array([[user_inputs[feature] for feature in features]]).reshape(1, -1)
                prediction = model.predict(input_data)
                if prediction[0] == 1:
                    st.error(f"The patient has {disease}.")
                else:
                    st.success(f"The patient does NOT have {disease}.")
            else:
                st.error("Error: Model not found!")

        if st.button("Logout"):
            st.session_state['menu'] = "Login"
            st.rerun()

