import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")
st.title("Heart Disease Prediction System")
st.write("Enter patient details to estimate whether heart disease is likely.")

art = joblib.load("heart_model.pkl")
model, enc, features = art["model"], art["encoders"], art["features"]

age = st.slider("Age", 20, 90, 50)
sex = st.selectbox("Sex", list(enc["sex"].classes_))
cp = st.selectbox("Chest pain type", list(enc["cp"].classes_))
thalach = st.slider("Maximum heart rate achieved", 70, 210, 150)

if st.button("Predict"):
    row = pd.DataFrame([{
        "age": age,
        "sex": enc["sex"].transform([sex])[0],
        "cp": enc["cp"].transform([cp])[0],
        "thalach": thalach,
    }])[features]
    pred = model.predict(row)[0]
    prob = model.predict_proba(row)[0][1]
    if pred == 1:
        st.error(f"Heart disease likely (probability {prob:.0%})")
    else:
        st.success(f"Heart disease unlikely (probability {prob:.0%})")
    st.caption("Educational project, not medical advice.")
