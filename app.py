import joblib
import pandas as pd
import streamlit as st

art = joblib.load("churn_artifacts.joblib")
model, columns = art["model"], art["columns"]
options, threshold = art["options"], art["threshold"]

st.title("Prediksi Churn Pelanggan Telco")
st.caption("Model: Logistic Regression (class_weight=balanced)")

col1, col2 = st.columns(2)
inputs = {}

with col1:
    inputs["tenure"] = st.slider("Lama berlangganan (bulan)", 0, 72, 12)
    inputs["MonthlyCharges"] = st.number_input("Biaya bulanan", 18.0, 120.0, 70.0)
    inputs["SeniorCitizen"] = st.selectbox("Senior citizen (0=tidak, 1=ya)", [0, 1])

# TotalCharges diperkirakan dari tenure x biaya bulanan
inputs["TotalCharges"] = inputs["tenure"] * inputs["MonthlyCharges"]

for i, (c, opts) in enumerate(options.items()):
    with (col1 if i % 2 == 0 else col2):
        inputs[c] = st.selectbox(c, opts)

if st.button("Prediksi"):
    row = pd.DataFrame([inputs])
    row = pd.get_dummies(row, dtype=int).reindex(columns=columns, fill_value=0)
    proba = model.predict_proba(row)[:, 1][0]

    st.metric("Probabilitas churn", f"{proba:.1%}")
    if proba >= threshold:
        st.error("Risiko tinggi churn. Pertimbangkan penawaran retensi (mis. promo kontrak tahunan).")
    else:
        st.success("Risiko churn rendah.")