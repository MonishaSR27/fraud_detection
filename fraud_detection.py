import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import joblib

model = joblib.load("fraud_detection_pipeline.pkl")

st.image("fraud_logo.PNG", width=100)

st.title("SafeTransfer: Real-time Fraud Detection Tool")

st.markdown("Please enter the transaction details and use the predict button")

st.divider()

transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"])
amount = st.number_input("Amount", min_value = 0.0, value=1000.0)
oldbalanceOrg = st.number_input("Old balance (Sender)", min_value = 0.0, value = 1000.0)
newbalanceOrig = st.number_input("New balance (Sender)", min_value = 0.0, value = 9000.0)
oldbalanceDest = st.number_input("Old balance (Receiver)", min_value = 0.0, value = 0.0)
newbalanceDest = st.number_input("New balance (Receiver)", min_value = 0.0, value = 0.0)

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "type" : transaction_type,
        "amount" : amount,
        "oldbalanceOrg" : oldbalanceOrg,
        "newbalanceOrig" : newbalanceOrig,
        "oldbalanceDest" : oldbalanceDest,
        "newbalanceDest" : newbalanceDest
    }])

    st.write("### Transaction Summary")
    st.write(input_data)


    prediction = model.predict(input_data)[0]

    st.subheader(f"Prediction : '{int(prediction)}'")

    if prediction == 1:
        st.error("This transaction can be fraud")
    else:
        st.success("This transaction looks like it is not a fraud")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)[0]
        confidence = proba[1] if prediction == 1 else proba[0]
        st.info(f"Model Confidence: {confidence * 100:.2f}%")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)[0]
        fig, ax = plt.subplots()
        ax.bar(["Not Fraud", "Fraud"], proba, color=["green", "red"])
        ax.set_ylabel("Probability")
        ax.set_title("Fraud Probability")
        st.pyplot(fig)
    




