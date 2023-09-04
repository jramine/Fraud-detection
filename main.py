import streamlit as st
import joblib
import tensorflow as tf

tf.keras.utils.get_custom_objects()['CustomAdam'] = CustomAdam

# Load the Keras model
model = tf.keras.models.load_model("fraud_detection_model.h5", custom_objects={'CustomAdam': CustomAdam})


# Load the scaler used during training
scaler = joblib.load("scaler.joblib")


# Streamlit interface
st.title("Fraud Detection")

# Input fields for user to enter values
distance_from_home = st.number_input("Distance from Home")
distance_from_last_transaction = st.number_input("Distance from Last Transaction")
ratio_to_median_purchase_price = st.number_input("Ratio to Median Purchase Price")
repeat_retailer = st.checkbox("Repeat Retailer")
used_chip = st.checkbox("Used Chip")
used_pin_number = st.checkbox("Used PIN Number")
online_order = st.checkbox("Online Order")

# Predict button
if st.button("Predict Fraud"):
    input_data = [
        distance_from_home, distance_from_last_transaction, ratio_to_median_purchase_price,
        float(repeat_retailer), float(used_chip), float(used_pin_number), float(online_order)
    ]
    
    input_data = [input_data]  # Reshape as a list of lists
    
    # Standardize input data using the loaded scaler
    input_data = scaler.transform(input_data)
    
    # Make a prediction
    prediction = model.predict(input_data)
    
    # Display the result
    if prediction >= 0.5:
        st.error("Prediction: Fraudulent Transaction")
    else:
        st.success("Prediction: Not a Fraudulent Transaction")
