import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Set page config
st.set_page_config(page_title="Flight Price Prediction", layout="centered")

st.title("✈️ Flight Price Prediction App")
st.markdown("Enter the flight details below to predict the total fare.")

# Load Artifacts
@st.cache_resource
def load_artifacts():
    artifacts_path = 'app_artifacts'
    if not os.path.exists(artifacts_path):
        st.error(f"Artifacts directory '{artifacts_path}' not found. Please run 'train_and_save.py' first.")
        return None, None, None
    
    try:
        model = joblib.load(os.path.join(artifacts_path, 'flight_price_model.pkl'))
        encoders = joblib.load(os.path.join(artifacts_path, 'encoders.pkl'))
        scaler = joblib.load(os.path.join(artifacts_path, 'scaler.pkl'))
        return model, encoders, scaler
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None, None

model, encoders, scaler = load_artifacts()

if model and encoders and scaler:
    # Input Form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        # Categorical Inputs
        # We need to access the classes_ from the encoders to show options
        
        with col1:
            airline = st.selectbox("Airline", options=encoders['Airline'].classes_)
            source = st.selectbox("Source", options=encoders['Source'].classes_)
            destination = st.selectbox("Destination", options=encoders['Destination'].classes_)
            aircraft_type = st.selectbox("Aircraft Type", options=encoders['Aircraft Type'].classes_)
            flight_class = st.selectbox("Class", options=encoders['Class'].classes_)
            
        with col2:
            booking_source = st.selectbox("Booking Source", options=encoders['Booking Source'].classes_)
            seasonality = st.selectbox("Seasonality", options=encoders['Seasonality'].classes_)
            stopovers = st.selectbox("Stopovers", options=encoders['Stopovers'].classes_)
            days_before = st.number_input("Days Before Departure", min_value=0, max_value=365, value=1)
        
        # Numerical Inputs (that are strictly features)
        st.subheader("Fare Components")
        st.info("Note: The model uses Base Fare and Tax as features to predict Total Fare.")
        
        col3, col4 = st.columns(2)
        with col3:
            base_fare = st.number_input("Base Fare (BDT)", min_value=0.0, value=1000.0)
        with col4:
            tax_surcharge = st.number_input("Tax & Surcharge (BDT)", min_value=0.0, value=100.0)

        submitted = st.form_submit_button("Predict Fare")
        
        if submitted:
            # Prepare input data
            input_data = {
                'Airline': airline,
                'Source': source,
                'Destination': destination,
                'Aircraft Type': aircraft_type,
                'Class': flight_class,
                'Booking Source': booking_source,
                'Seasonality': seasonality,
                'Stopovers': stopovers,
                'Base Fare (BDT)': base_fare,
                'Tax & Surcharge (BDT)': tax_surcharge,
                'Days Before Departure': days_before
            }
            
            # Create DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical features
            categorical_cols = ['Airline', 'Source', 'Destination', 'Aircraft Type', 'Class', 'Booking Source', 'Seasonality', 'Stopovers']
            
            for col in categorical_cols:
                le = encoders[col]
                input_df[col] = le.transform(input_df[col])
            
            # Scale features
            # The scaler expects all features including encoded ones
            # Order must match training: 
            # ['Airline', 'Source', 'Destination', 'Aircraft Type', 'Class', 'Booking Source', 'Seasonality', 'Stopovers', 'Base Fare (BDT)', 'Tax & Surcharge (BDT)', 'Days Before Departure']
            
            feature_cols = ['Airline', 'Source', 'Destination', 'Aircraft Type', 'Class', 'Booking Source', 'Seasonality', 'Stopovers', 'Base Fare (BDT)', 'Tax & Surcharge (BDT)', 'Days Before Departure']
            
            # Reorder columns to ensure match
            input_df = input_df[feature_cols]
            
            # Scale
            input_scaled = scaler.transform(input_df)
            
            # Predict
            prediction = model.predict(input_scaled)[0]
            
            st.success(f"Predicted Total Fare: {prediction:,.2f} BDT")
            
            # Optional: Show breakdown or difference
            calculated_total = base_fare + tax_surcharge
            diff = prediction - calculated_total
            if abs(diff) > 1:
                st.write(f"Difference from (Base + Tax): {diff:,.2f} BDT")
