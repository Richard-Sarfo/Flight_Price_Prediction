from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
import numpy as np

app = Flask(__name__)

# Load Artifacts
ARTIFACTS_PATH = 'app_artifacts'

def load_artifacts():
    if not os.path.exists(ARTIFACTS_PATH):
        return None, None, None
    
    try:
        model = joblib.load(os.path.join(ARTIFACTS_PATH, 'flight_price_model.pkl'))
        encoders = joblib.load(os.path.join(ARTIFACTS_PATH, 'encoders.pkl'))
        scaler = joblib.load(os.path.join(ARTIFACTS_PATH, 'scaler.pkl'))
        return model, encoders, scaler
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return None, None, None

model, encoders, scaler = load_artifacts()

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Flight Price Prediction API is running!',
        'endpoints': {
            '/predict': 'POST - Predict flight price',
            '/health': 'GET - Check API health'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not encoders or not scaler:
        return jsonify({'error': 'Model artifacts not found. Please run train_and_save.py first.'}), 500

    try:
        data = request.get_json()
        
        # Expected features
        features = [
            'Airline', 'Source', 'Destination', 'Aircraft Type', 'Class', 
            'Booking Source', 'Seasonality', 'Stopovers', 
            'Base Fare (BDT)', 'Tax & Surcharge (BDT)', 'Days Before Departure'
        ]
        
        # Create DataFrame
        input_df = pd.DataFrame([data])
        
        # Check for missing columns
        missing_cols = [col for col in features if col not in input_df.columns]
        if missing_cols:
            return jsonify({'error': f'Missing columns: {missing_cols}'}), 400

        # Encode categorical features
        categorical_cols = ['Airline', 'Source', 'Destination', 'Aircraft Type', 'Class', 'Booking Source', 'Seasonality', 'Stopovers']
        
        for col in categorical_cols:
            if col in input_df.columns:
                le = encoders[col]
                # specific handling for unseen labels could be added here, 
                # but following app.py logic we assume valid input or let it error
                input_df[col] = le.transform(input_df[col])
        
        # Select and reorder columns
        input_df = input_df[features]
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        
        return jsonify({
            'predicted_total_fare': float(prediction),
            'currency': 'BDT'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
