import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Create artifacts directory
if not os.path.exists('app_artifacts'):
    os.makedirs('app_artifacts')

print("Loading dataset...")
df = pd.read_csv("c:/Users/RichardAnaneSarfo/Desktop/ML/Dataset/Flight_Price_Dataset_of_Bangladesh.csv")

# --- Preprocessing (Matching Notebook Logic) ---
print("Preprocessing data...")

# Drop duplicates
df = df.drop_duplicates()

# Datetime conversion (though mostly unused in final model, good for consistency)
df['Departure Date & Time'] = pd.to_datetime(df['Departure Date & Time'])
df['Arrival Date & Time'] = pd.to_datetime(df['Arrival Date & Time'])
df['Departure_date'] = df['Departure Date & Time'].dt.date
df['Departure_time'] = df['Departure Date & Time'].dt.time
df['Arrival_date'] = df['Arrival Date & Time'].dt.date
df['Arrival_time'] = df['Arrival Date & Time'].dt.time

# Drop unused columns
df = df.drop(['Source Name', 'Destination Name', 'Departure Date & Time', 'Arrival Date & Time'], axis=1)

# Categorical columns to encode
categorical_cols = ['Airline', 'Source', 'Destination', 'Aircraft Type', 'Class', 'Booking Source', 'Seasonality', 'Stopovers']

# Dictionary to store encoders
encoders = {}

print("Encoding categorical variables...")
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Save encoders
joblib.dump(encoders, 'app_artifacts/encoders.pkl')

# Define Features and Target
# Note: Based on notebook Step 13, these are the selected features
feature_cols = ['Airline', 'Source', 'Destination', 'Aircraft Type', 'Class', 'Booking Source', 'Seasonality', 'Stopovers', 'Base Fare (BDT)', 'Tax & Surcharge (BDT)', 'Days Before Departure']
X = df[feature_cols]
y = df["Total Fare (BDT)"]

# Split data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
print("Scaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, 'app_artifacts/scaler.pkl')

# Train Model
print("Training Random Forest model (this may take a moment)...")
# Using Random Forest as it was the selected model in the notebook analysis
model = RandomForestRegressor(n_estimators=100, random_state=42) 
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"Model Training Completed. R2 Score: {score:.4f}")

# Save Model
print("Saving model...")
joblib.dump(model, 'app_artifacts/flight_price_model.pkl')
print("All artifacts saved to 'app_artifacts/' directory.")
