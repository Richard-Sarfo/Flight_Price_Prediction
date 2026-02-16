import requests
import json

# Define the API endpoint
url = 'http://127.0.0.1:5000/predict'

# Sample input data (Values taken from actual dataset to avoid encoding errors)
data = {
    "Airline": "US-Bangla Airlines",
    "Source": "DAC",
    "Destination": "DEL",
    "Aircraft Type": "Boeing 787",
    "Class": "First Class",
    "Booking Source": "Direct Booking",
    "Seasonality": "Regular",
    "Stopovers": "Direct",
    "Base Fare (BDT)": 116951.65,
    "Tax & Surcharge (BDT)": 19542.75,
    "Days Before Departure": 41
}

print(f"Sending request to {url} with data:")
print(json.dumps(data, indent=2))

# Send POST request
try:
    response = requests.post(url, json=data)
    
    # Check response status
    if response.status_code == 200:
        print("\nSuccess!")
        print("Response:", response.json())
    else:
        print(f"\nError: {response.status_code}")
        print("Response:", response.text)

except requests.exceptions.ConnectionError:
    print("\nError: Could not connect to the API. Please ensure 'api.py' is running.")
