import numpy as np
import joblib
import os
def generate_text_insights(df):
    print("\n--- INSIGHTS GENERATION ---")
    
    # 1. Aviation/Route Pricing
    print("\n[Airline Pricing Strategy]")
    if 'Airline' in df.columns and 'Total Fare (BDT)' in df.columns:
        airline_avg = df.groupby('Airline')['Total Fare (BDT)'].mean().sort_values(ascending=False)
        print("Average Fare by Airline (Descending):")
        print(airline_avg)
        print("\nMost Expensive Airline:", airline_avg.index[0])
        print("Cheapest Airline:", airline_avg.index[-1])
    
    # 2. Seasonality
    print("\n[Seasonality Trends]")
    if 'Seasonality' in df.columns and 'Total Fare (BDT)' in df.columns:
        season_avg = df.groupby('Seasonality')['Total Fare (BDT)'].mean().sort_values(ascending=False)
        print("Average Fare by Seasonality:")
        print(season_avg)

    # 3. Class Impact
    print("\n[Class Pricing]")
    if 'Class' in df.columns and 'Total Fare (BDT)' in df.columns:
        class_avg = df.groupby('Class')['Total Fare (BDT)'].mean().sort_values(ascending=False)
        print("Average Fare by Class:")
        print(class_avg)
        
    # 4. Correlation Analysis
    print("\n[Feature Correlations with Total Fare]")
    numeric_df = df.select_dtypes(include=[np.number])
    if 'Total Fare (BDT)' in numeric_df.columns:
        corr = numeric_df.corr()['Total Fare (BDT)'].sort_values(ascending=False)
        print("Top 5 Positive Correlations:")
        print(corr.head(6)) # Include target itself
        print("\nTop 5 Negative Correlations:")
        print(corr.tail(5))

def feature_importance_analysis():
    print("\n[Model Feature Importance]")
    model_path = 'app_artifacts/flight_price_model.pkl'
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Skipping importance analysis.")
        return

    try:
        model = joblib.load(model_path)
        # Based on train_and_save.py columns
        feature_cols = ['Airline', 'Source', 'Destination', 'Aircraft Type', 'Class', 'Booking Source', 'Seasonality', 'Stopovers', 'Base Fare (BDT)', 'Tax & Surcharge (BDT)', 'Days Before Departure']
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print("Feature Importance Ranking:")
            for i in range(len(feature_cols)):
                if i < len(indices): # Safety check
                    idx = indices[i]
                    if idx < len(feature_cols):
                        print(f"{i+1}. {feature_cols[idx]}: {importances[idx]:.4f}")
        else:
            print("Model does not expose feature_importances_.")
            
    except Exception as e:
        print(f"Error analyzing model: {e}")
