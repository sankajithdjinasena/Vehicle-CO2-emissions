# app.py
from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# --- Load the Model and Feature Lists ---
model = None
feature_lists = {}
try:
    model = joblib.load('best_model.pkl')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("ERROR: best_model.pkl not found. Run prepare_data.py first.")

try:
    feature_lists = joblib.load('feature_lists.pkl')
    print("Feature lists loaded successfully.")
except FileNotFoundError:
    print("ERROR: feature_lists.pkl not found. Run prepare_data.py first.")

# Define all required features for DataFrame creation
FEATURE_NAMES = [
    'Brand', 'Model', 'Vehicle Class', 'Engine Size(L)', 'Cylinders',
    'Transmission', 'Fuel Type', 'Fuel Consumption City (L/100 km)',
    'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)',
    'Fuel Consumption Comb (mpg)'
]

@app.route('/', methods=['GET'])
def home():
    """Renders the main form for input, passing categorical options."""
    # Pass the feature lists to the template
    return render_template('index.html', features=feature_lists)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the form."""
    if model is None:
        return render_template('index.html', features=feature_lists, prediction_result="Error: Model not loaded.")

    try:
        form_data = request.form.to_dict()
        
        # Prepare input_data (same logic as before, ensuring types are correct)
        input_data = {}
        for feature in FEATURE_NAMES:
            value = form_data.get(feature, None)
            
            # Simplified type handling (ensure numerical values are float)
            if feature in ['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)', 
                            'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)', 
                            'Fuel Consumption Comb (mpg)']:
                # Convert to float, defaulting to 0.0 if value is missing/empty
                input_data[feature] = float(value) if value else 0.0
            else:
                input_data[feature] = value if value else ''
        
        input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
        
        prediction = model.predict(input_df)[0]
        result = f"Predicted CO2 Emissions(g/km): {prediction:.2f}"
        
        # Pass feature lists back along with the result and form data
        return render_template('index.html', prediction_result=result, 
                                form_data=input_data, features=feature_lists)

    except ValueError as e:
        return render_template('index.html', prediction_result=f"Error: Invalid input data. Details: {e}", 
                                form_data=request.form.to_dict(), features=feature_lists)
    except Exception as e:
        return render_template('index.html', prediction_result=f"An unexpected error occurred: {e}", 
                                form_data=request.form.to_dict(), features=feature_lists)

