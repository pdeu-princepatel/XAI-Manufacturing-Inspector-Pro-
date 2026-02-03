#!/usr/bin/env python3
import sys, json, joblib, os
import numpy as np
import pandas as pd

try:
    if len(sys.argv) < 3:
        raise ValueError("Usage: predict.py <model_name> <features_json>")

    model_name = sys.argv[1]
    features_json = sys.argv[2]
    features_dict = json.loads(features_json)
    
    # Ensure correct feature order (XGBoost is picky about column order usually, 
    # but for simple array input we need to be careful. DataFrame is safer if we know cols.)
    # For this demo, we'll assume features_dict comes in correct order or create dataframe.
    # To be safe, let's match the columns from dataset.py: Temperature, Pressure, Vibration, RPM
    # Or just convert values to list if features_dict is actually just a dict
    
    # Ideally we'd pickle the column names too. For now let's trust the input.
    # Convert dict values to list for prediction
    
    # Expected keys for DataFrame construction
    expected_cols = ['Temperature', 'Pressure', 'Speed', 'Vibration', 'Humidity', 'Power_Consumption', 'Material_Hardness']
    
    # If input is a list, usage is direct. If dict, convert.
    if isinstance(features_dict, dict):
        # Fill missing with 0 or handle error
        data_list = [features_dict.get(col, 0) for col in expected_cols]
        # Create DataFrame to preserve feature names for SHAP and XGBoost
        features_df = pd.DataFrame([features_dict])
    else:
        # Assuming list
        features_df = pd.DataFrame([features_dict], columns=expected_cols)

    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', f'{model_name}.joblib')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    
    pred = model.predict(features_df)
    proba = model.predict_proba(features_df).tolist()
    
    print(json.dumps({
        'prediction': int(pred[0]), 
        'probabilities': proba[0]
    }))

except Exception as e:
    print(str(e), file=sys.stderr)
    sys.exit(1)
