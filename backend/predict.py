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
    
   
    expected_cols = ['Temperature', 'Pressure', 'Speed', 'Vibration', 'Humidity', 'Power_Consumption', 'Material_Hardness']
    
    # Map the incoming JSON payload smoothly whether it comes in as a standard dictionary or a flat list
    if isinstance(features_dict, dict):

        data_list = [features_dict.get(col, 0) for col in expected_cols]

        features_df = pd.DataFrame([features_dict])
    else:
        # If it comes as a raw list, we wrap it with our predefined sensor names
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
