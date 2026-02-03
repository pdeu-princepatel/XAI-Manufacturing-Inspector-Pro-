from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import pandas as pd
import numpy as np

# Adjust path to find modules if needed, or rely on running from backend/
# Assuming running `uvicorn main:app` from `backend/`
from ml.models import train_xgboost
from ml.dataset import build_training_sets, generate_manufacturing_data

app = FastAPI()

# Data Model for Prediction
class PredictionRequest(BaseModel):
    Temperature: float
    Pressure: float
    Speed: float
    Vibration: float
    Humidity: float
    Power_Consumption: float
    Material_Hardness: float

# Load Model (Global variable, loaded on startup or request)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'xgboost.joblib')
MODELS_DIR = os.path.dirname(MODEL_PATH)

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

@app.get("/")
def read_root():
    return {"message": "Manufacturing Inspector API is running"}

@app.post("/predict")
def predict_failure(request: PredictionRequest):
    model = load_model()
    if not model:
        raise HTTPException(status_code=500, detail="Model not found. Please train the model first.")
    
    # Create DataFrame for prediction
    features = pd.DataFrame([request.dict()])
    
    # Predict
    try:
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1] # Probability of class 1 (Failure)
        
        result = "Fail" if prediction == 1 else "Pass"
        
        # Determine strictness/quality message based on probability ? 
        # For now just return raw values
        
        return {
            "status": result,
            "failure_probability": float(probability),
            "prediction": int(prediction)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
def train_model():
    try:
        # Ensure data exists or generate it
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'manufacturing_data.csv')
        if not os.path.exists(data_path):
             generate_manufacturing_data()
        
        # Load data (modified build_training_sets needs to know where to look or we pass it)
        # Note: We need to ensure build_training_sets looks at the right CSV
        X, y = build_training_sets(path=data_path)
        
        model, metrics = train_xgboost(X, y)
        
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        
        return {
            "message": "Model trained successfully",
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
