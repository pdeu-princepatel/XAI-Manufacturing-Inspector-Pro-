from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import pandas as pd
import numpy as np

from models import generate_shap_explanation
from dataset import build_training_sets

app = FastAPI()

class PredictionRequest(BaseModel):
    Temperature: float
    Pressure: float
    Speed: float
    Vibration: float
    Humidity: float
    Power_Consumption: float
    Material_Hardness: float

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'xgboost_classifier.joblib')
MODELS_DIR = os.path.dirname(MODEL_PATH)

X_TRAIN_SAMPLE = None

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

def get_training_sample():
    global X_TRAIN_SAMPLE
    if X_TRAIN_SAMPLE is None:
        X, _ = build_training_sets()
        X_TRAIN_SAMPLE = X.iloc[:100]
    return X_TRAIN_SAMPLE

@app.get("/")
def read_root():
    return {"message": "Manufacturing Inspector API is running"}

@app.post("/predict")
def predict_failure(request: PredictionRequest):
    model = load_model()
    if not model:
        raise HTTPException(status_code=500, detail="Model not found. Run python backend/train_models.py")
    
    features = pd.DataFrame([request.dict()])
    
    try:
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        result = "Fail" if prediction == 1 else "Pass"
        
        explanation = {"features": [], "recommendation": "Analysis complete"}

        # Human-readable labels for every feature
        friendly_names = {
            "Temperature":       "System Temperature",
            "Pressure":          "Internal Pressure",
            "Speed":             "Operation Speed",
            "Vibration":         "Vibration Level",
            "Humidity":          "Ambient Humidity",
            "Power_Consumption": "Power Consumption",
            "Material_Hardness": "Material Hardness",
        }

        # Plain-English reason strings per feature × direction
        reason_templates = {
            "Temperature":       ("High temperature stresses components", "Temperature is within safe range"),
            "Pressure":          ("Elevated pressure increases wear risk",  "Pressure is nominal"),
            "Speed":             ("Excessive speed causing mechanical stress", "Speed is within safe limits"),
            "Vibration":         ("High vibration signals bearing problems",   "Vibration is minimal"),
            "Humidity":          ("Moisture exposure risk detected",           "Humidity is acceptable"),
            "Power_Consumption": ("Abnormal power draw detected",             "Power draw is normal"),
            "Material_Hardness": ("Hard material increases tooling wear",      "Material hardness is fine"),
        }

        X_sample = get_training_sample()
        shap_result = generate_shap_explanation(model, X_sample, features)

        if shap_result:
            shap_values = shap_result['shap_values']
            names = shap_result['feature_names']

            impacts = []
            for name, raw_val in zip(names, shap_values):
                pct       = abs(raw_val) * 100
                direction = "negative" if raw_val > 0 else "positive"
                reasons   = reason_templates.get(name, ("Contributes to risk", "No concern"))
                reason    = reasons[0] if direction == "negative" else reasons[1]
                impacts.append({
                    "name":        friendly_names.get(name, name),
                    "key":         name,
                    "raw_shap":    round(float(raw_val), 4),
                    "impact":      round(pct, 2),
                    "direction":   direction,
                    "reason":      reason,
                })

            explanation["features"] = sorted(impacts, key=lambda x: x["impact"], reverse=True)

        recs = []
        data = features.iloc[0]
        if data['Vibration'] > 4:
            recs.append("Inspect bearings for excessive wear")
        if data['Temperature'] > 90:
            recs.append("Check cooling system immediately")
        if data['Power_Consumption'] > 180:
            recs.append("Verify electrical load balancing")

        if recs:
            explanation["recommendation"] = " · ".join(recs)
        elif result == "Pass":
            explanation["recommendation"] = "All systems are running within healthy parameters. Schedule your next routine check as planned."

        return {
            "status":              result,
            "failure_probability": float(probability),
            "prediction":          int(prediction),
            "explanation":         explanation,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
def train_model():
    try:
        global X_TRAIN_SAMPLE
        X, y = build_training_sets()
        from models import train_xgboost_classifier
        model, metrics, X_train, _ = train_xgboost_classifier(X, y)
        
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        X_TRAIN_SAMPLE = X_train
        
        return {"message": "Model trained", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

