from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import pandas as pd
import numpy as np
import io
import base64
import matplotlib
matplotlib.use('Agg') # Ensuring safe Matplotlib rendering on backend servers without a graphical display
import matplotlib.pyplot as plt
import seaborn as sns

from models import generate_shap_explanation
from dataset import build_training_sets

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for frontend deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, you can restrict this to your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

        # Mapping sensor data keys to more natural, human-readable names for the user interface
        friendly_names = {
            "Temperature":       "System Temperature",
            "Pressure":          "Internal Pressure",
            "Speed":             "Operation Speed",
            "Vibration":         "Vibration Level",
            "Humidity":          "Ambient Humidity",
            "Power_Consumption": "Power Consumption",
            "Material_Hardness": "Material Hardness",
        }

        # Defining easy-to-understand explanations for when each sensor's readings go above or below normal levels
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

        # --- Additional Key Performance Indicators (KPIs) tailored for the Power BI dashboard visualizations ---
        sorted_impacts = explanation["features"]
        top_risk_factor = sorted_impacts[0]["name"] if sorted_impacts else "N/A"
        risk_factors_count = int(sum(1 for f in sorted_impacts if f["direction"] == "negative"))
        # Calculating how confident the model is in its prediction, scaled as a clear percentage (0 to 100)
        # Converting numpy float types to standard Python floats so FastAPI can successfully return them as JSON
        model_confidence = float(round(float(abs(probability - 0.5)) * 200, 1))

        recs = []
        data = features.iloc[0]
        if data['Vibration'] > 4:
            recs.append("Inspect bearings for excessive wear and calibrate the spindle head.")
        if data['Temperature'] > 90:
            recs.append("Check internal cooling system immediately to prevent thermal expansion damage.")
        if data['Power_Consumption'] > 180:
            recs.append("Verify electrical load balancing and inspect the primary power unit.")
        if data['Pressure'] > 100:
            recs.append("Recalibrate internal pressure valves or inspect for potential line blockages.")
        if data['Speed'] > 2000:
            recs.append("Reduce operating speed to ease mechanical stress on moving parts.")
        if data['Humidity'] > 60:
            recs.append("Address excess moisture in the ambient environment to prevent corrosion.")
        if data['Material_Hardness'] > 75:
            recs.append("Change to a specialized cutting tool designed for harder materials.")

        if recs:
            explanation["recommendation"] = " · ".join(recs)
        elif result == "Pass":
            explanation["recommendation"] = "All systems are running within healthy parameters. Schedule your next routine check as planned."
            recs = ["Maintain current operating conditions.", "Proceed with the routine maintenance schedule."]

        # Pinpointing exactly which sensors, based on negative impacts, are contributing the most to increasing the failure risk
        weaknesses = [f["name"] for f in sorted_impacts if f["direction"] == "negative"]

        # Structuring a comprehensive, easy-to-read diagnostic breakdown of the machine's current state
        risk_pct = round(float(probability) * 100, 1)
        if result == "Fail":
            problem_text = f"The machine is operating outside safe parameters with a {risk_pct}% calculated failure risk, indicating imminent stress."
            cause_text = f"The primary driver(s) raising the failure risk are: {', '.join(weaknesses[:3])}. These sensors are detecting anomalous behavior."
            solution_text = " ".join(recs)
        else:
            problem_text = "The equipment is currently operating normally within stable historical baseline parameters."
            cause_text = "All sensor readings show nominal behavior. No critical stress factors are active."
            solution_text = " ".join(recs)
            
        detailed_summary = {
            "problem": problem_text,
            "cause": cause_text,
            "solution": solution_text
        }

        return {
            "status":              result,
            "failure_probability": float(probability),
            "prediction":          int(prediction),
            "explanation":         explanation,
            "model_confidence":    model_confidence,
            "top_risk_factor":     top_risk_factor,
            "risk_factors_count":  risk_factors_count,
            "weaknesses":          weaknesses[:3],
            "improvement_plan":    recs,
            "detailed_summary":    detailed_summary,
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

@app.get("/dataset-analytics")
def dataset_analytics():
    try:
        X, y = build_training_sets()
        data = X.copy()
        data['Failure'] = y

        plt.style.use('dark_background')

        # 1. Correlation Matrix
        fig_corr = plt.figure(figsize=(7, 5))
        corr = data.corr()
        sns.heatmap(corr, annot=True, cmap="YlOrRd", fmt=".2f",
                    cbar_kws={'label': 'Correlation Factor'}, 
                    annot_kws={"size": 8})
        plt.title("Sensor Correlation Heatmap", fontsize=12)
        plt.tight_layout()
        buf_corr = io.BytesIO()
        plt.savefig(buf_corr, format='png', facecolor='#161b22', edgecolor='none')
        buf_corr.seek(0)
        b64_corr = base64.b64encode(buf_corr.read()).decode('utf-8')
        plt.close(fig_corr)

        # 2. Temperature & Vibration Distribution vs Failure
        fig_dist, ax = plt.subplots(1, 2, figsize=(10, 4))
        sns.kdeplot(data=data, x="Temperature", hue="Failure", fill=True, common_norm=False, palette=["#2ea043", "#f85149"], ax=ax[0])
        ax[0].set_title("Temperature (0=Pass, 1=Fail)")
        
        sns.kdeplot(data=data, x="Vibration", hue="Failure", fill=True, common_norm=False, palette=["#2ea043", "#f85149"], ax=ax[1])
        ax[1].set_title("Vibration (0=Pass, 1=Fail)")
        
        plt.tight_layout()
        buf_dist = io.BytesIO()
        plt.savefig(buf_dist, format='png', facecolor='#161b22', edgecolor='none')
        buf_dist.seek(0)
        b64_dist = base64.b64encode(buf_dist.read()).decode('utf-8')
        plt.close(fig_dist)

        return {
            "correlation_plot": f"data:image/png;base64,{b64_corr}",
            "distribution_plot": f"data:image/png;base64,{b64_dist}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
