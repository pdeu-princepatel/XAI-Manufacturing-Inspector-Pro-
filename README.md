# XAI Manufacturing Inspector Pro

> An Explainable AI (XAI) system for automated surface defect detection in manufacturing — providing transparent, human-readable decision-making for quality assurance.

---

## 🧠 What It Does

XAI Manufacturing Inspector Pro takes real-time sensor readings from industrial machines and predicts whether a machine is likely to **pass or fail** quality inspection. More importantly, it explains *why* — using SHAP and LIME to break down which sensor readings are driving the risk, and providing actionable maintenance recommendations.

---

## ✨ Features

- **Real-time failure prediction** — XGBoost classifier trained on 7 sensor inputs
- **Explainable AI (XAI)** — SHAP TreeExplainer + LIME for transparent predictions
- **Sensor impact analysis** — Ranked feature contributions with directional reasoning
- **Actionable recommendations** — Specific maintenance guidance based on sensor thresholds
- **Dataset analytics** — Correlation heatmap & distribution plots (Matplotlib + Seaborn)
- **Interactive dashboard** — D3.js & Chart.js powered frontend with KPI cards
- **Multiple ML models** — Linear/Logistic Regression, Random Forest, Decision Tree, XGBoost

---

## 🗂️ Project Structure

```
XAI-Manufacturing-Inspector-Pro/
├── backend/
│   ├── main.py              # FastAPI app — prediction, training & analytics endpoints
│   ├── models.py            # ML model training + SHAP/LIME explanation functions
│   ├── dataset.py           # Synthetic sensor dataset generation
│   ├── predict.py           # Standalone prediction utilities
│   ├── train_models.py      # Script to train and save the XGBoost model
│   ├── inspect_data.py      # Data inspection helpers
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   └── server.js        # Express.js server
│   ├── views/               # EJS templates
│   ├── public/
│   │   └── js/
│   │       └── charts.js    # D3.js / Chart.js dashboard visualisations
│   └── package.json
├── models/
│   └── xgboost_classifier.joblib   # Saved trained model (generated after training)
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| **ML / XAI Backend** | Python, FastAPI, XGBoost, scikit-learn, SHAP, LIME |
| **Data & Analytics** | Pandas, NumPy, Matplotlib, Seaborn |
| **Frontend Server** | Node.js, Express.js |
| **Templating** | EJS |
| **Charts / Visuals** | Chart.js, D3.js |
| **API Communication** | Axios |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+

---

### 1. Backend Setup

```bash
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Train the XGBoost model (saves to models/xgboost_classifier.joblib)
python train_models.py

# Start the FastAPI server on port 8000
uvicorn main:app --reload --port 8000
```

The API will be available at: `http://localhost:8000`

---

### 2. Frontend Setup

```bash
cd frontend

# Install Node.js dependencies
npm install express ejs axios body-parser chart.js

# Start the Express server
npm start
```

The dashboard will be available at: `http://localhost:3000`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Predict failure from sensor readings |
| `POST` | `/train` | Re-train the XGBoost model on fresh data |
| `GET` | `/dataset-analytics` | Returns correlation heatmap & distribution plots (base64 PNG) |

### `/predict` — Request Body

```json
{
  "Temperature": 85.0,
  "Pressure": 95.0,
  "Speed": 1800.0,
  "Vibration": 3.2,
  "Humidity": 45.0,
  "Power_Consumption": 160.0,
  "Material_Hardness": 65.0
}
```

### `/predict` — Response

```json
{
  "status": "Fail",
  "failure_probability": 0.87,
  "prediction": 1,
  "model_confidence": 74.0,
  "top_risk_factor": "Vibration Level",
  "risk_factors_count": 3,
  "weaknesses": ["Vibration Level", "System Temperature", "Internal Pressure"],
  "improvement_plan": ["Inspect bearings for excessive wear..."],
  "explanation": {
    "features": [...],
    "recommendation": "..."
  },
  "detailed_summary": {
    "problem": "...",
    "cause": "...",
    "solution": "..."
  }
}
```

---

## 🔬 Sensor Thresholds

| Sensor | Risk Threshold | Risk Reason |
|---|---|---|
| Temperature | > 90 °C | Thermal expansion / cooling failure |
| Vibration | > 4.0 | Bearing wear / spindle misalignment |
| Power Consumption | > 180 W | Electrical fault / overload |
| Pressure | > 100 bar | Valve blockage / line leak |
| Speed | > 2000 RPM | Mechanical stress on moving parts |
| Humidity | > 60 % | Corrosion / moisture ingress risk |
| Material Hardness | > 75 | Excessive tooling wear |

---

## 🤖 XAI Explainability

- **SHAP (SHapley Additive exPlanations):** Uses `TreeExplainer` to attribute each sensor's contribution to the final prediction, returning a ranked list of feature impacts with direction (positive/negative).
- **LIME (Local Interpretable Model-agnostic Explanations):** Fits a locally linear model around the prediction point to produce intuitive, instance-level explanations.

---

## 📊 Analytics

The `/dataset-analytics` endpoint returns two charts:

1. **Sensor Correlation Heatmap** — shows which sensors correlate with each other and with failure
2. **Temperature & Vibration KDE Distribution** — overlaid density plots comparing Pass vs Fail distributions

---

## 📄 License

This project was developed as a Web Technology semester project (Sem 6).
