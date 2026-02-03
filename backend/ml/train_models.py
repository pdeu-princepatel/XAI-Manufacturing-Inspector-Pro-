#!/usr/bin/env python3
import sys, json, os
import joblib
from dataset import build_training_sets
from models import train_xgboost

# Ensure models directory exists
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

try:
    X, y = build_training_sets()
    model, metrics = train_xgboost(X, y)
    
    model_path = os.path.join(MODELS_DIR, 'xgboost.joblib')
    joblib.dump(model, model_path)
    
    # Express reads stdout
    print(json.dumps(metrics))
except Exception as e:
    # Print error to stderr so Node.js catches it
    print(f"Error: {str(e)}", file=sys.stderr)
    sys.exit(1)
